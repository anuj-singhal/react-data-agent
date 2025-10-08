# services/rag_agent/graph_context_retriever.py
"""Graph-based RAG Context Retriever using JSON schema files."""
import os
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ============== Data Models ==============
@dataclass
class TableSchema:
    """Schema definition from JSON."""
    table_name: str
    description: str
    ddl: str
    columns: Dict[str, Dict[str, Any]]  # column_name -> {type, description, nullable, etc.}
    primary_key: List[str] = field(default_factory=list)
    indexes: List[Dict[str, Any]] = field(default_factory=list)
    file_path: Optional[str] = None
    last_modified: Optional[float] = None
    checksum: Optional[str] = None


@dataclass
class Relationship:
    """Relationship definition from JSON."""
    name: str
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    relationship_type: str  # ONE_TO_ONE, ONE_TO_MANY, MANY_TO_MANY
    description: Optional[str] = None


@dataclass
class GraphContext:
    """Retrieved context with graph relationships."""
    primary_tables: List[str]
    related_tables: List[str]
    columns: Dict[str, List[str]]  # table -> relevant columns
    relationships: List[Relationship]
    ddl_snippets: Dict[str, str]  # table -> DDL
    data_dictionary: Dict[str, Dict[str, str]]  # table -> column -> description
    relationship_paths: List[List[str]]  # Paths through the graph
    confidence_scores: Dict[str, float]


# ============== JSON Schema Loader ==============
class JSONSchemaLoader:
    """Loads and monitors JSON schema files."""
    
    def __init__(self, schema_dir: str = "./data/schema_definitions"):
        self.schema_dir = Path(schema_dir)
        self.schema_cache: Dict[str, TableSchema] = {}
        self.relationships: List[Relationship] = []
        self.file_checksums: Dict[str, str] = {}
        
    def get_file_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file."""
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_schemas(self) -> Tuple[List[TableSchema], List[Relationship]]:
        """Load all schema JSON files from directory."""
        if not self.schema_dir.exists():
            logger.warning(f"Schema directory {self.schema_dir} does not exist")
            return [], []
        
        tables = []
        relationships = []
        
        # Process each JSON file
        for json_file in self.schema_dir.glob("*.json"):
            try:
                current_checksum = self.get_file_checksum(json_file)
                
                # Skip if file hasn't changed
                if (json_file.name in self.file_checksums and 
                    self.file_checksums[json_file.name] == current_checksum):
                    logger.debug(f"Skipping unchanged file: {json_file.name}")
                    # Still add cached version to results
                    if json_file.stem in self.schema_cache:
                        tables.append(self.schema_cache[json_file.stem])
                    continue
                
                # Load and parse JSON
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Process based on file type
                if 'table_name' in data:  # Table schema file
                    table = self._parse_table_schema(data, json_file)
                    tables.append(table)
                    self.schema_cache[table.table_name] = table
                    logger.info(f"Loaded table schema: {table.table_name}")
                    
                elif 'relationships' in data:  # Relationships file
                    rels = self._parse_relationships(data)
                    relationships.extend(rels)
                    logger.info(f"Loaded {len(rels)} relationships from {json_file.name}")
                
                # Update checksum
                self.file_checksums[json_file.name] = current_checksum
                
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")
        
        self.relationships = relationships
        return tables, relationships
    
    def _parse_table_schema(self, data: Dict, file_path: Path) -> TableSchema:
        """Parse table schema from JSON data."""
        return TableSchema(
            table_name=data['table_name'],
            description=data.get('description', ''),
            ddl=data.get('ddl', ''),
            columns=data.get('columns', {}),
            primary_key=data.get('primary_key', []),
            indexes=data.get('indexes', []),
            file_path=str(file_path),
            last_modified=file_path.stat().st_mtime,
            checksum=self.get_file_checksum(file_path)
        )
    
    def _parse_relationships(self, data: Dict) -> List[Relationship]:
        """Parse relationships from JSON data."""
        relationships = []
        for rel_data in data.get('relationships', []):
            rel = Relationship(
                name=rel_data.get('name', ''),
                from_table=rel_data['from_table'],
                from_column=rel_data['from_column'],
                to_table=rel_data['to_table'],
                to_column=rel_data['to_column'],
                relationship_type=rel_data.get('type', 'ONE_TO_MANY'),
                description=rel_data.get('description')
            )
            relationships.append(rel)
        return relationships
    
    def needs_reindexing(self, file_path: str) -> bool:
        """Check if a file needs reindexing."""
        path = Path(file_path)
        if not path.exists():
            return False
        
        current_checksum = self.get_file_checksum(path)
        return (path.name not in self.file_checksums or 
                self.file_checksums[path.name] != current_checksum)


# ============== Graph-based RAG Retriever ==============
class GraphRAGRetriever:
    """Graph-based context retriever with automatic embedding updates."""
    
    def __init__(
        self,
        schema_dir: str = "./data/schema_definitions",
        embedding_provider=None,
        vector_store=None
    ):
        self.schema_loader = JSONSchemaLoader(schema_dir)
        self.embedding_provider = embedding_provider or self._get_default_embedder()
        self.vector_store = vector_store or self._get_default_store()
        
        # Graph structures
        self.table_graph: Dict[str, Set[str]] = {}  # table -> connected tables
        self.column_graph: Dict[str, Dict[str, Set[str]]] = {}  # table -> column -> related columns
        
        # Auto-load on initialization
        self._auto_index_schemas()
    
    def _get_default_embedder(self):
        """Get default embedding provider."""
        try:
            # Try OpenAI first
            from config.settings import settings
            api_key = getattr(settings, 'OPENAI_API_KEY', None)
            if api_key:
                try:
                    from openai import AsyncOpenAI
                    # Use inline OpenAI provider
                    class InlineOpenAIProvider:
                        def __init__(self, api_key):
                            self.client = AsyncOpenAI(api_key=api_key)
                            self.model = "text-embedding-3-large"
                        
                        async def create_embedding(self, text: str):
                            response = await self.client.embeddings.create(
                                model=self.model,
                                input=text
                            )
                            return response.data[0].embedding
                        
                        async def create_embeddings(self, texts: list):
                            response = await self.client.embeddings.create(
                                model=self.model,
                                input=texts
                            )
                            return [item.embedding for item in response.data]
                    
                    return InlineOpenAIProvider(api_key)
                except ImportError:
                    pass
        except:
            pass
        
        # Fallback to local sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            
            class InlineLocalProvider:
                def __init__(self):
                    self.model = SentenceTransformer("all-MiniLM-L6-v2")
                
                async def create_embedding(self, text: str):
                    embedding = self.model.encode(text)
                    return embedding.tolist()
                
                async def create_embeddings(self, texts: list):
                    embeddings = self.model.encode(texts)
                    return embeddings.tolist()
            
            return InlineLocalProvider()
        except ImportError:
            logger.warning("No embedding provider available (install sentence-transformers)")
            return None
    
    def _get_default_store(self):
        """Get default vector store."""
        try:
            import chromadb
            
            class InlineChromaStore:
                def __init__(self, collection_name="oracle_graph_schema"):
                    self.client = chromadb.PersistentClient(path="./vector_db/chroma_db")
                    self.collection_name = collection_name
                    try:
                        self.collection = self.client.get_collection(collection_name)
                    except:
                        self.collection = self.client.create_collection(
                            name=collection_name,
                            metadata={"hnsw:space": "cosine"}
                        )
                
                async def add_documents(self, documents, embeddings, ids):
                    metadatas = []
                    for doc in documents:
                        metadata = {}
                        for key, value in doc.items():
                            if isinstance(value, (str, int, float, bool)):
                                metadata[key] = value
                            else:
                                metadata[key] = json.dumps(value)
                        metadatas.append(metadata)
                    
                    self.collection.add(
                        embeddings=embeddings,
                        metadatas=metadatas,
                        ids=ids
                    )
                
                async def query(self, query_embedding, k=5):
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=k
                    )
                    
                    documents = []
                    if results['metadatas'] and results['distances']:
                        for metadata, distance in zip(results['metadatas'][0], results['distances'][0]):
                            doc = {}
                            for key, value in metadata.items():
                                if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                    try:
                                        doc[key] = json.loads(value)
                                    except:
                                        doc[key] = value
                                else:
                                    doc[key] = value
                            similarity = 1 - distance
                            documents.append((doc, similarity))
                    
                    return documents
                
                async def delete_collection(self):
                    self.client.delete_collection(self.collection_name)
                    self.collection = None
                
                async def exists(self):
                    try:
                        self.client.get_collection(self.collection_name)
                        return True
                    except:
                        return False
            
            return InlineChromaStore()
            
        except ImportError:
            logger.warning("ChromaDB not available (install chromadb)")
            return None
    
    def _auto_index_schemas(self):
        """Automatically index new or changed schema files."""
        try:
            tables, relationships = self.schema_loader.load_schemas()
            
            if not tables and not relationships:
                logger.info("No schema files to index")
                return
            
            # Build graph structure
            self._build_graph(tables, relationships)
            
            # Index new or changed tables
            new_tables = []
            for table in tables:
                if self.schema_loader.needs_reindexing(table.file_path):
                    new_tables.append(table)
            
            if new_tables:
                logger.info(f"Indexing {len(new_tables)} new/changed tables")
                self._index_tables(new_tables)
            
            logger.info(f"Graph RAG ready with {len(tables)} tables and {len(relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Error in auto-indexing: {e}")
    
    def _build_graph(self, tables: List[TableSchema], relationships: List[Relationship]):
        """Build graph structure from tables and relationships."""
        # Initialize table graph
        for table in tables:
            if table.table_name not in self.table_graph:
                self.table_graph[table.table_name] = set()
        
        # Build connections from relationships
        for rel in relationships:
            # Add bidirectional connection
            self.table_graph[rel.from_table].add(rel.to_table)
            self.table_graph[rel.to_table].add(rel.from_table)
            
            # Build column graph
            if rel.from_table not in self.column_graph:
                self.column_graph[rel.from_table] = {}
            if rel.from_column not in self.column_graph[rel.from_table]:
                self.column_graph[rel.from_table][rel.from_column] = set()
            
            self.column_graph[rel.from_table][rel.from_column].add(
                f"{rel.to_table}.{rel.to_column}"
            )
    
    def _index_tables(self, tables: List[TableSchema]):
        """Index tables into vector store."""
        if not self.embedding_provider or not self.vector_store:
            logger.warning("Cannot index: embedding provider or vector store not available")
            return
        
        documents = []
        texts = []
        ids = []
        
        for table in tables:
            # Create searchable text
            text_parts = [
                f"Table: {table.table_name}",
                f"Description: {table.description}"
            ]
            
            # Add column information
            for col_name, col_info in table.columns.items():
                text_parts.append(f"Column {col_name}: {col_info.get('description', col_info.get('type', ''))}")
            
            # Create document
            doc = {
                "type": "table",
                "table_name": table.table_name,
                "description": table.description,
                "columns": list(table.columns.keys()),
                "ddl": table.ddl[:500] if table.ddl else "",  # Store truncated DDL
                "checksum": table.checksum
            }
            
            documents.append(doc)
            texts.append('\n'.join(text_parts))
            ids.append(f"table_{table.table_name}")
        
        try:
            import asyncio
            # Create embeddings
            loop = asyncio.get_event_loop()
            embeddings = loop.run_until_complete(
                self.embedding_provider.create_embeddings(texts)
            )
            
            # Store in vector database
            loop.run_until_complete(
                self.vector_store.add_documents(documents, embeddings, ids)
            )
            
            logger.info(f"Indexed {len(tables)} tables")
            
        except Exception as e:
            logger.error(f"Error indexing tables: {e}")
    
    async def retrieve_context(
        self,
        intent: Dict[str, Any],
        user_query: str = "",
        k: int = 3,
        depth: int = 1
    ) -> GraphContext:
        """
        Retrieve context based on intent using graph traversal.
        
        Args:
            intent: Semantic intent from Intent Agent
            k: Number of primary tables to retrieve
            depth: Graph traversal depth for related tables
            
        Returns:
            GraphContext with relevant schema information
        """
        # Reload schemas to check for updates
        self._auto_index_schemas()
        
        # Build query from intent
        #query_text = self._build_query_from_intent(intent)
        query_text = self._build_query(intent)
        
        # Get primary tables from vector search
        primary_tables = await self._get_primary_tables(query_text, k)
        
        # Traverse graph for related tables
        related_tables = self._traverse_graph(primary_tables, depth)
        
        # Build comprehensive context
        context = self._build_context(primary_tables, related_tables, intent)
        
        logger.info(f"Retrieved context: {len(context.primary_tables)} primary, {len(context.related_tables)} related tables")
        
        return context
    
    def _build_query(self, intent) -> str:
        """Build search query from intent."""
        parts = []
        
        if intent.get('suggested_approach'):
            parts.append(f"Intent suggestion : {intent['suggested_approach']}")
        
        #parts.append(f"Query: {user_query}")
        
        return ' '.join(parts)

    # def _build_query_from_intent(self, intent: Dict[str, Any]) -> str:
    #     """Build search query from intent."""
    #     parts = []
        
    #     if intent.get('primary_intent'):
    #         parts.append(f"Intent: {intent['primary_intent']}")
        
    #     if intent.get('domain'):
    #         parts.append(f"Domain: {intent['domain']}")
        
    #     if intent.get('metrics'):
    #         parts.append(f"Metrics: {', '.join(intent['metrics'])}")
        
    #     if intent.get('dimensions'):
    #         parts.append(f"Dimensions: {', '.join(intent['dimensions'])}")
        
    #     if intent.get('entities'):
    #         for entity_type, values in intent['entities'].items():
    #             parts.append(f"{entity_type}: {', '.join(values)}")
        
    #     return ' '.join(parts)
    
    async def _get_primary_tables(self, query: str, k: int) -> List[str]:
        """Get primary tables from vector search."""
        if not self.embedding_provider or not self.vector_store:
            # Fallback: return all cached tables
            return list(self.schema_loader.schema_cache.keys())[:k]
        
        try:
            # Create query embedding
            query_embedding = await self.embedding_provider.create_embedding(query)
            
            # Search vector store
            results = await self.vector_store.query(query_embedding, k=k)
            
            # Extract table names
            tables = []
            for doc, score in results:
                if doc.get('type') == 'table':
                    tables.append(doc['table_name'])
            
            return tables
            
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return list(self.schema_loader.schema_cache.keys())[:k]
    
    def _traverse_graph(self, start_tables: List[str], depth: int) -> List[str]:
        """Traverse graph to find related tables."""
        if depth <= 0:
            return []
        
        related = set()
        visited = set(start_tables)
        current_level = set(start_tables)
        
        for _ in range(depth):
            next_level = set()
            for table in current_level:
                if table in self.table_graph:
                    for connected_table in self.table_graph[table]:
                        if connected_table not in visited:
                            next_level.add(connected_table)
                            related.add(connected_table)
                            visited.add(connected_table)
            
            current_level = next_level
            if not current_level:
                break
        
        return list(related)
    
    def _build_context(
        self,
        primary_tables: List[str],
        related_tables: List[str],
        intent: Dict[str, Any]
    ) -> GraphContext:
        """Build comprehensive context from tables and intent."""
        all_tables = primary_tables + related_tables
        
        # Extract relevant columns based on intent
        columns = {}
        for table in all_tables:
            if table in self.schema_loader.schema_cache:
                schema = self.schema_loader.schema_cache[table]
                relevant_cols = self._get_relevant_columns(schema, intent)
                if relevant_cols:
                    columns[table] = relevant_cols
        
        # Get DDL snippets
        ddl_snippets = {}
        data_dictionary = {}
        for table in all_tables:
            if table in self.schema_loader.schema_cache:
                schema = self.schema_loader.schema_cache[table]
                ddl_snippets[table] = schema.ddl[:1000] if schema.ddl else ""
                data_dictionary[table] = {
                    col: info.get('description', info.get('type', ''))
                    for col, info in schema.columns.items()
                }
        
        # Get relevant relationships
        relevant_relationships = []
        for rel in self.schema_loader.relationships:
            if rel.from_table in all_tables and rel.to_table in all_tables:
                relevant_relationships.append(rel)
        
        # Find relationship paths
        paths = self._find_relationship_paths(primary_tables, intent)
        
        return GraphContext(
            primary_tables=primary_tables,
            related_tables=related_tables,
            columns=columns,
            relationships=relevant_relationships,
            ddl_snippets=ddl_snippets,
            data_dictionary=data_dictionary,
            relationship_paths=paths,
            confidence_scores={table: 1.0 for table in primary_tables}
        )
    
    def _get_relevant_columns(self, schema: TableSchema, intent: Dict[str, Any]) -> List[str]:
        """Extract relevant columns based on intent."""
        relevant = []
        
        # Get metrics columns
        if intent.get('metrics'):
            for metric in intent['metrics']:
                for col in schema.columns:
                    if metric.lower() in col.lower():
                        relevant.append(col)
        
        # Get dimension columns
        if intent.get('dimensions'):
            for dim in intent['dimensions']:
                for col in schema.columns:
                    if dim.lower() in col.lower():
                        relevant.append(col)
        
        # Add primary key columns
        relevant.extend(schema.primary_key)
        
        # If no specific columns found, return key columns
        if not relevant:
            # Return primary key and first few columns
            relevant = schema.primary_key + list(schema.columns.keys())[:5]
        
        return list(set(relevant))  # Remove duplicates
    
    def _find_relationship_paths(self, tables: List[str], intent: Dict[str, Any]) -> List[List[str]]:
        """Find paths through the graph connecting tables."""
        paths = []
        
        # Simple path finding for joins
        if len(tables) > 1 and intent.get('requires_join'):
            for i, table1 in enumerate(tables):
                for table2 in tables[i+1:]:
                    path = self._find_path(table1, table2)
                    if path:
                        paths.append(path)
        
        return paths
    
    def _find_path(self, start: str, end: str, max_depth: int = 3) -> Optional[List[str]]:
        """Find path between two tables in the graph."""
        if start == end:
            return [start]
        
        visited = set()
        queue = [(start, [start])]
        
        while queue and len(visited) < max_depth * 10:
            current, path = queue.pop(0)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current in self.table_graph:
                for neighbor in self.table_graph[current]:
                    if neighbor == end:
                        return path + [neighbor]
                    
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return None


# ============== Singleton Instance ==============
graph_rag_retriever = GraphRAGRetriever()