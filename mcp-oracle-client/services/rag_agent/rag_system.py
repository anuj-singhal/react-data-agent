"""
RAG System - Handles all retrieval and context building
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime
from .rag_config import (
    RAGConfig, DataLoader, Table, Relationship, QueryHistory
)


class RAGSystem:
    """Complete RAG system for schema search and context building"""
    
    def __init__(self, 
                 data_dict_path: str = None,
                 relationships_path: str = None,
                 query_history_path: str = None,
                 use_sample_values: bool = True):
        
        self.config = RAGConfig()
        self.use_sample_values = use_sample_values
        
        # Use default paths if not provided
        self.data_dict_path = data_dict_path or self.config.DEFAULT_DATA_DICT_PATH
        self.relationships_path = relationships_path or self.config.DEFAULT_RELATIONSHIPS_PATH
        self.query_history_path = query_history_path or self.config.DEFAULT_QUERY_HISTORY_PATH
        
        
    def initialize_rag(self):
        # Initialize RAG components
        self._initialize_components()
        self._load_data()
        self._index_data()
        return True

    def _initialize_components(self):
        """Initialize RAG components"""
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=self.config.CHROMA_DB_PATH,
            settings=Settings(**self.config.CHROMA_SETTINGS)
        )
        
        # Create collections
        self.table_collection = self.chroma_client.get_or_create_collection(
            name="table_metadata",
            metadata={"hnsw:space": "cosine"}
        )
        self.column_collection = self.chroma_client.get_or_create_collection(
            name="column_metadata",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Query history
        self.query_history = []
        self.query_embeddings = None
    
    def _load_data(self):
        """Load all data from JSON files"""
        print("Loading schema data...")
        self.tables = DataLoader.load_data_dictionary(self.data_dict_path)
        self.relationships = DataLoader.load_relationships(self.relationships_path)
        self.query_history = DataLoader.load_query_history(self.query_history_path)
        
        # Build graph
        self._build_graph()
        
        # Store tables dict for quick access
        self.tables_dict = {table.table_name: table for table in self.tables}
        
        print(f"Loaded {len(self.tables)} tables, {len(self.relationships)} relationships")
    
    def _build_graph(self):
        """Build relationship graph"""
        # Add nodes
        for table in self.tables:
            self.graph.add_node(table.table_name, data=table)
        
        # Add edges
        for rel in self.relationships:
            self.graph.add_edge(
                rel.parent_table,
                rel.child_table,
                relationship=rel
            )
    
    def _index_data(self):
        """Index schema and query history"""
        print("Indexing schema...")
        self._index_schema()
        
        print("Indexing query history...")
        self._index_query_history()
    
    def _index_schema(self):
        """Index tables and columns for semantic search"""
        table_texts = []
        table_metadatas = []
        table_ids = []
        
        column_texts = []
        column_metadatas = []
        column_ids = []
        
        for table in self.tables:
            # Table indexing
            table_text = table.to_text()
            table_texts.append(table_text)
            table_metadatas.append({
                "table_name": table.table_name,
                "description": table.table_description
            })
            table_ids.append(table.table_name)
            
            # Column indexing
            for col in table.columns:
                samples_text = f" (Examples: {', '.join(str(v) for v in col.sample_values[:3])})" if col.sample_values else ""
                column_text = f"{table.table_name}.{col.column_name}: {col.description}{samples_text}"
                
                column_texts.append(column_text)
                column_metadatas.append({
                    "table_name": table.table_name,
                    "column_name": col.column_name,
                    "data_type": col.data_type
                })
                column_ids.append(f"{table.table_name}.{col.column_name}")
        
        # Clear and add to collections
        self._update_collection(self.table_collection, table_texts, table_metadatas, table_ids)
        self._update_collection(self.column_collection, column_texts, column_metadatas, column_ids)
    
    def _update_collection(self, collection, texts, metadatas, ids):
        """Update ChromaDB collection"""
        if texts:
            try:
                existing_ids = collection.get()['ids']
                if existing_ids:
                    collection.delete(ids=existing_ids)
            except:
                pass
            
            embeddings = self.embedding_model.encode(texts).tolist()
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
    
    def _index_query_history(self):
        """Index historical queries for similarity matching"""
        if not self.query_history:
            return
        
        all_texts = []
        for query in self.query_history:
            texts = [query.natural_language] + query.variations
            all_texts.append(" | ".join(texts))
        
        if all_texts:
            self.query_embeddings = self.embedding_model.encode(all_texts)
    
    def search_relevant_tables(self, query: str) -> Dict[str, Any]:
        """Search for relevant tables and columns"""
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Search tables
        table_results = self.table_collection.query(
            query_embeddings=query_embedding,
            n_results=min(self.config.TOP_K_TABLES, len(self.tables_dict))
        )
        
        # Search columns
        column_results = self.column_collection.query(
            query_embeddings=query_embedding,
            n_results=self.config.TOP_K_COLUMNS
        )
        
        # Extract relevant tables
        relevant_tables = set()
        if table_results['ids']:
            relevant_tables.update(table_results['ids'][0])
        
        if column_results['metadatas']:
            for metadata_list in column_results['metadatas']:
                for metadata in metadata_list:
                    if 'table_name' in metadata:
                        relevant_tables.add(metadata['table_name'])
        
        # Expand with related tables
        expanded_tables = self._expand_with_relationships(relevant_tables)
        
        return {
            'tables': list(expanded_tables),
            'table_results': table_results,
            'column_results': column_results,
            'sample_values': self._get_sample_values(expanded_tables)
        }
    
    def _expand_with_relationships(self, tables: Set[str]) -> Set[str]:
        """Expand tables with their relationships"""
        expanded = set(tables)
        
        for table in tables:
            if table in self.graph:
                # Add related tables
                try:
                    # Get neighbors within depth
                    for depth in range(1, self.config.MAX_RELATIONSHIP_DEPTH + 1):
                        neighbors = nx.single_source_shortest_path_length(
                            self.graph, table, cutoff=depth
                        ).keys()
                        expanded.update(neighbors)
                    
                    # Add ancestors
                    ancestors = nx.ancestors(self.graph, table)
                    expanded.update(ancestors)
                except:
                    pass
        
        return expanded
    
    def _get_sample_values(self, tables: Set[str]) -> Dict[str, Dict]:
        """Get sample values for tables"""
        sample_values = {}
        for table_name in tables:
            if table_name in self.tables_dict:
                table = self.tables_dict[table_name]
                sample_values[table_name] = {
                    col.column_name: col.sample_values
                    for col in table.columns
                    if col.sample_values
                }
        return sample_values
    
    def find_similar_query(self, query: str) -> Optional[Tuple[QueryHistory, float]]:
        """Find similar query in history"""
        if not self.query_history or self.query_embeddings is None:
            return None
        
        query_embedding = self.embedding_model.encode([query])
        similarities = np.dot(self.query_embeddings, query_embedding.T).flatten()
        
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= self.config.SIMILARITY_THRESHOLD:
            matched_query = self.query_history[best_idx]
            matched_query.usage_count += 1
            matched_query.last_used = datetime.now().isoformat()
            return matched_query, best_score
        
        return None
    
    def add_query_to_history(self, nl_query: str, sql_query: str, tables_used: List[str]):
        """Add new query to history"""
        new_query = QueryHistory(
            id=f"q_{len(self.query_history) + 1:03d}",
            natural_language=nl_query,
            variations=[],
            sql_query=sql_query,
            tables_used=tables_used,
            query_type="generated",
            performance_score=0.0,
            usage_count=1,
            last_used=datetime.now().isoformat()
        )
        
        self.query_history.append(new_query)
        DataLoader.save_query_history(self.query_history_path, self.query_history)
        
        # Re-index
        self._index_query_history()
        
        return new_query
    
    def build_context(self, tables: List[str], user_query: str) -> Dict[str, Any]:
        """Build context for LLM"""
        context = {
            'user_query': user_query,
            'tables': {},
            'relationships': [],
            'sample_values': {}
        }
        
        # Add table information
        for table_name in tables:
            if table_name in self.tables_dict:
                table = self.tables_dict[table_name]
                
                table_info = {
                    'description': table.table_description,
                    'columns': []
                }
                
                for col in table.columns:
                    col_info = {
                        'name': col.column_name,
                        'type': col.data_type,
                        'description': col.description,
                        'is_pk': col.is_primary_key
                    }
                    if col.sample_values:
                        col_info['samples'] = col.sample_values[:3]
                    table_info['columns'].append(col_info)
                
                context['tables'][table_name] = table_info
                
                # Add sample values
                if self.use_sample_values:
                    context['sample_values'][table_name] = {
                        col.column_name: col.sample_values
                        for col in table.columns
                        if col.sample_values
                    }
        
        # Add relationships
        for rel in self.relationships:
            if rel.parent_table in tables and rel.child_table in tables:
                context['relationships'].append({
                    'from': f"{rel.parent_table}.{rel.parent_column}",
                    'to': f"{rel.child_table}.{rel.child_column}",
                    'type': rel.relationship_type,
                    'description': rel.description
                })
        
        return context
    
# Singleton instance
rag_agent = RAGSystem()