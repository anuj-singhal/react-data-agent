"""
RAG System - Handles all retrieval and context building
Enhanced with query history similarity matching
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Set, Optional, Tuple
from datetime import datetime
from colorama import Fore, Style
from .rag_config import (
    RAGConfig, DataLoader, Table, Relationship, QueryHistory
)
import logging

logger = logging.getLogger(__name__)


class RAGSystem:
    """Complete RAG system for schema search and context building"""
    
    def __init__(self, 
                 data_dict_path: str = None,
                 relationships_path: str = None,
                 query_history_path: str = None,
                 business_rules_path: str = None,
                 sql_rules_path: str = None,
                 use_sample_values: bool = True):
        
        self.config = RAGConfig()
        self.use_sample_values = use_sample_values
        
        # Use default paths if not provided
        self.data_dict_path = data_dict_path or self.config.DEFAULT_DATA_DICT_PATH
        self.relationships_path = relationships_path or self.config.DEFAULT_RELATIONSHIPS_PATH
        self.query_history_path = query_history_path or self.config.DEFAULT_QUERY_HISTORY_PATH
        self.business_rules_path = business_rules_path or self.config.DEFAULT_BUSINESS_RULES_PATH
        self.sql_rules_path = sql_rules_path or self.config.DEFAULT_SQL_RULES_PATH
        
    def initialize_rag(self):
        # Initialize RAG components
        self._initialize_components()
        self._load_data()
        self._index_data()
        return True
        # # Initialize components
        # self._initialize_components()
        # self._load_data()
        # self._index_data()
    
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
        self.business_rules = DataLoader.load_rules(self.business_rules_path, "business_rules")
        self.sql_rules = DataLoader.load_rules(self.sql_rules_path, "sql_rules")
        
        # Build graph
        self._build_graph()
        
        # Store tables dict for quick access
        self.tables_dict = {table.table_name: table for table in self.tables}
        
        print(f"Loaded {len(self.tables)} tables, {len(self.relationships)} relationships, {len(self.query_history)} historical queries")
        print(f"Loaded {len(self.business_rules)} business rules, {len(self.sql_rules)} SQL Rules")

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
        
        # Create separate embeddings for each query and its variations
        self.query_texts = []
        self.query_embeddings_list = []
        self.query_mapping = []  # Maps embedding index to query index
        
        for query_idx, query in enumerate(self.query_history):
            # Add natural language
            self.query_texts.append(query.natural_language)
            self.query_mapping.append(query_idx)
            
            # Add each variation
            for variation in query.variations:
                self.query_texts.append(variation)
                self.query_mapping.append(query_idx)
        
        if self.query_texts:
            print(f"  Indexing {len(self.query_texts)} query texts from {len(self.query_history)} queries")
            self.query_embeddings = self.embedding_model.encode(self.query_texts)
    
       
    def check_query_history(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check if a similar query exists in history
        """
        try:
            # Look for similar queries with 90% threshold
            similar = self._find_similar_query_enhanced(query, threshold=0.90)
            return similar
        except Exception as e:
            logger.error(f"Query history check failed: {e}")
            return None
    


    def _find_similar_query_enhanced(self, query: str, threshold: float = 0.90) -> Optional[Dict[str, Any]]:
        """
        Enhanced query similarity matching with 90% threshold
        Returns the matched query with similarity score
        """
        if not self.query_history or self.query_embeddings is None:
            logger.info("No query history available for matching")
            return None
        
        print(f"\n{Fore.BLUE}🔍 Searching query history...{Style.RESET_ALL}")
        
        # Encode the input query
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate similarities with all historical queries (including variations)
        similarities = np.dot(self.query_embeddings, query_embedding.T).flatten()
        
        # Find best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        # Get the actual query (not variation) index
        query_idx = self.query_mapping[best_idx]
        matched_text = self.query_texts[best_idx]
        
        print(f"  Best match: {best_score:.2%} similarity")
        print(f"  Matched text: '{matched_text[:50]}...'")
        print(f"  Threshold: {threshold:.2%}")
        
        # Check if similarity exceeds threshold
        if best_score >= threshold:
            matched_query = self.query_history[query_idx]
            
            print(f"{Fore.GREEN}  ✓ Match found! Query ID: {matched_query.id}{Style.RESET_ALL}")
            logger.info(f"Found similar query with {best_score:.2%} similarity")
            
            return {
                'matched_query': matched_query,
                'similarity_score': float(best_score),
                'query_id': matched_query.id,
                'natural_language': matched_query.natural_language,
                'sql_query': matched_query.sql_query,
                'validation_result': matched_query.validation_result,
                'overall_confidence': getattr(matched_query, 'overall_confidence', None),
                'variations': matched_query.variations,
                'matched_text': matched_text
            }
        else:
            print(f"{Fore.YELLOW}  ✗ No match (below threshold){Style.RESET_ALL}")
        
        return None
    
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
    
    def add_validated_query_to_history(self, 
                                      nl_query: str, 
                                      sql_query: str, 
                                      validation_result: Dict[str, Any],
                                      overall_confidence: float,
                                      variations: List[str] = None) -> Optional[QueryHistory]:
        """
        Add validated query to history with validation results
        Only add if overall confidence is > 90%
        """
        # Check confidence threshold
        # if overall_confidence <= 90:
        #     print(f"{Fore.YELLOW}  ℹ Query not added to history (confidence {overall_confidence}% <= 90%){Style.RESET_ALL}")
        #     logger.info(f"Query not cached due to low confidence: {overall_confidence}%")
        #     return None
        
        # Generate ID
        query_id = f"q_{len(self.query_history) + 1:03d}"
        
        # Create new query entry
        new_query = QueryHistory(
            id=query_id,
            natural_language=nl_query,
            variations=variations or [],
            sql_query=sql_query,
            validation_result=validation_result,
            overall_confidence=overall_confidence,
            last_used=datetime.now().isoformat()
        )
        
        # Add to history
        self.query_history.append(new_query)
        
        # Save to file
        DataLoader.save_query_history(self.query_history_path, self.query_history)
        
        # Re-index for future similarity searches
        self._index_query_history()
        
        print(f"{Fore.GREEN}  ✓ Query added to history (ID: {query_id}, Confidence: {overall_confidence}%){Style.RESET_ALL}")
        logger.info(f"Added validated query to history: {query_id}")
        return new_query
    
    def build_context(self, tables: List[str], user_query: str) -> Dict[str, Any]:
        """Build context for LLM"""
        context = {
            'user_query': user_query,
            'tables': {},
            'relationships': [],
            'sample_values': {},
            'business_rules': [],
            'sql_rules': [],
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
        
        # Add business rules
        for rul in self.business_rules:
            context['business_rules'].append({
                'name': rul.name,
                'rule': rul.rule,
                'active': rul.active
            })

        # Add SQL rules
        for rul in self.sql_rules:
            context['sql_rules'].append({
                'name': rul.name,
                'rule': rul.rule,
                'active': rul.active
            })
        
        return context
    
rag_agent = RAGSystem()    