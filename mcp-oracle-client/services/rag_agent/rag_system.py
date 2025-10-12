"""
Advanced RAG System with Schema-Aware Retrieval and Pattern Matching.
"""

import logging
import hashlib
import json
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import networkx as nx
import sqlparse
from collections import defaultdict
import re

logger = logging.getLogger(__name__)

@dataclass
class SchemaNode:
    """Represents a table in the schema graph."""
    table_name: str
    columns: Dict[str, Any]
    primary_keys: List[str]
    foreign_keys: Dict[str, str]
    indexes: List[str]
    relationships: Dict[str, str]
    common_queries: List[str]

@dataclass
class QueryPattern:
    """Represents a reusable query pattern."""
    pattern_id: str
    name: str
    template: str
    description: str
    applicable_conditions: List[str]
    example_usage: str
    performance_notes: str

class RAGSystem:
    """
    Advanced RAG system with schema-aware retrieval and pattern matching.
    """
    
    def __init__(self, 
                 persist_directory: str = "./vector_db2/chroma_db_advanced",
                 collection_prefix: str = "oracle_advanced",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize Advanced RAG system."""
        self.persist_directory = persist_directory
        self.collection_prefix = collection_prefix
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding function
        try:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model,
                device="cpu"
            )
        except:
            logger.warning("Using default embedding function")
            self.embedding_function = None
        
        # Initialize collections
        self._initialize_collections()
        
        # Initialize schema graph
        self.schema_graph = nx.DiGraph()
        self.table_index = {}
        self.column_index = defaultdict(list)
        
        # Initialize pattern library
        self.pattern_library = {}
        
        # Query cache
        self.query_cache = {}
        self.cache_ttl = timedelta(hours=1)
        
        logger.info("Advanced RAG System initialized")
    
    def _initialize_collections(self):
        """Initialize ChromaDB collections."""
        collection_configs = {
            'schema': 'Schema and DDL information',
            'queries': 'Validated SQL queries',
            'patterns': 'Query patterns and templates',
            'dictionary': 'Data dictionary entries',
            'relationships': 'Table relationships'
        }
        
        self.collections = {}
        for name, description in collection_configs.items():
            collection_name = f"{self.collection_prefix}_{name}"
            try:
                if self.embedding_function:
                    self.collections[name] = self.client.get_or_create_collection(
                        name=collection_name,
                        metadata={"description": description},
                        embedding_function=self.embedding_function
                    )
                else:
                    self.collections[name] = self.client.get_or_create_collection(
                        name=collection_name,
                        metadata={"description": description}
                    )
            except Exception as e:
                logger.error(f"Failed to create collection {name}: {e}")
    
    def build_schema_graph(self, data_dictionary: Dict[str, Dict[str, Any]]) -> None:
        """Build schema graph from data dictionary."""
        for table_name, table_info in data_dictionary.items():
            # Add table node
            self.schema_graph.add_node(table_name, **table_info)
            self.table_index[table_name] = table_info
            
            # Index columns
            for column_name in table_info.get('columns', {}):
                self.column_index[column_name].append(table_name)
            
            # Add relationships as edges
            relationships = table_info.get('relationships', {})
            for related_table, relationship_type in relationships.items():
                self.schema_graph.add_edge(
                    table_name, 
                    related_table,
                    relationship=relationship_type
                )
        
        logger.info(f"Built schema graph with {len(self.schema_graph.nodes)} tables")
    
    def get_related_tables(self, tables: List[str], max_depth: int = 2) -> Set[str]:
        """Get related tables within specified depth."""
        related = set(tables)
        
        for table in tables:
            if table in self.schema_graph:
                # Get neighbors within max_depth
                for depth in range(1, max_depth + 1):
                    neighbors = nx.single_source_shortest_path_length(
                        self.schema_graph, table, cutoff=depth
                    )
                    related.update(neighbors.keys())
        
        return related
    
    def identify_required_tables(self, task: str) -> List[str]:
        """Identify tables required for a task using NLP and schema analysis."""
        required_tables = []
        task_lower = task.lower()
        
        # Direct table mentions
        for table_name in self.table_index.keys():
            if table_name.lower() in task_lower:
                required_tables.append(table_name)
        
        # Column mentions
        for column_name, tables in self.column_index.items():
            if column_name.lower() in task_lower:
                required_tables.extend(tables)
        
        # Business term mapping
        business_terms = {
            'profit': ['financial_performance'],
            'revenue': ['financial_performance'],
            'income': ['financial_performance'],
            'market cap': ['market_data'],
            'share price': ['market_data'],
            'capital': ['market_data'],
            'bank': ['banks']
        }
        
        for term, tables in business_terms.items():
            if term in task_lower:
                required_tables.extend(tables)
        
        # Get unique tables and their related tables
        unique_tables = list(set(required_tables))
        related_tables = self.get_related_tables(unique_tables, max_depth=1)
        
        return list(related_tables)
    
    def get_relevant_context(self, task: str, top_k: int = 10) -> Dict[str, Any]:
        """Get comprehensive context for SQL generation."""
        context = {
            'schema_graph': {},
            'ddl': [],
            'dictionary': [],
            'patterns': [],
            'validated_queries': [],
            'relationships': [],
            'schema_summary': {}
        }
        
        # Identify required tables
        required_tables = self.identify_required_tables(task)
        
        # Build schema graph context
        for table in required_tables:
            if table in self.table_index:
                context['schema_graph'][table] = self.table_index[table]
                context['schema_summary'][table] = list(
                    self.table_index[table].get('columns', {}).keys()
                )
        
        # Retrieve DDL for required tables
        if self.collections.get('schema'):
            where_clause = {"$or": [{"table_name": table} for table in required_tables]} if required_tables else None
            results = self.collections['schema'].query(
                query_texts=[task],
                n_results=min(top_k, len(required_tables) * 2) if required_tables else top_k,
                where=where_clause
            )
            
            if results['documents'][0]:
                for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                    context['ddl'].append({
                        'content': doc,
                        'metadata': metadata
                    })
        
        # Retrieve similar validated queries
        if self.collections.get('queries'):
            results = self.collections['queries'].query(
                query_texts=[f"Task: {task}"],
                n_results=top_k,
                where={"user_validated": True}
            )
            
            if results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0], 
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    similarity = 1 - (distance / 2)
                    if similarity > 0.7:
                        context['validated_queries'].append({
                            'task': metadata.get('task', ''),
                            'sql': metadata.get('sql_query', ''),
                            'similarity': similarity
                        })
        
        # Retrieve applicable patterns
        patterns = self.find_applicable_patterns(task)
        context['patterns'] = patterns
        
        # Add relationship information
        for table in required_tables:
            if table in self.schema_graph:
                neighbors = list(self.schema_graph.neighbors(table))
                if neighbors:
                    context['relationships'].append({
                        'table': table,
                        'related_to': neighbors
                    })
        
        return context
    
    def find_applicable_patterns(self, task: str) -> List[Dict[str, Any]]:
        """Find applicable query patterns for the task."""
        applicable = []
        task_lower = task.lower()
        
        # Pattern detection rules
        pattern_rules = {
            'aggregation': ['sum', 'average', 'count', 'total', 'mean', 'max', 'min'],
            'join_query': ['compare', 'across', 'between', 'with', 'join'],
            'year_over_year': ['year-over-year', 'yoy', 'growth', 'change', 'trend'],
            'window_function': ['rank', 'top', 'bottom', 'running', 'cumulative', 'lag', 'lead'],
            'filtering': ['where', 'filter', 'only', 'specific', 'equal', 'greater', 'less']
        }
        
        for pattern_name, keywords in pattern_rules.items():
            if any(keyword in task_lower for keyword in keywords):
                if pattern_name in self.pattern_library:
                    applicable.append({
                        'name': pattern_name,
                        'template': self.pattern_library[pattern_name].template,
                        'description': self.pattern_library[pattern_name].description
                    })
        
        return applicable
    
    def add_validated_query(self, 
                           task: str, 
                           sql_query: str,
                           metadata: Dict[str, Any] = None) -> bool:
        """Add a validated query to the system."""
        try:
            # Parse SQL to extract tables and columns
            tables, columns = self._extract_sql_elements(sql_query)
            
            doc_id = f"query_{hashlib.md5((task + sql_query).encode()).hexdigest()[:16]}"
            doc_text = f"Task: {task}\nSQL: {sql_query}"
            
            query_metadata = {
                'task': task,
                'sql_query': sql_query,
                'tables_used': json.dumps(list(tables)),
                'columns_used': json.dumps(list(columns)),
                'timestamp': datetime.now().isoformat(),
                'user_validated': True,
                'execution_count': 0
            }
            
            if metadata:
                query_metadata.update(metadata)
            
            self.collections['queries'].add(
                documents=[doc_text],
                metadatas=[query_metadata],
                ids=[doc_id]
            )
            
            # Update pattern library if applicable
            self._update_pattern_library(task, sql_query)
            
            logger.info(f"Added validated query for task: {task[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error adding validated query: {e}")
            return False
    
    def _extract_sql_elements(self, sql_query: str) -> Tuple[Set[str], Set[str]]:
        """Extract tables and columns from SQL query."""
        tables = set()
        columns = set()
        
        try:
            parsed = sqlparse.parse(sql_query)[0]
            
            # Extract tables (simplified - you might want to use a proper SQL parser)
            from_idx = None
            tokens = parsed.tokens
            
            for i, token in enumerate(tokens):
                if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == 'FROM':
                    from_idx = i
                    break
            
            if from_idx is not None:
                # Look for table names after FROM
                for token in tokens[from_idx + 1:]:
                    if token.ttype is sqlparse.tokens.Keyword:
                        break
                    if not token.is_whitespace:
                        # Extract table name (simplified)
                        table_str = str(token).strip()
                        # Handle aliases
                        table_parts = table_str.split()
                        if table_parts:
                            tables.add(table_parts[0])
            
            # Extract columns (simplified)
            select_pattern = r'SELECT\s+(.*?)\s+FROM'
            select_match = re.search(select_pattern, sql_query, re.IGNORECASE | re.DOTALL)
            if select_match:
                select_clause = select_match.group(1)
                # Parse column names (simplified)
                column_parts = select_clause.split(',')
                for part in column_parts:
                    # Extract column name before any alias
                    col_match = re.search(r'(\w+\.)?(\w+)', part.strip())
                    if col_match:
                        columns.add(col_match.group(2))
            
        except Exception as e:
            logger.error(f"Error extracting SQL elements: {e}")
        
        return tables, columns
    
    def _update_pattern_library(self, task: str, sql_query: str):
        """Update pattern library based on validated queries."""
        # Analyze query structure to identify patterns
        task_lower = task.lower()
        
        # Check for year-over-year pattern
        if 'lag(' in sql_query.lower() or 'lead(' in sql_query.lower():
            if 'year_over_year' not in self.pattern_library:
                self.pattern_library['year_over_year'] = QueryPattern(
                    pattern_id='yoy_001',
                    name='year_over_year',
                    template=sql_query,
                    description='Year-over-year comparison pattern',
                    applicable_conditions=['growth', 'trend', 'change'],
                    example_usage=task,
                    performance_notes='Uses window functions, ensure proper indexing'
                )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the RAG system."""
        stats = {
            'collections': {},
            'schema_graph': {
                'tables': len(self.schema_graph.nodes),
                'relationships': len(self.schema_graph.edges),
                'avg_connections': nx.average_degree_connectivity(self.schema_graph) if self.schema_graph.nodes else 0
            },
            'pattern_library': {
                'total_patterns': len(self.pattern_library),
                'pattern_names': list(self.pattern_library.keys())
            },
            'cache': {
                'cached_queries': len(self.query_cache)
            }
        }
        
        # Collection statistics
        for name, collection in self.collections.items():
            try:
                stats['collections'][name] = collection.count()
            except:
                stats['collections'][name] = 0
        
        return stats
    
    def clear_cache(self):
        """Clear query cache."""
        self.query_cache = {}
        logger.info("Query cache cleared")