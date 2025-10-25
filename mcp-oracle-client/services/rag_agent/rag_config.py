"""
RAG Configuration and Data Models
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional
import json


# ========================= Data Models =========================

@dataclass
class Column:
    column_name: str
    data_type: str
    description: str
    is_primary_key: bool = False
    nullable: bool = True
    sample_values: List[Any] = field(default_factory=list)
    
    def to_text(self):
        pk_text = " (PRIMARY KEY)" if self.is_primary_key else ""
        null_text = " NOT NULL" if not self.nullable else ""
        samples_text = f" Examples: {', '.join(str(v) for v in self.sample_values[:3])}" if self.sample_values else ""
        return f"{self.column_name} {self.data_type}{pk_text}{null_text}: {self.description}{samples_text}"


@dataclass
class Table:
    table_name: str
    table_description: str
    columns: List[Column]
    
    def to_text(self):
        columns_text = "\n".join([col.to_text() for col in self.columns])
        return f"""
        Table: {self.table_name}
        Description: {self.table_description}
        Columns:
        {columns_text}
        Keywords: {self._generate_keywords()}
        """
    
    def _generate_keywords(self):
        keywords = [self.table_name.lower()]
        keywords.extend([col.column_name.lower() for col in self.columns])
        if self.table_name.endswith('S'):
            keywords.append(self.table_name[:-1].lower())
        else:
            keywords.append(f"{self.table_name}S".lower())
        return ", ".join(set(keywords))


@dataclass
class Relationship:
    parent_table: str
    parent_column: str
    child_table: str
    child_column: str
    relationship_type: str
    description: str


@dataclass
class QueryHistory:
    id: str
    natural_language: str
    variations: List[str]
    sql_query: str
    tables_used: List[str]
    query_type: str
    performance_score: float = 0.0
    usage_count: int = 0
    last_used: Optional[str] = None


# ========================= Configuration =========================

class RAGConfig:
    """Configuration settings for RAG system"""
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD = 0.85
    
    # Search settings
    TOP_K_TABLES = 3
    TOP_K_COLUMNS = 5
    MAX_RELATIONSHIP_DEPTH = 2
    
    # ChromaDB settings
    CHROMA_DB_PATH = "./chroma_db"
    CHROMA_SETTINGS = {
        "anonymized_telemetry": False,
        "allow_reset": True,
        "is_persistent": True
    }
    
    # File paths
    DEFAULT_DATA_DICT_PATH = "./data/data_dictionary.json"
    DEFAULT_RELATIONSHIPS_PATH = "./data/relationships.json"
    DEFAULT_QUERY_HISTORY_PATH = "./data/query_history.json"


# ========================= Data Loader =========================

class DataLoader:
    @staticmethod
    def load_data_dictionary(file_path: str) -> List[Table]:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        tables = []
        for table_data in data['tables']:
            columns = [
                Column(**{k: v for k, v in col.items() if k in Column.__dataclass_fields__})
                for col in table_data['columns']
            ]
            tables.append(Table(
                table_name=table_data['table_name'],
                table_description=table_data['table_description'],
                columns=columns
            ))
        return tables
    
    @staticmethod
    def load_relationships(file_path: str) -> List[Relationship]:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return [Relationship(**rel) for rel in data['relationships']]
    
    @staticmethod
    def load_query_history(file_path: str) -> List[QueryHistory]:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return [QueryHistory(**query) for query in data['queries']]
        except FileNotFoundError:
            return []
    
    @staticmethod
    def save_query_history(file_path: str, queries: List[QueryHistory]):
        data = {
            'queries': [
                {
                    'id': q.id,
                    'natural_language': q.natural_language,
                    'variations': q.variations,
                    'sql_query': q.sql_query,
                    'tables_used': q.tables_used,
                    'query_type': q.query_type,
                    'performance_score': q.performance_score,
                    'usage_count': q.usage_count,
                    'last_used': q.last_used
                } for q in queries
            ]
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)