"""
RAG Configuration and Data Models
Enhanced with validation result support
"""

from dataclasses import dataclass, field
from typing import List, Any, Optional, Dict
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
class Rules:
    id: str
    name: str
    rule: str
    active: str
    created_date: str

@dataclass
class QueryHistory:
    id: str
    natural_language: str
    variations: List[str]
    sql_query: str
    validation_result: Optional[Dict[str, Any]] = None
    overall_confidence: Optional[float] = None
    last_used: Optional[str] = None
    
    def __post_init__(self):
        """Ensure validation_result has the right structure"""
        if self.validation_result is None:
            self.validation_result = {
                'schema': 0,
                'syntax': 0,
                'semantic': 0,
                'completeness': 0
            }


# ========================= Configuration =========================

class RAGConfig:
    """Configuration settings for RAG system"""
    
    # Model settings
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD = 0.90  # Updated to 90% for query history matching
    
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
    DEFAULT_BUSINESS_RULES_PATH = "./data/business_rules.json"
    DEFAULT_SQL_RULES_PATH = "./data/sql_generation_rules.json"


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
    def load_rules(file_path: str, rule_type:str) -> List[Rules]:
        with open(file_path, 'r') as f:
            data = json.load(f)

        return [Rules(**rul) for rul in data[rule_type]]

    @staticmethod
    def load_query_history(file_path: str) -> List[QueryHistory]:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            queries = []
            for query_data in data['queries']:
                # Handle validation_result field
                validation_result = query_data.get('validation_result', {})
                
                queries.append(QueryHistory(
                    id=query_data['id'],
                    natural_language=query_data['natural_language'],
                    variations=query_data.get('variations', []),
                    sql_query=query_data['sql_query'],
                    validation_result=validation_result,
                    overall_confidence=query_data.get('overall_confidence'),
                    last_used=query_data.get('last_used')
                ))
            
            return queries
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
                    'validation_result': q.validation_result,
                    'overall_confidence': q.overall_confidence,
                    'last_used': q.last_used
                } for q in queries
            ]
        }
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)