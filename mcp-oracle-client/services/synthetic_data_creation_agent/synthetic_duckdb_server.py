"""
Synthetic Data MCP Server using DuckDB
Split tools for single table and multi-table generation
"""

from mcp.server.fastmcp import FastMCP
import duckdb
import pandas as pd
import json
import logging
import sys
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

# SDV imports
try:
    from sdv.metadata import Metadata
    from sdv.lite import SingleTablePreset
    from sdv.multi_table import HMASynthesizer
    from sdv.evaluation.single_table import evaluate_quality
    from sdv.evaluation.single_table import get_column_plot
except ImportError:
    raise ImportError("SDV not installed. Please install using: pip install sdv")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("synthetic-duckdb-mcp")

# Initialize FastMCP server
mcp = FastMCP("Synthetic-DuckDB-Server")

# Database configuration
DB_PATH = "synthetic_data.db"
connection = duckdb.connect(DB_PATH)


@mcp.tool()
async def list_tables() -> str:
    """List all tables in the DuckDB database.
    
    Returns:
        str: List of all table names in the database
    """
    try:
        result = connection.execute("SHOW TABLES").fetchall()
        if not result:
            return "No tables found in database"
        tables = [row[0] for row in result]
        return f"Tables in database: {', '.join(tables)}"
    except Exception as e:
        return f"Error listing tables: {str(e)}"


@mcp.tool()
async def describe_table(table_name: str) -> str:
    """Describe the structure and row count of a specific table in DuckDB.
    
    Args:
        table_name: Name of the table to describe
        
    Returns:
        str: Table structure information including columns, types, and row count
    """
    try:
        result = connection.execute(f"DESCRIBE {table_name}").fetchall()
        description = f"Table: {table_name}\n"
        description += "Columns:\n"
        for row in result:
            description += f"  - {row[0]}: {row[1]}\n"
        
        # Get row count
        count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        description += f"\nTotal rows: {count}"
        
        return description
    except Exception as e:
        return f"Error describing table {table_name}: {str(e)}"


@mcp.tool()
async def load_metadata(metadata_path: str, create_tables: bool = True) -> str:
    """Load metadata JSON file and optionally create tables based on schema.
    
    Args:
        metadata_path: Path to the metadata JSON file
        create_tables: Whether to create tables if they don't exist (default: True)
        
    Returns:
        str: Status message about metadata loading and table creation
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        # Store metadata in a configuration table for reference
        metadata_json = json.dumps(metadata_dict)
        connection.execute("""
            CREATE OR REPLACE TABLE metadata_config (
                config_json TEXT,
                loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        connection.execute("INSERT INTO metadata_config (config_json) VALUES (?)", [metadata_json])
        
        created_tables = []
        skipped_tables = []
        
        if create_tables:
            # Get existing tables
            existing_tables = set()
            result = connection.execute("SHOW TABLES").fetchall()
            for row in result:
                existing_tables.add(row[0].lower())
            
            # Create tables based on metadata
            for table_name, table_info in metadata_dict.get('tables', {}).items():
                if table_name.lower() in existing_tables:
                    skipped_tables.append(table_name)
                    continue
                
                # Build CREATE TABLE statement
                columns = []
                for col_name, col_info in table_info.get('columns', {}).items():
                    # Use data_type directly from metadata
                    col_type = col_info.get('data_type', 'TEXT')
                    
                    # Add primary key constraint if specified
                    if col_name == table_info.get('primary_key'):
                        col_type += " PRIMARY KEY"
                    
                    columns.append(f"{col_name} {col_type}")
                
                if columns:
                    create_stmt = f"CREATE TABLE {table_name} ({', '.join(columns)})"
                    connection.execute(create_stmt)
                    created_tables.append(table_name)
                    logger.info(f"Created table: {table_name}")
            
            # Add foreign key constraints (as comments since DuckDB has limited FK support)
            for rel in metadata_dict.get('relationships', []):
                parent = rel.get('parent_table_name')
                child = rel.get('child_table_name')
                parent_key = rel.get('parent_primary_key')
                child_key = rel.get('child_foreign_key')
                
                # Store relationship info in metadata
                logger.info(f"Relationship noted: {child}.{child_key} -> {parent}.{parent_key}")
        
        # Prepare summary
        tables = list(metadata_dict.get('tables', {}).keys())
        relationships = len(metadata_dict.get('relationships', []))
        
        summary = f"Metadata loaded. Total tables defined: {len(tables)}, Relationships: {relationships}"
        if create_tables:
            summary += f"\nCreated tables: {', '.join(created_tables) if created_tables else 'none'}"
            summary += f"\nExisting tables skipped: {', '.join(skipped_tables) if skipped_tables else 'none'}"
        
        return summary
        
    except Exception as e:
        return f"Error loading metadata: {str(e)}"


@mcp.tool()
async def create_sample_tables(
    metadata_path: str, 
    rows_per_table: int = 100, 
    use_existing_schema: bool = True
) -> str:
    """Create and populate sample tables with realistic data based on metadata.
    
    Args:
        metadata_path: Path to metadata JSON file
        rows_per_table: Number of sample rows to create per table (default: 100)
        use_existing_schema: If True, populate existing tables; if False, recreate tables
        
    Returns:
        str: Status message about table creation and data population
    """
    try:
        import random
        from datetime import datetime, timedelta
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # First ensure tables exist
        if not use_existing_schema:
            await load_metadata(metadata_path, create_tables=True)
        
        populated_tables = []
        
        # Track foreign key relationships for consistency
        foreign_key_map = {}
        for rel in metadata.get('relationships', []):
            if rel['child_table_name'] not in foreign_key_map:
                foreign_key_map[rel['child_table_name']] = []
            foreign_key_map[rel['child_table_name']].append({
                'parent_table': rel['parent_table_name'],
                'parent_key': rel['parent_primary_key'],
                'child_key': rel['child_foreign_key']
            })
        
        # Generate parent tables first (tables without foreign keys or with fewer dependencies)
        table_order = _get_table_creation_order(metadata)
        
        generated_ids = {}  # Store generated IDs for foreign key references
        
        for table_name in table_order:
            if table_name not in metadata['tables']:
                continue
                
            table_info = metadata['tables'][table_name]
            columns = []
            values_dict = {}
            
            # Generate data for each column
            for col_name, col_info in table_info['columns'].items():
                data_type = col_info.get('data_type', 'TEXT').upper()
                sample_values = col_info.get('sample_values', [])
                
                # Extract base type (e.g., VARCHAR from VARCHAR(50))
                base_type = data_type.split('(')[0]
                
                # Generate appropriate data based on type and context
                if col_name == table_info.get('primary_key'):
                    # Generate unique IDs
                    prefix = col_name.upper().replace('_ID', '')
                    col_values = [f"{prefix}_{i:06d}" for i in range(1, rows_per_table + 1)]
                    generated_ids[table_name] = col_values
                    
                elif col_name in [fk['child_key'] for fk in foreign_key_map.get(table_name, [])]:
                    # Foreign key
                    for fk in foreign_key_map[table_name]:
                        if fk['child_key'] == col_name:
                            parent_ids = generated_ids.get(fk['parent_table'], [])
                            if parent_ids:
                                col_values = [random.choice(parent_ids) for _ in range(rows_per_table)]
                            else:
                                col_values = [f"FK_{i:06d}" for i in range(rows_per_table)]
                            break
                else:
                    # Use sample values if available, otherwise generate
                    if sample_values and len(sample_values) >= 3:
                        col_values = [random.choice(sample_values) for _ in range(rows_per_table)]
                    else:
                        # Generate based on data type
                        if base_type in ['DECIMAL', 'DOUBLE', 'FLOAT']:
                            col_values = [round(random.uniform(100, 100000), 2) for _ in range(rows_per_table)]
                        elif base_type in ['INTEGER', 'INT']:
                            col_values = [random.randint(1, 100) for _ in range(rows_per_table)]
                        elif base_type == 'DATE':
                            base_date = datetime.now()
                            col_values = [
                                (base_date - timedelta(days=random.randint(0, 365*5))).strftime('%Y-%m-%d')
                                for _ in range(rows_per_table)
                            ]
                        elif base_type == 'TIMESTAMP':
                            base_date = datetime.now()
                            col_values = [
                                (base_date - timedelta(
                                    days=random.randint(0, 365*5),
                                    hours=random.randint(0, 23),
                                    minutes=random.randint(0, 59)
                                )).strftime('%Y-%m-%d %H:%M:%S')
                                for _ in range(rows_per_table)
                            ]
                        elif base_type in ['BOOLEAN', 'BOOL']:
                            col_values = [random.choice([True, False]) for _ in range(rows_per_table)]
                        else:
                            col_values = [f"{col_name}_value_{i}" for i in range(rows_per_table)]
                
                columns.append(col_name)
                values_dict[col_name] = col_values
            
            # Create DataFrame and insert into database
            if columns:
                df = pd.DataFrame(values_dict)
                
                # Check if table exists and truncate it
                existing_tables = [t[0] for t in connection.execute("SHOW TABLES").fetchall()]
                if table_name in existing_tables:
                    connection.execute(f"DELETE FROM {table_name}")
                
                # Insert data
                connection.execute(f"INSERT INTO {table_name} SELECT * FROM df")
                
                row_count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                populated_tables.append(f"{table_name} ({row_count} rows)")
                logger.info(f"Populated table {table_name} with {row_count} rows")
        
        return f"Successfully populated tables with sample data: {', '.join(populated_tables)}"
        
    except Exception as e:
        return f"Error creating sample tables: {str(e)}"


@mcp.tool()
async def generate_single_table_synthetic(
    source_table_name: str,
    target_table_name: Optional[str] = None,
    num_rows: Optional[int] = None,
    with_relationships: bool = False,
    metadata_path: Optional[str] = None
) -> str:
    """Generate synthetic data for a single table, optionally including related tables.
    
    Args:
        source_table_name: Name of the source table to generate synthetic data from
        target_table_name: Name for the synthetic table (defaults to source_table_synthetic)
        num_rows: Number of synthetic rows to generate (defaults to source table size)
        with_relationships: If True and metadata provided, also generate related child tables
        metadata_path: Optional path to metadata for relationship awareness
        
    Returns:
        str: Status message about synthetic data generation
    """
    try:
        # Check if source table exists
        existing_tables = [row[0] for row in connection.execute("SHOW TABLES").fetchall()]
        if source_table_name not in existing_tables:
            return f"Error: Source table '{source_table_name}' not found in database"
        
        # Read source data
        source_df = connection.execute(f"SELECT * FROM {source_table_name}").df()
        
        if source_df.empty:
            return f"Error: Source table {source_table_name} is empty"
        
        # Determine number of rows
        if num_rows is None:
            num_rows = len(source_df)
        
        logger.info(f"Generating {num_rows} synthetic rows for {source_table_name}")
        
        # Generate synthetic data for main table
        synthesizer = SingleTablePreset(source_df, name='FAST_ML')
        synthetic_df = synthesizer.sample(num_rows=num_rows)
        
        # Save main synthetic table
        if target_table_name is None:
            target_table_name = f"{source_table_name}_synthetic"
        
        connection.execute(f"DROP TABLE IF EXISTS {target_table_name}")
        connection.register(target_table_name, synthetic_df)
        connection.execute(f"CREATE TABLE {target_table_name} AS SELECT * FROM {target_table_name}")
        
        generated_tables = [f"{target_table_name} ({num_rows} rows)"]
        
        # If with_relationships is True and metadata is provided, generate related tables
        if with_relationships and metadata_path:
            logger.info(f"Generating related data for child tables of {source_table_name}")
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            # Get relationships where source_table is parent
            relationships = metadata_dict.get('relationships', [])
            child_relationships = [
                rel for rel in relationships 
                if rel['parent_table_name'] == source_table_name
            ]
            
            # Get primary key of source table
            primary_key = metadata_dict['tables'][source_table_name].get('primary_key')
            if not primary_key:
                primary_key = synthetic_df.columns[0]
            
            generated_ids = synthetic_df[primary_key].tolist()
            
            # Generate child tables
            for rel in child_relationships:
                child_table = rel['child_table_name']
                child_foreign_key = rel['child_foreign_key']
                
                if child_table not in existing_tables:
                    continue
                
                # Read existing child table data
                child_df = connection.execute(f"SELECT * FROM {child_table}").df()
                if child_df.empty:
                    continue
                
                # Calculate average child records per parent
                avg_children = max(1, len(child_df) // source_df[primary_key].nunique())
                child_rows = num_rows * avg_children
                
                # Generate synthetic child data
                child_synthesizer = SingleTablePreset(child_df, name='FAST_ML')
                synthetic_child_df = child_synthesizer.sample(num_rows=child_rows)
                
                # Assign parent IDs
                parent_ids_repeated = []
                for parent_id in generated_ids:
                    parent_ids_repeated.extend([parent_id] * avg_children)
                synthetic_child_df[child_foreign_key] = parent_ids_repeated[:len(synthetic_child_df)]
                
                # Save child synthetic table
                child_target = f"{child_table}_synthetic"
                connection.execute(f"DROP TABLE IF EXISTS {child_target}")
                connection.register(child_target, synthetic_child_df)
                connection.execute(f"CREATE TABLE {child_target} AS SELECT * FROM {child_target}")
                
                generated_tables.append(f"{child_target} ({len(synthetic_child_df)} rows)")
        
        return f"Successfully generated synthetic data: {', '.join(generated_tables)}"
        
    except Exception as e:
        return f"Error generating synthetic data: {str(e)}"


@mcp.tool()
async def generate_multi_table_synthetic(
    metadata_path: str,
    ensure_tables_exist: bool = True
) -> str:
    """Generate synthetic data for multiple related tables based on metadata.
    
    Args:
        metadata_path: Path to metadata JSON file defining all tables and relationships
        ensure_tables_exist: If True, create missing tables before generation (default: True)
        
    Returns:
        str: Status message about synthetic data generation for all tables
    """
    try:
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        
        if ensure_tables_exist:
            # Check which tables exist
            existing_tables = set()
            result = connection.execute("SHOW TABLES").fetchall()
            for row in result:
                existing_tables.add(row[0].lower())
            
            # Identify missing tables
            missing_tables = []
            for table_name in metadata_dict['tables'].keys():
                if table_name.lower() not in existing_tables:
                    missing_tables.append(table_name)
            
            # Create missing tables
            if missing_tables:
                logger.info(f"Creating missing tables: {missing_tables}")
                
                # Load metadata to create tables
                await load_metadata(metadata_path, create_tables=True)
                
                # Populate with sample data
                await create_sample_tables(
                    metadata_path=metadata_path,
                    rows_per_table=100,
                    use_existing_schema=True
                )
        
        # Clean metadata for SDV
        clean_metadata_dict = _clean_metadata_for_sdv(metadata_dict)
        
        # Load all existing data
        real_data = {}
        for table_name in clean_metadata_dict['tables'].keys():
            try:
                df = connection.execute(f"SELECT * FROM {table_name}").df()
                if not df.empty:
                    real_data[table_name] = df
                    logger.info(f"Loaded {len(df)} rows from {table_name}")
            except Exception as e:
                logger.warning(f"Could not load {table_name}: {e}")
        
        if not real_data:
            return "Error: No valid tables found in database"
        
        # Get relationships
        relationships = metadata_dict.get('relationships', [])
        
        # Determine generation order (parents first)
        table_order = _get_table_creation_order(metadata_dict)
        
        synthetic_data = {}
        id_mappings = {}
        
        # Generate each table using fast approach
        for table_name in table_order:
            if table_name not in real_data:
                continue
            
            df = real_data[table_name]
            logger.info(f"Generating synthetic data for {table_name}")
            
            try:
                # Generate synthetic data
                synthesizer = SingleTablePreset(df, name='FAST_ML')
                synthetic_df = synthesizer.sample(num_rows=len(df))
                
                # Store primary key mappings
                primary_key = metadata_dict['tables'][table_name].get('primary_key')
                if primary_key and primary_key in synthetic_df.columns:
                    id_mappings[table_name] = {
                        'primary_key': primary_key,
                        'ids': synthetic_df[primary_key].tolist()
                    }
                
                # Fix foreign keys
                for rel in relationships:
                    if rel['child_table_name'] == table_name:
                        parent_table = rel['parent_table_name']
                        child_key = rel['child_foreign_key']
                        
                        if parent_table in id_mappings and child_key in synthetic_df.columns:
                            valid_parent_ids = id_mappings[parent_table]['ids']
                            if valid_parent_ids:
                                # Randomly assign valid parent IDs
                                synthetic_df[child_key] = np.random.choice(
                                    valid_parent_ids,
                                    size=len(synthetic_df)
                                )
                                logger.info(f"Fixed foreign key {table_name}.{child_key} -> {parent_table}")
                
                synthetic_data[table_name] = synthetic_df
                
            except Exception as e:
                logger.error(f"Failed to generate {table_name}: {e}")
                # Fallback to sampling
                synthetic_data[table_name] = df.sample(n=len(df), replace=True).reset_index(drop=True)
        
        # Save all synthetic tables
        generated_tables = []
        for table_name, synthetic_df in synthetic_data.items():
            synthetic_table_name = f"{table_name}_synthetic"
            
            connection.execute(f"DROP TABLE IF EXISTS {synthetic_table_name}")
            connection.register(synthetic_table_name, synthetic_df)
            connection.execute(f"CREATE TABLE {synthetic_table_name} AS SELECT * FROM {synthetic_table_name}")
            
            generated_tables.append(f"{synthetic_table_name} ({len(synthetic_df)} rows)")
        
        return f"Successfully generated synthetic data for all tables: {', '.join(generated_tables)}"
        
    except Exception as e:
        return f"Error in multi-table generation: {str(e)}"

@mcp.tool()
async def evaluate_synthetic_quality(
    real_table: str,
    synthetic_table: str,
    metadata_path: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate the quality of synthetic data compared to real data.
    
    Args:
        real_table: Name of the real data table
        synthetic_table: Name of the synthetic data table
        metadata_path: Optional path to metadata JSON for better evaluation
        
    Returns:
        dict: Quality metrics including overall score and property scores
    """
    try:
        # Load real and synthetic data
        real_df = connection.execute(f"SELECT * FROM {real_table}").df()
        synthetic_df = connection.execute(f"SELECT * FROM {synthetic_table}").df()
        
        if real_df.empty or synthetic_df.empty:
            return {"error": "One or both tables are empty"}
        
        # Basic validation
        if len(real_df.columns) != len(synthetic_df.columns):
            return {
                "error": f"Column mismatch: real has {len(real_df.columns)} columns, synthetic has {len(synthetic_df.columns)}"
            }
        
        # Load metadata if provided
        metadata = None
        if metadata_path:
            try:
                with open(metadata_path, 'r') as f:
                    metadata_dict = json.load(f)
                    
                    # Clean metadata for SDV
                    clean_metadata_dict = _clean_metadata_for_sdv(metadata_dict)
                    
                    if real_table in clean_metadata_dict.get('tables', {}):
                        table_metadata = {
                            'tables': {real_table: clean_metadata_dict['tables'][real_table]},
                            'METADATA_SPEC_VERSION': 'V1'
                        }
                        metadata = Metadata.load_from_dict(table_metadata)
            except Exception as e:
                logger.warning(f"Could not load metadata for evaluation: {e}")
                metadata = None
        
        # Try different evaluation approaches based on SDV version
        try:
            # Try the newer SDV evaluation API
            from sdv.evaluation.single_table import evaluate_quality
            
            # Evaluate without metadata if it causes issues
            if metadata:
                try:
                    quality_report = evaluate_quality(
                        real_data=real_df,
                        synthetic_data=synthetic_df,
                        metadata=metadata
                    )
                except:
                    # Fallback to no metadata
                    quality_report = evaluate_quality(
                        real_data=real_df,
                        synthetic_data=synthetic_df
                    )
            else:
                quality_report = evaluate_quality(
                    real_data=real_df,
                    synthetic_data=synthetic_df
                )
            
            # Extract scores safely
            overall_score = float(quality_report.get_score())
            
            # Try to get property scores
            property_scores = {}
            try:
                properties = quality_report.get_properties()
                if properties is not None and not properties.empty:
                    # Try to extract specific property scores
                    for prop in properties.index:
                        try:
                            property_scores[prop.replace(' ', '_').lower()] = float(properties.loc[prop, 'Score'])
                        except:
                            pass
            except Exception as e:
                logger.warning(f"Could not extract property scores: {e}")
            
            # Get column-level details if available
            column_details = {}
            try:
                # Try to get column shapes details
                details = quality_report.get_details('Column Shapes')
                if details is not None and not details.empty:
                    column_details = details.to_dict()
            except:
                pass
            
            return {
                "overall_score": overall_score,
                "properties": property_scores if property_scores else {"note": "Property scores not available"},
                "details": column_details if column_details else {"note": "Column details not available"},
                "rows_evaluated": {
                    "real": len(real_df),
                    "synthetic": len(synthetic_df)
                }
            }
            
        except ImportError:
            # Fallback to basic statistical comparison if SDV evaluation fails
            logger.warning("SDV evaluation not available, using basic statistical comparison")
            return await _basic_quality_evaluation(real_df, synthetic_df)
            
    except Exception as e:
        # Provide more detailed error information
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Evaluation failed with details: {error_details}")
        return {"error": f"Evaluation failed: {str(e)}", "details": error_details[:500]}


async def _basic_quality_evaluation(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Basic quality evaluation when SDV evaluation is not available.
    Compares statistical properties of real vs synthetic data.
    """
    try:
        import numpy as np
        from scipy import stats
        
        evaluation_results = {
            "method": "basic_statistical_comparison",
            "columns_evaluated": [],
            "overall_score": 0.0
        }
        
        column_scores = []
        
        for col in real_df.columns:
            if col not in synthetic_df.columns:
                continue
            
            col_eval = {"column": col}
            
            # For numeric columns
            if pd.api.types.is_numeric_dtype(real_df[col]):
                real_values = real_df[col].dropna()
                synthetic_values = synthetic_df[col].dropna()
                
                if len(real_values) > 0 and len(synthetic_values) > 0:
                    # Compare distributions using KS test
                    ks_statistic, ks_pvalue = stats.ks_2samp(real_values, synthetic_values)
                    
                    # Compare basic statistics
                    real_mean = real_values.mean()
                    synthetic_mean = synthetic_values.mean()
                    real_std = real_values.std()
                    synthetic_std = synthetic_values.std()
                    
                    # Calculate similarity score (0-1, higher is better)
                    mean_similarity = 1 - min(abs(real_mean - synthetic_mean) / (abs(real_mean) + 1e-10), 1)
                    std_similarity = 1 - min(abs(real_std - synthetic_std) / (abs(real_std) + 1e-10), 1)
                    distribution_similarity = ks_pvalue  # Higher p-value means more similar
                    
                    col_score = (mean_similarity + std_similarity + distribution_similarity) / 3
                    
                    col_eval.update({
                        "type": "numeric",
                        "real_mean": float(real_mean),
                        "synthetic_mean": float(synthetic_mean),
                        "real_std": float(real_std),
                        "synthetic_std": float(synthetic_std),
                        "ks_statistic": float(ks_statistic),
                        "ks_pvalue": float(ks_pvalue),
                        "similarity_score": float(col_score)
                    })
                    
                    column_scores.append(col_score)
            
            # For categorical columns
            else:
                real_values = real_df[col].dropna()
                synthetic_values = synthetic_df[col].dropna()
                
                if len(real_values) > 0 and len(synthetic_values) > 0:
                    # Compare category distributions
                    real_counts = real_values.value_counts(normalize=True)
                    synthetic_counts = synthetic_values.value_counts(normalize=True)
                    
                    # Calculate overlap of categories
                    all_categories = set(real_counts.index) | set(synthetic_counts.index)
                    overlap_categories = set(real_counts.index) & set(synthetic_counts.index)
                    category_overlap = len(overlap_categories) / len(all_categories) if all_categories else 0
                    
                    # Calculate distribution similarity for overlapping categories
                    distribution_similarity = 0
                    if overlap_categories:
                        for cat in overlap_categories:
                            real_prob = real_counts.get(cat, 0)
                            synthetic_prob = synthetic_counts.get(cat, 0)
                            distribution_similarity += 1 - abs(real_prob - synthetic_prob)
                        distribution_similarity /= len(overlap_categories)
                    
                    col_score = (category_overlap + distribution_similarity) / 2
                    
                    col_eval.update({
                        "type": "categorical",
                        "unique_values_real": int(real_values.nunique()),
                        "unique_values_synthetic": int(synthetic_values.nunique()),
                        "category_overlap": float(category_overlap),
                        "distribution_similarity": float(distribution_similarity),
                        "similarity_score": float(col_score)
                    })
                    
                    column_scores.append(col_score)
            
            evaluation_results["columns_evaluated"].append(col_eval)
        
        # Calculate overall score
        if column_scores:
            evaluation_results["overall_score"] = float(np.mean(column_scores))
        
        # Add summary statistics
        evaluation_results["summary"] = {
            "columns_compared": len(column_scores),
            "average_score": float(np.mean(column_scores)) if column_scores else 0,
            "min_score": float(np.min(column_scores)) if column_scores else 0,
            "max_score": float(np.max(column_scores)) if column_scores else 0,
            "rows_real": len(real_df),
            "rows_synthetic": len(synthetic_df)
        }
        
        return evaluation_results
        
    except Exception as e:
        return {
            "error": f"Basic evaluation failed: {str(e)}",
            "method": "basic_statistical_comparison"
        }


# Alternative simplified evaluation for quick checks
@mcp.tool()
async def quick_quality_check(
    real_table: str,
    synthetic_table: str
) -> Dict[str, Any]:
    """Quick quality check comparing basic statistics of real vs synthetic data.
    
    Args:
        real_table: Name of the real data table
        synthetic_table: Name of the synthetic data table
        
    Returns:
        dict: Basic quality metrics and statistics comparison
    """
    try:
        # Load data
        real_df = connection.execute(f"SELECT * FROM {real_table}").df()
        synthetic_df = connection.execute(f"SELECT * FROM {synthetic_table}").df()
        
        if real_df.empty or synthetic_df.empty:
            return {"error": "One or both tables are empty"}
        
        results = {
            "table": real_table,
            "row_counts": {
                "real": len(real_df),
                "synthetic": len(synthetic_df)
            },
            "column_comparisons": {}
        }
        
        # Compare each column
        for col in real_df.columns:
            if col not in synthetic_df.columns:
                continue
            
            if pd.api.types.is_numeric_dtype(real_df[col]):
                # Numeric column comparison
                results["column_comparisons"][col] = {
                    "type": "numeric",
                    "real_mean": float(real_df[col].mean()),
                    "synthetic_mean": float(synthetic_df[col].mean()),
                    "real_std": float(real_df[col].std()),
                    "synthetic_std": float(synthetic_df[col].std()),
                    "real_min": float(real_df[col].min()),
                    "synthetic_min": float(synthetic_df[col].min()),
                    "real_max": float(real_df[col].max()),
                    "synthetic_max": float(synthetic_df[col].max())
                }
            else:
                # Categorical column comparison
                results["column_comparisons"][col] = {
                    "type": "categorical",
                    "real_unique": int(real_df[col].nunique()),
                    "synthetic_unique": int(synthetic_df[col].nunique()),
                    "real_top_value": str(real_df[col].mode().iloc[0] if not real_df[col].mode().empty else "N/A"),
                    "synthetic_top_value": str(synthetic_df[col].mode().iloc[0] if not synthetic_df[col].mode().empty else "N/A")
                }
        
        # Calculate a simple quality score
        score = 0
        total_checks = 0
        
        for col, stats in results["column_comparisons"].items():
            if stats["type"] == "numeric":
                # Check if means are within 20% of each other
                real_mean = stats["real_mean"]
                synthetic_mean = stats["synthetic_mean"]
                if real_mean != 0:
                    mean_diff = abs(real_mean - synthetic_mean) / abs(real_mean)
                    if mean_diff < 0.2:
                        score += 1
                total_checks += 1
            else:
                # Check if unique value counts are similar
                real_unique = stats["real_unique"]
                synthetic_unique = stats["synthetic_unique"]
                if real_unique > 0:
                    unique_diff = abs(real_unique - synthetic_unique) / real_unique
                    if unique_diff < 0.3:
                        score += 1
                total_checks += 1
        
        results["quality_score"] = score / total_checks if total_checks > 0 else 0
        results["quality_rating"] = (
            "Excellent" if results["quality_score"] > 0.8 else
            "Good" if results["quality_score"] > 0.6 else
            "Fair" if results["quality_score"] > 0.4 else
            "Poor"
        )
        
        return results
        
    except Exception as e:
        return {"error": f"Quick check failed: {str(e)}"}

@mcp.tool()
async def export_synthetic_data(
    table_name: str,
    output_path: str,
    format: str = "csv"
) -> str:
    """Export synthetic data from DuckDB to file.
    
    Args:
        table_name: Name of the table to export
        output_path: Path where to save the file
        format: Export format - csv, parquet, or json (default: csv)
        
    Returns:
        str: Status message about the export
    """
    try:
        if format.lower() == "csv":
            query = f"COPY {table_name} TO '{output_path}' (HEADER, DELIMITER ',')"
        elif format.lower() == "parquet":
            query = f"COPY {table_name} TO '{output_path}' (FORMAT PARQUET)"
        elif format.lower() == "json":
            query = f"COPY {table_name} TO '{output_path}' (FORMAT JSON)"
        else:
            return f"Error: Unsupported format {format}. Use csv, parquet, or json"
        
        connection.execute(query)
        
        # Get row count
        count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        return f"Successfully exported {count} rows from {table_name} to {output_path}"
        
    except Exception as e:
        return f"Error exporting data: {str(e)}"


# Helper functions

def _get_table_creation_order(metadata: Dict[str, Any]) -> List[str]:
    """Determine the order to create/populate tables based on foreign key dependencies."""
    relationships = metadata.get('relationships', [])
    tables = set(metadata.get('tables', {}).keys())
    
    # Build dependency graph
    dependencies = {}
    for table in tables:
        dependencies[table] = set()
    
    for rel in relationships:
        child = rel.get('child_table_name')
        parent = rel.get('parent_table_name')
        if child in dependencies and parent in tables:
            dependencies[child].add(parent)
    
    # Topological sort
    ordered = []
    visited = set()
    
    def visit(table):
        if table in visited:
            return
        visited.add(table)
        for dep in dependencies.get(table, []):
            visit(dep)
        ordered.append(table)
    
    for table in tables:
        visit(table)
    
    return ordered


def _clean_metadata_for_sdv(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Clean metadata dictionary to be compatible with SDV."""
    import copy
    
    # First convert SQL to SDV format if needed
    if any('data_type' in col_info 
           for table in metadata_dict.get('tables', {}).values() 
           for col_info in table.get('columns', {}).values()):
        metadata_dict = _convert_sql_metadata_to_sdv(metadata_dict)
    
    # SDV allowed column fields
    allowed_column_fields = {
        'sdtype', 'regex_format', 'datetime_format', 
        'computer_representation', 'anonymization'
    }
    
    # Clean to ensure only allowed fields
    clean_metadata = copy.deepcopy(metadata_dict)
    
    for table_name, table_info in clean_metadata.get('tables', {}).items():
        for col_name, col_info in table_info.get('columns', {}).items():
            fields_to_remove = [
                field for field in col_info 
                if field not in allowed_column_fields
            ]
            for field in fields_to_remove:
                del col_info[field]
    
    return clean_metadata


def _convert_sql_metadata_to_sdv(metadata_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Convert SQL-based metadata to SDV format."""
    import copy
    sdv_metadata = copy.deepcopy(metadata_dict)
    
    # SQL to SDV type mapping
    sql_to_sdv_mapping = {
        'VARCHAR': 'categorical',
        'TEXT': 'text',
        'INTEGER': 'numerical',
        'INT': 'numerical',
        'DECIMAL': 'numerical',
        'DOUBLE': 'numerical',
        'FLOAT': 'numerical',
        'BOOLEAN': 'boolean',
        'BOOL': 'boolean',
        'DATE': 'datetime',
        'TIMESTAMP': 'datetime',
        'DATETIME': 'datetime',
    }
    
    # Process each table
    for table_name, table_info in sdv_metadata.get('tables', {}).items():
        for col_name, col_info in table_info.get('columns', {}).items():
            data_type = col_info.get('data_type', 'TEXT').upper()
            
            # Remove description and sample_values for SDV
            if 'description' in col_info:
                del col_info['description']
            if 'sample_values' in col_info:
                del col_info['sample_values']
            
            # Extract base type
            base_type = data_type.split('(')[0]
            
            # Determine SDV type
            if col_name.endswith('_id') or col_name == table_info.get('primary_key'):
                sdv_type = 'id'
            elif 'email' in col_name.lower():
                sdv_type = 'email'
            elif 'phone' in col_name.lower():
                sdv_type = 'phone_number'
            elif 'address' in col_name.lower() and base_type == 'TEXT':
                sdv_type = 'address'
            elif 'name' in col_name.lower() and base_type == 'VARCHAR':
                sdv_type = 'name'
            elif base_type in sql_to_sdv_mapping:
                sdv_type = sql_to_sdv_mapping[base_type]
            else:
                sdv_type = 'text'
            
            # Add datetime format for date/timestamp fields
            if base_type in ['DATE', 'TIMESTAMP', 'DATETIME']:
                if base_type == 'DATE':
                    col_info['datetime_format'] = '%Y-%m-%d'
                else:
                    col_info['datetime_format'] = '%Y-%m-%d %H:%M:%S'
            
            # Set the sdtype
            col_info['sdtype'] = sdv_type
            
            # Remove data_type as SDV doesn't use it
            if 'data_type' in col_info:
                del col_info['data_type']
    
    logger.info("Converted SQL metadata to SDV format")
    return sdv_metadata


@mcp.prompt()
async def synthetic_data_instructions() -> str:
    """Instructions for using the Synthetic Data DuckDB MCP Server."""
    return """
    ## Synthetic Data Generation with DuckDB
    
    This server provides tools for generating synthetic data directly in DuckDB:
    
    ### Available Tools:
    
    1. **list_tables**: Show all tables in the database
    2. **describe_table(table_name)**: Get structure and row count of a table
    3. **load_metadata(metadata_path)**: Load metadata JSON configuration
    4. **create_sample_tables(metadata_path, rows_per_table)**: Create and populate sample tables
    5. **generate_single_table_synthetic(source_table, num_rows, with_relationships)**: Generate synthetic data for single table
    6. **generate_multi_table_synthetic(metadata_path)**: Generate synthetic data for all tables
    7. **evaluate_synthetic_quality(real_table, synthetic_table)**: Evaluate quality of synthetic data
    8. **export_synthetic_data(table_name, output_path, format)**: Export data to files
    
    ### Workflow Examples:
    
    1. Single table generation (10 clients):
       - generate_single_table_synthetic("clients", num_rows=10, with_relationships=True)
    
    2. All tables generation:
       - generate_multi_table_synthetic("metadata.json", ensure_tables_exist=True)
    
    3. Evaluate quality:
       - evaluate_synthetic_quality("clients", "clients_synthetic")
    """


# Run the server
if __name__ == "__main__":
    logger.info("Starting Synthetic Data DuckDB MCP Server...")
    mcp.run(transport="stdio")
    logger.info("Server stopped")
    connection.close()