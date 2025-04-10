import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import re
from scipy import stats

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema.runnable import RunnablePassthrough
from dotenv import load_dotenv
import os
import json

# Load environment variables from .env file
load_dotenv()

def get_llm(model_name: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    """
    Create and return a ChatOpenAI LLM with specified model and temperature.
    """
    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=os.getenv("OPENAI_API_KEY", "")
    )

llm = get_llm()

# Define prompt for LLM to identify data cleaning issues
analysis_prompt_template = PromptTemplate(
    input_variables=["data_head", "metadata"],
    template="""
    You are an experienced data analyst. Given the dataset preview and metadata below, identify potential data cleaning problems and suggest clear actionable steps for cleaning.

    Dataset Preview:
    {data_head}

    Dataset Metadata:
    {metadata}

    Clearly outline steps needed for cleaning:
    1.
    2.
    3.
    (and so on...)
    """
)

# Define a structured prompt for the LLM to generate executable cleaning operations
cleaning_prompt_template = PromptTemplate(
    input_variables=["data_head", "metadata"],
    template="""
    You are an experienced data scientist tasked with cleaning a dataset. Given the dataset preview and metadata below, 
    identify data cleaning operations needed and return ONLY a JSON response with the cleaning steps.
    
    Dataset Preview:
    {data_head}
    
    Dataset Metadata:
    {metadata}
    
    Return a JSON object with the following structure:
    ```json
    {{
        "operations": [
            {{
                "operation": "handle_missing_values",
                "columns": ["column1", "column2"],
                "method": "mean|median|mode|remove_rows|fill_value",
                "fill_value": "value_if_applicable" 
            }},
            {{
                "operation": "fix_data_types",
                "columns": ["column1"],
                "target_type": "float|int|str|category"
            }},
            {{
                "operation": "handle_outliers",
                "columns": ["column1"],
                "method": "zscore|iqr",
                "threshold": 3.0,
                "action": "remove|cap"
            }},
            {{
                "operation": "standardize_values",
                "column": "column_name",
                "replacements": {{
                    "old_value1": "new_value1",
                    "old_value2": "new_value2"
                }}
            }},
            {{
                "operation": "drop_columns",
                "columns": ["column1"]
            }},
            {{
                "operation": "drop_duplicates",
                "subset": ["column1", "column2"] 
            }}
        ]
    }}
    ```
    
    Important: Only include operations that are necessary based on the data. Return ONLY the JSON object without any additional text.
    """
)

# Modern approach using RunnableSequence instead of LLMChain
analysis_chain = analysis_prompt_template | llm
cleaning_chain = cleaning_prompt_template | llm

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV with a simple check for file existence.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df

def get_data_preview(df: pd.DataFrame, num_rows: int = 5) -> str:
    """Return the first few rows as string."""
    return df.head(num_rows).to_string()

def get_metadata(df: pd.DataFrame) -> dict:
    """Get basic metadata about the dataset."""
    metadata = {
        "Number of rows": len(df),
        "Number of columns": len(df.columns),
        "Column names": df.columns.tolist(),
        "Data types": df.dtypes.astype(str).to_dict(),
        "Missing values per column": df.isnull().sum().to_dict(),
        "Unique values per column": {col: df[col].nunique() for col in df.columns},
        "Sample values": {col: df[col].dropna().sample(min(5, df[col].count())).tolist() 
                         for col in df.columns}
    }
    return metadata

def get_cleaning_analysis(df: pd.DataFrame) -> str:
    """Get human-readable data cleaning suggestions from OpenAI GPT."""
    data_head = get_data_preview(df)
    metadata = get_metadata(df)

    suggestions = analysis_chain.invoke({
        "data_head": data_head,
        "metadata": str(metadata)
    })

    # Extract the content from AIMessage object
    if hasattr(suggestions, 'content'):
        return suggestions.content.strip()
    
    # Fallback in case it's already a string
    return str(suggestions).strip()

def get_cleaning_operations(df: pd.DataFrame) -> Dict:
    """Get structured cleaning operations from OpenAI GPT."""
    data_head = get_data_preview(df)
    metadata = get_metadata(df)

    response = cleaning_chain.invoke({
        "data_head": data_head,
        "metadata": str(metadata)
    })
    
    content = response.content if hasattr(response, 'content') else str(response)
    
    # Extract JSON from the response
    try:
        # Try to directly parse the content
        operations = json.loads(content)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON using regex
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                operations = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                # If still failing, create an empty operations object
                operations = {"operations": []}
        else:
            operations = {"operations": []}
    
    return operations

# Data cleaning functions
def handle_missing_values(df: pd.DataFrame, columns: List[str], method: str = 'median', fill_value: Optional[Any] = None) -> pd.DataFrame:
    """Handle missing values in specified columns."""
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            print(f"Warning: Column '{col}' not found in dataframe")
            continue
            
        if method == 'mean' and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col].fillna(df_copy[col].mean(), inplace=True)
        elif method == 'median' and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col].fillna(df_copy[col].median(), inplace=True)
        elif method == 'mode':
            df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else "Unknown", inplace=True)
        elif method == 'fill_value' and fill_value is not None:
            df_copy[col].fillna(fill_value, inplace=True)
        elif method == 'remove_rows':
            df_copy = df_copy.dropna(subset=[col])
    
    return df_copy

def fix_data_types(df: pd.DataFrame, columns: List[str], target_type: str) -> pd.DataFrame:
    """Convert columns to the specified data type."""
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns:
            print(f"Warning: Column '{col}' not found in dataframe")
            continue
            
        if target_type == 'float':
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype(float)
        elif target_type == 'int':
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('Int64')  # Use Int64 to handle NaN
        elif target_type == 'str':
            df_copy[col] = df_copy[col].astype(str)
        elif target_type == 'category':
            df_copy[col] = df_copy[col].astype('category')
    
    return df_copy

def handle_outliers(df: pd.DataFrame, columns: List[str], method: str = 'zscore', threshold: float = 3.0, action: str = 'cap') -> pd.DataFrame:
    """Handle outliers in specified columns."""
    df_copy = df.copy()
    
    for col in columns:
        if col not in df_copy.columns or not pd.api.types.is_numeric_dtype(df_copy[col]):
            print(f"Warning: Column '{col}' not found or not numeric")
            continue
            
        if method == 'zscore':
            z_scores = np.abs(stats.zscore(df_copy[col].dropna()))
            outlier_indices = df_copy[col].dropna().index[z_scores > threshold]
            if action == 'remove':
                df_copy = df_copy.drop(outlier_indices)
            elif action == 'cap':
                # Calculate bounds using z-score method
                col_mean = df_copy[col].mean()
                col_std = df_copy[col].std()
                lower_bound = col_mean - (threshold * col_std)
                upper_bound = col_mean + (threshold * col_std)
                # Cap values
                df_copy.loc[df_copy[col] < lower_bound, col] = lower_bound
                df_copy.loc[df_copy[col] > upper_bound, col] = upper_bound
        
        elif method == 'iqr':
            q1, q3 = df_copy[col].quantile(0.25), df_copy[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_indices = df_copy[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)].index
            if action == 'remove':
                df_copy = df_copy.drop(outlier_indices)
            elif action == 'cap':
                df_copy.loc[df_copy[col] < lower_bound, col] = lower_bound
                df_copy.loc[df_copy[col] > upper_bound, col] = upper_bound
    
    return df_copy

def standardize_values(df: pd.DataFrame, column: str, replacements: Dict[str, str]) -> pd.DataFrame:
    """Standardize values in a column based on a replacement dictionary."""
    df_copy = df.copy()
    
    if column not in df_copy.columns:
        print(f"Warning: Column '{column}' not found in dataframe")
        return df_copy
        
    df_copy[column] = df_copy[column].replace(replacements)
    return df_copy

def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Drop specified columns from the dataframe."""
    return df.drop(columns=columns, errors='ignore')

def drop_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Drop duplicate rows from the dataframe."""
    return df.drop_duplicates(subset=subset)

def auto_clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, str]:
    """
    Automatically clean the dataframe based on LLM suggestions.
    Returns:
    - cleaned dataframe
    - dictionary of operations performed
    - human-readable analysis
    """
    # Get cleaning analysis in human-readable format
    analysis = get_cleaning_analysis(df)
    
    # Get structured cleaning operations
    operations_dict = get_cleaning_operations(df)
    operations = operations_dict.get('operations', [])
    
    # Keep track of changes
    cleaned_df = df.copy()
    applied_operations = []
    
    # Apply each cleaning operation
    for op in operations:
        operation_type = op.get('operation')
        
        try:
            if operation_type == 'handle_missing_values':
                columns = op.get('columns', [])
                method = op.get('method', 'median')
                fill_value = op.get('fill_value')
                cleaned_df = handle_missing_values(cleaned_df, columns, method, fill_value)
                applied_operations.append(f"Handled missing values in {columns} using {method}")
                
            elif operation_type == 'fix_data_types':
                columns = op.get('columns', [])
                target_type = op.get('target_type', 'float')
                cleaned_df = fix_data_types(cleaned_df, columns, target_type)
                applied_operations.append(f"Fixed data types in {columns} to {target_type}")
                
            elif operation_type == 'handle_outliers':
                columns = op.get('columns', [])
                method = op.get('method', 'zscore')
                threshold = op.get('threshold', 3.0)
                action = op.get('action', 'cap')
                cleaned_df = handle_outliers(cleaned_df, columns, method, threshold, action)
                applied_operations.append(f"Handled outliers in {columns} using {method} with threshold {threshold} and action {action}")
                
            elif operation_type == 'standardize_values':
                column = op.get('column', '')
                replacements = op.get('replacements', {})
                cleaned_df = standardize_values(cleaned_df, column, replacements)
                applied_operations.append(f"Standardized values in {column}")
                
            elif operation_type == 'drop_columns':
                columns = op.get('columns', [])
                cleaned_df = drop_columns(cleaned_df, columns)
                applied_operations.append(f"Dropped columns: {columns}")
                
            elif operation_type == 'drop_duplicates':
                subset = op.get('subset')
                cleaned_df = drop_duplicates(cleaned_df, subset)
                applied_operations.append(f"Dropped duplicates based on {subset if subset else 'all columns'}")
        
        except Exception as e:
            applied_operations.append(f"Error applying {operation_type}: {str(e)}")
    
    return cleaned_df, {"operations": operations, "applied": applied_operations}, analysis

def main():
    file_path = "/Users/shivanshmahajan/Desktop/data/sample_data/unclean_smartwatch_health_data.csv"
    df = load_data(file_path)

    print("\n=== Initial Data Preview ===\n")
    print(get_data_preview(df))

    print("\n=== Dataset Metadata ===\n")
    metadata = get_metadata(df)
    for key, value in metadata.items():
        if key != "Sample values":  # Skip showing sample values to keep output clean
            print(f"{key}: {value}")

    print("\n=== Starting Automatic Data Cleaning ===\n")
    cleaned_df, operations, analysis = auto_clean_data(df)
    
    print("\n=== Human-Readable Cleaning Analysis ===\n")
    print(analysis)
    
    print("\n=== Operations Applied ===\n")
    for idx, op in enumerate(operations["applied"], 1):
        print(f"{idx}. {op}")
    
    print("\n=== Cleaned Data Preview ===\n")
    print(get_data_preview(cleaned_df))
    
    # Show cleaning impact statistics
    print("\n=== Cleaning Impact ===\n")
    print(f"Rows before cleaning: {len(df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Missing values before: {df.isna().sum().sum()}")
    print(f"Missing values after: {cleaned_df.isna().sum().sum()}")
    
    # Ask if user wants to drop any columns
    user_decision = input("\nDo you want to drop any columns? (yes/no): ")
    if user_decision.lower() == "yes":
        cols_to_drop = input("Enter columns to drop (comma-separated): ")
        cols_list = [col.strip() for col in cols_to_drop.split(",")]
        cleaned_df = drop_columns(cleaned_df, cols_list)
        print("\nColumns dropped successfully. Updated data preview:\n")
        print(get_data_preview(cleaned_df))
    
    # Ask if user wants to standardize values
    user_decision = input("\nDo you want to standardize any values? (yes/no): ")
    if user_decision.lower() == "yes":
        column_to_standardize = input("Enter the column to standardize: ")
        replacements_input = input("Enter the replacements in the format 'old_value1:new_value1, old_value2:new_value2': ")
        replacements = dict(item.split(":") for item in replacements_input.split(","))
        cleaned_df = standardize_values(cleaned_df, column_to_standardize, replacements)
        print(f"\nValues in '{column_to_standardize}' standardized successfully. Updated data preview:\n")
        print(get_data_preview(cleaned_df))
    
    # Save the cleaned data
    output_path = os.path.splitext(file_path)[0] + "_cleaned.csv"
    cleaned_df.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    
    # Ask if user wants to see more detailed stats
    user_decision = input("\nDo you want to see more detailed statistics about the cleaning? (yes/no): ")
    if user_decision.lower() == "yes":
        print("\n=== Detailed Statistics ===\n")
        print("Before cleaning:")
        print(df.describe().to_string())
        print("\nAfter cleaning:")
        print(cleaned_df.describe().to_string())

if __name__ == "__main__":
    main()
