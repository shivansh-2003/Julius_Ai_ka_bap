import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Visualization functions
def generate_histogram(df, column, title=None, color=None, nbins=30):
    """Generate a histogram for a numerical column"""
    fig = px.histogram(df, x=column, color=color, nbins=nbins, 
                      title=title or f"Distribution of {column}", 
                      marginal="box")
    return fig

def generate_boxplot(df, x_col, y_col=None, color=None, title=None):
    """Generate a box plot"""
    if y_col:
        fig = px.box(df, x=x_col, y=y_col, color=color, 
                    title=title or f"Distribution of {y_col} by {x_col}")
    else:
        fig = px.box(df, x=x_col, color=color, 
                    title=title or f"Box Plot of {x_col}")
    return fig

def generate_scatterplot(df, x_col, y_col, color=None, size=None, title=None):
    """Generate a scatter plot between two numerical columns"""
    fig = px.scatter(df, x=x_col, y=y_col, color=color, size=size, 
                    title=title or f"Scatter Plot: {x_col} vs {y_col}",
                    opacity=0.7, hover_data=df.columns)
    return fig

def generate_lineplot(df, x_col, y_col, color=None, title=None, markers=True):
    """Generate a line plot"""
    fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, color=color, 
                 title=title or f"{y_col} over {x_col}", markers=markers)
    return fig

def generate_barplot(df, x_col, y_col, color=None, title=None, orientation='v'):
    """Generate a bar plot"""
    fig = px.bar(df, x=x_col, y=y_col, color=color, 
                title=title or f"{y_col} by {x_col}", orientation=orientation)
    return fig

def generate_pieplot(df, names, values, title=None, hole=0.3):
    """Generate a pie chart"""
    fig = px.pie(df, names=names, values=values, hole=hole,
                title=title or f"{values} Distribution by {names}")
    return fig

def generate_heatmap(df, columns=None, title=None):
    """Generate a correlation heatmap"""
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    corr = df[columns].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto",
                   color_continuous_scale="RdBu_r", 
                   title=title or "Correlation Matrix")
    return fig

def generate_pairplot(df, columns=None, dimensions=None, color=None, title="Pair Plot"):
    """Generate a pair plot matrix"""
    if columns is None:
        columns = df.select_dtypes(include=['int64', 'float64']).columns[:4]  # Limit to 4 columns for readability
    
    if dimensions is None:
        dimensions = columns
    
    fig = px.scatter_matrix(df, dimensions=dimensions, color=color)
    fig.update_layout(title=title)
    return fig

def generate_stacked_bar(df, x_col, y_col, color, title=None):
    """Generate a stacked bar chart"""
    fig = px.bar(df, x=x_col, y=y_col, color=color, 
                title=title or f"Stacked {y_col} by {x_col}", 
                barmode='stack')
    return fig

def generate_area_plot(df, x_col, y_col, color=None, title=None):
    """Generate an area plot"""
    fig = px.area(df.sort_values(x_col), x=x_col, y=y_col, color=color,
                 title=title or f"{y_col} over {x_col}")
    return fig

def generate_violin_plot(df, x_col, y_col, color=None, title=None, box=True):
    """Generate a violin plot"""
    fig = px.violin(df, x=x_col, y=y_col, color=color, box=box,
                   title=title or f"Distribution of {y_col} by {x_col}")
    return fig

def generate_funnel_chart(df, x_col, y_col, title=None):
    """Generate a funnel chart"""
    fig = px.funnel(df, x=x_col, y=y_col,
                   title=title or f"Funnel Chart: {y_col} by {x_col}")
    return fig

def generate_time_series(df, date_col, value_col, color=None, title=None):
    """Generate a time series plot"""
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    
    fig = px.line(df.sort_values(date_col), x=date_col, y=value_col, color=color,
                 title=title or f"{value_col} over Time")
    return fig

def resample_time_series(df, date_col, value_col, period='M', agg_func='mean'):
    """Resample time series data"""
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    # Set index and resample
    df_resampled = df_copy.set_index(date_col)
    
    # Apply aggregation
    if agg_func == 'mean':
        result = df_resampled.resample(period).mean()
    elif agg_func == 'sum':
        result = df_resampled.resample(period).sum()
    elif agg_func == 'min':
        result = df_resampled.resample(period).min()
    elif agg_func == 'max':
        result = df_resampled.resample(period).max()
    else:
        result = df_resampled.resample(period).mean()
    
    return result.reset_index()

# Data processing functions
def group_and_aggregate(df, group_cols, agg_cols, agg_func='mean'):
    """Group data and apply aggregation"""
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    if isinstance(agg_cols, str):
        agg_cols = [agg_cols]
    
    # Build the aggregation dictionary
    agg_dict = {col: agg_func for col in agg_cols}
    
    # Perform the groupby operation
    result = df.groupby(group_cols).agg(agg_dict).reset_index()
    return result

def filter_data(df, column, condition, value):
    """Filter data based on condition"""
    if condition == 'equals' or condition == '==':
        return df[df[column] == value]
    elif condition == 'not equals' or condition == '!=':
        return df[df[column] != value]
    elif condition == 'greater than' or condition == '>':
        return df[df[column] > value]
    elif condition == 'less than' or condition == '<':
        return df[df[column] < value]
    elif condition == 'greater than or equals' or condition == '>=':
        return df[df[column] >= value]
    elif condition == 'less than or equals' or condition == '<=':
        return df[df[column] <= value]
    elif condition == 'contains':
        return df[df[column].astype(str).str.contains(str(value))]
    elif condition == 'starts with':
        return df[df[column].astype(str).str.startswith(str(value))]
    elif condition == 'ends with':
        return df[df[column].astype(str).str.endswith(str(value))]
    else:
        return df

def calculate_summary_stats(df, columns=None):
    """Calculate summary statistics for a dataframe"""
    if columns is None:
        # Only use numeric columns for statistics
        columns = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Calculate statistics
    stats = df[columns].describe()
    return stats

def detect_outliers(df, column, method='iqr', threshold=1.5):
    """Detect outliers in a column"""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    elif method == 'z-score':
        from scipy import stats
        z_scores = stats.zscore(df[column])
        outliers = df[abs(z_scores) > threshold]
        return outliers
    else:
        return df[df[column] > df[column].mean() + threshold * df[column].std()]

def get_data_overview(df):
    """Generate a comprehensive overview of the dataframe"""
    overview = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isna().sum().to_dict(),
        'unique_values': {col: df[col].nunique() for col in df.columns},
        'memory_usage': df.memory_usage(deep=True).sum() / (1024 * 1024),  # In MB
    }
    return overview

def clean_column_names(df):
    """Clean column names for easier usage"""
    df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    return df

def decode_nl_query(query, df):
    """
    Attempt to decode natural language query into pandas operations
    This is a basic implementation - in production you would use an LLM
    """
    # Extract column names for easier reference
    columns = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    query = query.lower()
    
    # Example patterns (very simple - a real implementation would use NLP/LLM)
    results = {
        'operation': None,
        'columns': [],
        'params': {}
    }
    
    # Average/mean detection
    if 'average' in query or 'mean' in query:
        results['operation'] = 'aggregate'
        results['params']['func'] = 'mean'
        
        # Try to identify columns
        for col in numeric_cols:
            if col.lower() in query:
                results['columns'].append(col)
    
    # Count detection
    elif 'count' in query or 'how many' in query:
        results['operation'] = 'aggregate'
        results['params']['func'] = 'count'
        
        # Try to identify columns
        for col in columns:
            if col.lower() in query:
                results['columns'].append(col)
    
    # Group by detection
    if 'group by' in query or 'grouped by' in query:
        results['operation'] = 'group'
        
        # Try to identify grouping columns
        for col in categorical_cols:
            if col.lower() in query:
                if 'group_by' not in results['params']:
                    results['params']['group_by'] = []
                results['params']['group_by'].append(col)
    
    # Filter detection
    if 'where' in query or 'filter' in query:
        if 'filters' not in results['params']:
            results['params']['filters'] = []
        
        # Very basic filter detection
        for col in columns:
            if col.lower() in query:
                # Check for various conditions
                if f'{col.lower()} greater than' in query or f'{col.lower()} >' in query:
                    # This is just a placeholder - a real implementation would
                    # parse the actual values from the query
                    results['params']['filters'].append({
                        'column': col,
                        'condition': '>',
                        'value': 'PLACEHOLDER'
                    })
    
    # Visualization detection
    if 'plot' in query or 'chart' in query or 'graph' in query or 'visualize' in query:
        # Detect chart type
        if 'histogram' in query:
            results['operation'] = 'visualize'
            results['params']['type'] = 'histogram'
        elif 'bar' in query:
            results['operation'] = 'visualize'
            results['params']['type'] = 'bar'
        elif 'scatter' in query:
            results['operation'] = 'visualize'
            results['params']['type'] = 'scatter'
        elif 'line' in query:
            results['operation'] = 'visualize'
            results['params']['type'] = 'line'
        elif 'pie' in query:
            results['operation'] = 'visualize'
            results['params']['type'] = 'pie'
        elif 'box' in query:
            results['operation'] = 'visualize'
            results['params']['type'] = 'box'
        elif 'heat' in query:
            results['operation'] = 'visualize'
            results['params']['type'] = 'heatmap'
        elif 'correlation' in query:
            results['operation'] = 'visualize'
            results['params']['type'] = 'heatmap'
    
    return results 