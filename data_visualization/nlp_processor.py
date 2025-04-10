import pandas as pd
import re
from typing import Dict, List, Any, Tuple, Optional

class DataQueryProcessor:
    """
    Process natural language queries into data analysis operations.
    This is a simple version - in production, you would use an LLM for this.
    """
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with a dataframe"""
        self.df = df
        self.numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        self.datetime_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        self.all_cols = df.columns.tolist()
    
    def extract_column_mentions(self, query: str) -> List[str]:
        """Find column names mentioned in the query"""
        mentioned_cols = []
        for col in self.all_cols:
            if col.lower() in query.lower():
                mentioned_cols.append(col)
        return mentioned_cols
    
    def detect_operation_type(self, query: str) -> str:
        """Determine the main operation requested"""
        query = query.lower()
        
        # Detect visualization requests
        viz_keywords = ['plot', 'chart', 'graph', 'visualize', 'show me', 'display']
        for keyword in viz_keywords:
            if keyword in query:
                return self._detect_visualization_type(query)
        
        # Detect aggregation/grouping
        agg_keywords = ['average', 'mean', 'sum', 'count', 'max', 'min', 'group by', 'group']
        for keyword in agg_keywords:
            if keyword in query:
                return 'aggregation'
        
        # Detect filtering
        filter_keywords = ['where', 'filter', 'only', 'excluding', 'greater than', 'less than', 'equal to']
        for keyword in filter_keywords:
            if keyword in query:
                return 'filtering'
        
        # Detect simple statistics
        stat_keywords = ['describe', 'statistics', 'summary', 'distribution']
        for keyword in stat_keywords:
            if keyword in query:
                return 'statistics'
        
        # Detect sorting
        sort_keywords = ['sort', 'order', 'rank', 'top', 'bottom']
        for keyword in sort_keywords:
            if keyword in query:
                return 'sorting'
        
        # Default to simple query
        return 'query'
    
    def _detect_visualization_type(self, query: str) -> str:
        """Determine the visualization type requested"""
        query = query.lower()
        
        if 'histogram' in query:
            return 'histogram'
        elif 'bar' in query:
            return 'bar'
        elif 'scatter' in query:
            return 'scatter'
        elif 'line' in query:
            return 'line'
        elif 'pie' in query:
            return 'pie'
        elif 'box' in query:
            return 'box'
        elif 'heatmap' in query or 'correlation' in query:
            return 'heatmap'
        elif 'pair' in query:
            return 'pair'
        elif 'area' in query:
            return 'area'
        elif 'violin' in query:
            return 'violin'
        elif 'funnel' in query:
            return 'funnel'
        elif 'stacked' in query:
            return 'stacked_bar'
        
        # Default to generic visualization
        return 'visualization'
    
    def extract_aggregation_info(self, query: str) -> Tuple[str, List[str], List[str]]:
        """Extract aggregation function and columns"""
        query = query.lower()
        
        # Determine aggregation function
        if 'average' in query or 'mean' in query:
            agg_func = 'mean'
        elif 'sum' in query or 'total' in query:
            agg_func = 'sum'
        elif 'max' in query or 'maximum' in query or 'highest' in query:
            agg_func = 'max'
        elif 'min' in query or 'minimum' in query or 'lowest' in query:
            agg_func = 'min'
        elif 'count' in query:
            agg_func = 'count'
        else:
            agg_func = 'mean'  # Default
        
        # Extract group by columns (categorical)
        group_cols = []
        if 'group by' in query or 'grouped by' in query:
            for col in self.categorical_cols:
                if col.lower() in query.lower():
                    group_cols.append(col)
        
        # Extract aggregation columns (numeric)
        agg_cols = []
        for col in self.numeric_cols:
            if col.lower() in query.lower():
                agg_cols.append(col)
        
        return agg_func, group_cols, agg_cols
    
    def extract_filter_info(self, query: str) -> List[Dict[str, Any]]:
        """Extract filtering conditions"""
        query = query.lower()
        filters = []
        
        # Check for each column in filtering context
        for col in self.all_cols:
            col_lower = col.lower()
            
            # Look for the column name in the query
            if col_lower in query:
                # Check for various condition patterns
                # Equal to
                match = re.search(f"{col_lower}\\s*(is|=|equal to|equals|==)\\s*['\"]?([\\w\\d\\s.]+)['\"]?", query)
                if match:
                    filters.append({
                        'column': col,
                        'condition': '==',
                        'value': match.group(2).strip()
                    })
                    continue
                
                # Greater than
                match = re.search(f"{col_lower}\\s*(>|greater than)\\s*([\\d.]+)", query)
                if match:
                    filters.append({
                        'column': col,
                        'condition': '>',
                        'value': float(match.group(2))
                    })
                    continue
                
                # Less than
                match = re.search(f"{col_lower}\\s*(<|less than)\\s*([\\d.]+)", query)
                if match:
                    filters.append({
                        'column': col,
                        'condition': '<',
                        'value': float(match.group(2))
                    })
                    continue
                
                # Contains
                match = re.search(f"{col_lower}\\s*(contains|has)\\s*['\"]?([\\w\\d\\s.]+)['\"]?", query)
                if match:
                    filters.append({
                        'column': col,
                        'condition': 'contains',
                        'value': match.group(2).strip()
                    })
                    continue
        
        return filters
    
    def extract_visualization_info(self, query: str, viz_type: str) -> Dict[str, Any]:
        """Extract visualization parameters"""
        query = query.lower()
        result = {'type': viz_type}
        
        # Different visualizations need different parameters
        if viz_type in ['histogram', 'box']:
            # Need one numeric column
            for col in self.numeric_cols:
                if col.lower() in query:
                    result['x'] = col
                    break
        
        elif viz_type == 'bar':
            # Need one categorical and potentially one numeric
            for col in self.categorical_cols:
                if col.lower() in query:
                    result['x'] = col
                    break
            
            for col in self.numeric_cols:
                if col.lower() in query:
                    result['y'] = col
                    break
        
        elif viz_type in ['scatter', 'line']:
            # Need two numeric columns
            found_cols = []
            for col in self.numeric_cols:
                if col.lower() in query:
                    found_cols.append(col)
                    if len(found_cols) == 2:
                        break
            
            if len(found_cols) >= 2:
                result['x'] = found_cols[0]
                result['y'] = found_cols[1]
            
            # Check for color by
            for col in self.categorical_cols:
                if col.lower() in query and 'by' in query and col.lower() in query.split('by')[1]:
                    result['color'] = col
                    break
        
        elif viz_type == 'pie':
            # Need one categorical and one numeric
            for col in self.categorical_cols:
                if col.lower() in query:
                    result['names'] = col
                    break
            
            for col in self.numeric_cols:
                if col.lower() in query:
                    result['values'] = col
                    break
        
        elif viz_type == 'heatmap':
            # Uses all numeric columns by default or specified ones
            mentioned_cols = []
            for col in self.numeric_cols:
                if col.lower() in query:
                    mentioned_cols.append(col)
            
            if mentioned_cols:
                result['columns'] = mentioned_cols
        
        return result
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and return structured representation"""
        operation_type = self.detect_operation_type(query)
        result = {'operation': operation_type}
        
        if operation_type == 'aggregation':
            agg_func, group_cols, agg_cols = self.extract_aggregation_info(query)
            result['agg_func'] = agg_func
            result['group_cols'] = group_cols
            result['agg_cols'] = agg_cols
        
        elif operation_type == 'filtering':
            result['filters'] = self.extract_filter_info(query)
        
        elif operation_type in ['histogram', 'bar', 'scatter', 'line', 'pie', 'box', 'heatmap', 
                              'pair', 'area', 'violin', 'funnel', 'stacked_bar', 'visualization']:
            result.update(self.extract_visualization_info(query, operation_type))
        
        return result
    
    def execute_query(self, query: str) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """Execute the query and return results"""
        query_info = self.process_query(query)
        operation = query_info['operation']
        result_df = None
        
        if operation == 'aggregation':
            # Perform aggregation
            if query_info['group_cols'] and query_info['agg_cols']:
                result_df = self.df.groupby(query_info['group_cols'])[query_info['agg_cols']].agg(query_info['agg_func']).reset_index()
            elif query_info['agg_cols']:
                result_df = pd.DataFrame({
                    col: [self.df[col].agg(query_info['agg_func'])] 
                    for col in query_info['agg_cols']
                })
            else:
                # If no columns specified, use all numeric columns
                result_df = pd.DataFrame({
                    col: [self.df[col].agg(query_info['agg_func'])] 
                    for col in self.numeric_cols
                })
        
        elif operation == 'filtering':
            # Apply filters
            filtered_df = self.df.copy()
            for filter_info in query_info.get('filters', []):
                column = filter_info['column']
                condition = filter_info['condition']
                value = filter_info['value']
                
                if condition == '==':
                    # Handle string conversions if needed
                    if filtered_df[column].dtype == 'object':
                        filtered_df = filtered_df[filtered_df[column].astype(str) == str(value)]
                    else:
                        filtered_df = filtered_df[filtered_df[column] == value]
                elif condition == '>':
                    filtered_df = filtered_df[filtered_df[column] > value]
                elif condition == '<':
                    filtered_df = filtered_df[filtered_df[column] < value]
                elif condition == 'contains':
                    filtered_df = filtered_df[filtered_df[column].astype(str).str.contains(str(value))]
            
            result_df = filtered_df
        
        elif operation == 'statistics':
            # Generate statistics for numeric columns
            result_df = self.df.describe()
        
        elif operation == 'sorting':
            # Very basic sorting - would need to parse sort columns and directions
            # Just using first numeric column as an example
            if self.numeric_cols:
                result_df = self.df.sort_values(by=self.numeric_cols[0], ascending=False)
            else:
                result_df = self.df
        
        else:
            # For visualization and other operations, just return the dataframe
            result_df = self.df
        
        return result_df, query_info 