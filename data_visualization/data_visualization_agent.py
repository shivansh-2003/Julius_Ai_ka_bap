import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from typing import Dict, List, Any, Optional, Tuple
import re

# Import local modules
try:
    from utils import (
        generate_histogram, generate_boxplot, generate_scatterplot, generate_lineplot, 
        generate_barplot, generate_pieplot, generate_heatmap, generate_pairplot,
        generate_stacked_bar, generate_area_plot, generate_violin_plot, generate_funnel_chart,
        generate_time_series, resample_time_series, group_and_aggregate,
        filter_data, calculate_summary_stats, detect_outliers, get_data_overview
    )
    from nlp_processor import DataQueryProcessor
except ImportError:
    # If running from a different directory, adjust the import
    from data_visualization.utils import (
        generate_histogram, generate_boxplot, generate_scatterplot, generate_lineplot, 
        generate_barplot, generate_pieplot, generate_heatmap, generate_pairplot,
        generate_stacked_bar, generate_area_plot, generate_violin_plot, generate_funnel_chart,
        generate_time_series, resample_time_series, group_and_aggregate,
        filter_data, calculate_summary_stats, detect_outliers, get_data_overview
    )
    from data_visualization.nlp_processor import DataQueryProcessor

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(page_title="Data Analyst Agent", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
    }
    .stProgress > div > div {
        background-color: #1E3A8A;
    }
    .output-container {
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .code-container {
        background-color: #f7f7f7;
        padding: 15px;
        border-radius: 5px;
        font-family: monospace;
        white-space: pre-wrap;
        margin-bottom: 20px;
    }
    .divider {
        margin: 20px 0;
        border-bottom: 1px solid #e0e0e0;
    }
    .tab-container {
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("📊 Advanced Data Analyst Agent")
st.markdown("Upload your data and let AI analyze it for you. Get instant visualizations and insights to understand your data better.")

# Sidebar for model selection and settings
with st.sidebar:
    st.header("⚙️ Settings")
    llm_option = st.selectbox(
        "Select LLM Provider",
        ["OpenAI", "Anthropic", "Google"]
    )
    
    if llm_option == "OpenAI":
        openai_api_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        openai_model = st.selectbox("Model", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        os.environ["OPENAI_API_KEY"] = openai_api_key
    elif llm_option == "Anthropic":
        anthropic_api_key = st.text_input("Anthropic API Key", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))
        anthropic_model = st.selectbox("Model", ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    elif llm_option == "Google":
        google_api_key = st.text_input("Google API Key", type="password", value=os.getenv("GOOGLE_API_KEY", ""))
        google_model = st.selectbox("Model", ["gemini-pro", "gemini-ultra"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        os.environ["GOOGLE_API_KEY"] = google_api_key
    
    st.header("📋 About")
    st.markdown("""
    This tool helps you quickly visualize and analyze data using AI.
    
    **Features:**
    - Automatic data loading for various formats
    - Initial exploratory data analysis
    - Interactive visualizations
    - AI-powered insights
    - Custom visualization requests
    """)

# Set matplotlib backend to prevent non-interactive warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.ioff()  # Turn off interactive mode

# Load data based on file path
def load_data(file):
    file_name = file.name
    file_type = file_name.split('.')[-1].lower()

    try:
        if file_type == 'csv':
            df = pd.read_csv(file)
        elif file_type in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        elif file_type == 'json':
            df = pd.read_json(file)
        elif file_type == 'txt':
            # Try to infer delimiter
            df = pd.read_csv(file, sep=None, engine='python')
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# Initialize LLM based on selection
def initialize_llm():
    if llm_option == "OpenAI":
        return ChatOpenAI(
            model=openai_model,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
    elif llm_option == "Anthropic":
        return ChatAnthropic(
            model=anthropic_model,
            temperature=temperature,
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )
    else:
        # Default to OpenAI if no valid option is selected
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=os.environ.get("OPENAI_API_KEY")
        )

# Display data summary
def display_data_summary(df):
    st.subheader("📋 Data Overview")
    
    overview = get_data_overview(df)
    
    # Create a tabbed interface for data summary
    overview_tabs = st.tabs(["Basic Info", "Columns", "Sample Data", "Statistics", "Missing Values"])
    
    # Basic Info tab
    with overview_tabs[0]:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Rows:** {df.shape[0]}")
            st.write(f"**Columns:** {df.shape[1]}")
        
        with col2:
            st.write(f"**Missing Values:** {df.isna().sum().sum()}")
            st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")
        
        with col3:
            st.write(f"**Memory Usage:** {overview['memory_usage']:.2f} MB")
            st.write(f"**Data Types:** {len(df.select_dtypes(include=['int64', 'float64']).columns)} numeric, {len(df.select_dtypes(include=['object']).columns)} text")
    
    # Columns tab
    with overview_tabs[1]:
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values,
            'Null Count': df.isna().sum().values,
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Example Value': [str(df[col].iloc[0]) if len(df) > 0 else "" for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    # Sample Data tab
    with overview_tabs[2]:
        st.subheader("First 5 Rows")
        st.dataframe(df.head(), use_container_width=True)
        
        if len(df) > 5:
            st.subheader("Last 5 Rows")
            st.dataframe(df.tail(), use_container_width=True)
    
    # Statistics tab
    with overview_tabs[3]:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
            st.write("### Numeric Columns Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            # Show correlation matrix
            if len(numeric_cols) > 1:
                st.write("### Correlation Heatmap")
                corr = df[numeric_cols].corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r",
                    title="Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True, key="overview_correlation_matrix")
            else:
                st.info("No numeric columns found for statistical analysis.")
    
    # Missing Values tab
    with overview_tabs[4]:
        missing_data = df.isna().sum().reset_index()
        missing_data.columns = ['Column', 'Missing Values']
        missing_data['Percentage'] = (missing_data['Missing Values'] / len(df)) * 100
        
        if missing_data['Missing Values'].sum() > 0:
            st.write("### Missing Values by Column")
            fig = px.bar(
                missing_data[missing_data['Missing Values'] > 0], 
                x='Column', 
                y='Percentage',
                title="Percentage of Missing Values by Column"
            )
            st.plotly_chart(fig, use_container_width=True, key="missing_values_chart")
            st.dataframe(missing_data, use_container_width=True)
        else:
            st.success("No missing values found in the dataset.")
            
    # Show data quality summary
    st.subheader("🔍 Data Quality Check")
    quality_issues = []
    
    # Check for missing values
    missing_cols = df.columns[df.isna().any()].tolist()
    if missing_cols:
        quality_issues.append(f"Missing values found in {len(missing_cols)} columns: {', '.join(missing_cols[:3])}{'...' if len(missing_cols) > 3 else ''}")
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        quality_issues.append(f"Found {duplicate_count} duplicate rows ({(duplicate_count/len(df))*100:.1f}% of data)")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        quality_issues.append(f"Found {len(constant_cols)} columns with constant values: {', '.join(constant_cols[:3])}{'...' if len(constant_cols) > 3 else ''}")
    
    # Check for high cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > len(df) * 0.9:  # If more than 90% of values are unique
            quality_issues.append(f"Column '{col}' has high cardinality ({df[col].nunique()} unique values)")
    
    # Suggest data transformations if needed
    try:
        # Try to detect outliers in numeric columns
        for col in df.select_dtypes(include=['int64', 'float64']).columns[:3]:  # Check first 3 numeric cols
            outliers, lower, upper = detect_outliers(df, col)
            if len(outliers) > 0:
                quality_issues.append(f"Found {len(outliers)} potential outliers in '{col}' (below {lower:.2f} or above {upper:.2f})")
    except:
        pass  # Skip if outlier detection fails
        
    if quality_issues:
        for issue in quality_issues:
            st.warning(issue)
    else:
        st.success("No major data quality issues detected.")

# Create and display visualizations
def create_visualizations(df):
    st.subheader("📈 Comprehensive Data Visualization")
    
    # Identify column types
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    temporal_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    
    # Add date-like columns that might not be in datetime format
    for col in df.columns:
        if col not in temporal_cols and ('date' in col.lower() or 'time' in col.lower()):
            try:
                pd.to_datetime(df[col])
                temporal_cols.append(col)
            except:
                pass
    
    # Create a tabbed interface for different visualization types
    tabs = st.tabs([
        "Distribution", "Correlation", "Relationship", 
        "Time Series", "Categorical", "Advanced"
    ])
    
    # Distribution tab - histograms and box plots
    with tabs[0]:
        if numeric_cols:
            st.write("### Numerical Distributions")
            selected_num_col = st.selectbox("Select column for distribution analysis", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                # Histogram with density curve
                fig = generate_histogram(df, selected_num_col)
                st.plotly_chart(fig, use_container_width=True, key=f"histogram_{selected_num_col}")
            
            with col2:
                # Box plot
                fig = generate_boxplot(df, selected_num_col)
                st.plotly_chart(fig, use_container_width=True, key=f"boxplot_{selected_num_col}")
            
            # Show statistics for the selected column
            st.write(f"### Statistics for {selected_num_col}")
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q1', 'Q3'],
                'Value': [
                    f"{df[selected_num_col].mean():.2f}",
                    f"{df[selected_num_col].median():.2f}",
                    f"{df[selected_num_col].std():.2f}",
                    f"{df[selected_num_col].min():.2f}",
                    f"{df[selected_num_col].max():.2f}",
                    f"{df[selected_num_col].quantile(0.25):.2f}",
                    f"{df[selected_num_col].quantile(0.75):.2f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True)
            
            # Show violin plot if categorical columns exist
            if categorical_cols:
                st.write("### Distribution by Category")
                cat_col = st.selectbox("Select category", categorical_cols, key="dist_cat_col")
                fig = generate_violin_plot(df, cat_col, selected_num_col)
                st.plotly_chart(fig, use_container_width=True, key=f"violin_{cat_col}_{selected_num_col}")
        else:
            st.info("No numerical columns available for distribution analysis.")
    
    # Correlation tab - correlation matrix and heatmaps
    with tabs[1]:
        if len(numeric_cols) > 1:
            st.write("### Correlation Analysis")
            
            # Correlation method selection
            corr_method = st.radio("Correlation Method", ["Pearson", "Spearman"], horizontal=True)
            
            # Select columns for correlation
            selected_cols = st.multiselect("Select columns for correlation analysis", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
            
            if selected_cols and len(selected_cols) > 1:
                # Plot correlation heatmap
                corr = df[selected_cols].corr(method=corr_method.lower())
                fig = generate_heatmap(df, selected_cols)
                st.plotly_chart(fig, use_container_width=True, key="correlation_matrix")
                
                # Correlation table
                st.write("### Correlation Table")
                st.dataframe(corr.style.background_gradient(cmap='coolwarm'), use_container_width=True)
                
                # Show pair plot for deeper analysis
                st.write("### Pair Plot")
                if len(selected_cols) <= 4:  # Limit to 4 columns for readability
                    fig = generate_pairplot(df, selected_cols)
                    st.plotly_chart(fig, use_container_width=True, key="pair_plot")
                else:
                    st.warning("Select 4 or fewer columns for pair plot visualization")
            else:
                st.warning("Please select at least 2 columns for correlation analysis")
        else:
            st.info("Need at least 2 numerical columns for correlation analysis.")
    
    # Relationship tab - scatter plots and regression
    with tabs[2]:
        if len(numeric_cols) > 1:
            st.write("### Relationship Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X axis", numeric_cols)
            with col2:
                y_col = st.selectbox("Y axis", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            # Optional color by category
            color_col = None
            if categorical_cols:
                color_col = st.selectbox("Color by (optional)", ["None"] + categorical_cols)
                if color_col == "None":
                    color_col = None
            
            # Create scatter plot with trend line
            fig = generate_scatterplot(df, x_col, y_col, color_col)
            
            # Add trendline if no color grouping
            if color_col is None:
                fig.update_layout(title=f"Scatter Plot with Trend Line: {x_col} vs {y_col}")
                fig.add_traces(
                    px.scatter(df, x=x_col, y=y_col, trendline="ols").data[1]
                )
            
            st.plotly_chart(fig, use_container_width=True, key=f"scatter_{x_col}_{y_col}")
            
            # Calculate and show correlation
            corr_value = df[[x_col, y_col]].corr().iloc[0, 1]
            st.write(f"Correlation between {x_col} and {y_col}: **{corr_value:.4f}**")
            
            # Interpret correlation
            if abs(corr_value) > 0.7:
                st.write("This indicates a **strong** correlation.")
            elif abs(corr_value) > 0.3:
                st.write("This indicates a **moderate** correlation.")
            else:
                st.write("This indicates a **weak** correlation.")
        else:
            st.info("Need at least 2 numerical columns for relationship analysis.")
    
    # Time Series tab
    with tabs[3]:
        if temporal_cols:
            st.write("### Time Series Analysis")
            
            # Select date column
            date_col = st.selectbox("Select date column", temporal_cols)
            
            # Ensure date column is in datetime format
            df_copy = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
                try:
                    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                except:
                    st.warning(f"Could not convert {date_col} to datetime format.")
                    return
            
            if numeric_cols:
                # Select value column
                value_col = st.selectbox("Select value column", numeric_cols)
                
                # Optional group by
                group_col = None
                if categorical_cols:
                    group_col = st.selectbox("Group by (optional)", ["None"] + categorical_cols)
                    if group_col == "None":
                        group_col = None
                
                # Time series line chart
                fig = generate_time_series(df_copy, date_col, value_col, group_col)
                st.plotly_chart(fig, use_container_width=True, key=f"line_{value_col}_over_time")
                
                # Resample by time period
                if len(df_copy[date_col].unique()) > 10:
                    st.write("### Resampled Time Series")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        resample_period = st.selectbox(
                            "Resample period",
                            ["Day", "Week", "Month", "Quarter", "Year"],
                            index=2
                        )
                    
                    with col2:
                        agg_func = st.selectbox("Aggregation function", ["mean", "sum", "min", "max"])
                    
                    # Create mapping for time periods
                    period_map = {
                        "Day": "D",
                        "Week": "W",
                        "Month": "M",
                        "Quarter": "Q",
                        "Year": "Y"
                    }
                    
                    # Resample time series
                    resampled = resample_time_series(df_copy, date_col, value_col, 
                                                     period=period_map[resample_period], 
                                                     agg_func=agg_func)
                    
                    if not resampled.empty and value_col in resampled:
                        fig = px.line(
                            resampled, 
                            x=date_col, 
                            y=value_col,
                            title=f"{value_col} ({agg_func}) by {resample_period}"
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"resampled_{value_col}_{resample_period}")
                    else:
                        st.warning("Could not resample data with selected parameters.")
            else:
                st.info("No numeric columns available for time series analysis.")
        else:
            st.info("No datetime columns detected for time series analysis.")
    
    # Categorical tab
    with tabs[4]:
        if categorical_cols:
            st.write("### Categorical Data Analysis")
            
            # Select categorical column
            cat_col = st.selectbox("Select categorical column", categorical_cols)
            
            # Count and percentage distribution
            count_df = df[cat_col].value_counts().reset_index()
            count_df.columns = [cat_col, 'Count']
            count_df['Percentage'] = count_df['Count'] / count_df['Count'].sum() * 100
            
            # Bar chart of category counts
            fig = px.bar(
                count_df,
                x=cat_col,
                y='Count',
                title=f"Count of {cat_col}",
                text='Count'
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True, key=f"bar_count_{cat_col}")
            
            # Pie chart of category distribution
            fig = px.pie(
                count_df,
                names=cat_col,
                values='Count',
                title=f"Distribution of {cat_col}"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"pie_{cat_col}")
            
            # Show count table
            st.write(f"### {cat_col} Value Counts")
            st.dataframe(count_df, use_container_width=True)
            
            # Relationship with numeric columns
            if numeric_cols:
                st.write("### Categorical vs Numerical")
                
                # Select numeric column
                num_col = st.selectbox("Select numerical column", numeric_cols)
                
                # Box plot showing distribution by category
                fig = generate_boxplot(df, cat_col, num_col)
                st.plotly_chart(fig, use_container_width=True, key=f"box_{cat_col}_{num_col}")
                
                # Bar chart of average value by category
                agg_df = df.groupby(cat_col)[num_col].mean().reset_index()
                fig = px.bar(
                    agg_df,
                    x=cat_col,
                    y=num_col,
                    title=f"Average {num_col} by {cat_col}",
                    text_auto='.2f'
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True, key=f"bar_avg_{cat_col}_{num_col}")
        else:
            st.info("No categorical columns available for analysis.")
    
    # Advanced visualizations tab
    with tabs[5]:
        st.write("### Advanced Visualizations")
        
        # Select visualization type
        viz_type = st.selectbox(
            "Select visualization type",
            ["Stacked Bar Chart", "Area Chart", "Funnel Chart", "Violin Plot", "Treemap"]
        )
        
        if viz_type == "Stacked Bar Chart":
            if categorical_cols and len(categorical_cols) >= 2 and numeric_cols:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_col = st.selectbox("X-axis (Base Category)", categorical_cols)
                
                with col2:
                    color_col = st.selectbox("Stack By", [c for c in categorical_cols if c != x_col], 
                                             key="stack_color")
                
                with col3:
                    y_col = st.selectbox("Value", numeric_cols)
                
                # Create a stacked bar chart
                agg_df = df.groupby([x_col, color_col])[y_col].mean().reset_index()
                fig = generate_stacked_bar(agg_df, x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True, key=f"stacked_bar_{x_col}_{color_col}")
            else:
                st.warning("Need at least 2 categorical columns and 1 numeric column for stacked bar charts.")
        
        elif viz_type == "Area Chart":
            if temporal_cols and numeric_cols:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    x_col = st.selectbox("X-axis (Date)", temporal_cols)
                
                with col2:
                    y_col = st.selectbox("Y-axis (Value)", numeric_cols, key="area_y")
                
                with col3:
                    color_col = None
                    if categorical_cols:
                        color_option = st.selectbox("Group By (optional)", ["None"] + categorical_cols)
                        if color_option != "None":
                            color_col = color_option
                
                # Create area chart
                fig = generate_area_plot(df, x_col, y_col, color_col)
                st.plotly_chart(fig, use_container_width=True, key=f"area_{x_col}_{y_col}")
            else:
                st.warning("Need at least 1 date column and 1 numeric column for area charts.")
        
        elif viz_type == "Funnel Chart":
            if categorical_cols and numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("Category", categorical_cols, key="funnel_x")
                
                with col2:
                    y_col = st.selectbox("Value", numeric_cols, key="funnel_y")
                
                # Create funnel chart
                count_df = df.groupby(x_col)[y_col].sum().reset_index().sort_values(y_col, ascending=False)
                fig = generate_funnel_chart(count_df, x_col, y_col)
                st.plotly_chart(fig, use_container_width=True, key=f"funnel_{x_col}_{y_col}")
            else:
                st.warning("Need at least 1 categorical column and 1 numeric column for funnel charts.")
        
        elif viz_type == "Violin Plot":
            if categorical_cols and numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    x_col = st.selectbox("Category", categorical_cols, key="violin_x")
                
                with col2:
                    y_col = st.selectbox("Value", numeric_cols, key="violin_y")
                
                # Create violin plot
                fig = generate_violin_plot(df, x_col, y_col)
                st.plotly_chart(fig, use_container_width=True, key=f"violin_{x_col}_{y_col}")
            else:
                st.warning("Need at least 1 categorical column and 1 numeric column for violin plots.")
        
        elif viz_type == "Treemap":
            if categorical_cols and numeric_cols:
                # For treemap, we can use multiple categorical columns for hierarchy
                path_cols = st.multiselect("Hierarchy (Path)", categorical_cols, 
                                           default=[categorical_cols[0]] if categorical_cols else [])
                
                value_col = st.selectbox("Value", numeric_cols, key="treemap_value")
                
                if path_cols:
                    # Create treemap
                    fig = px.treemap(
                        df, 
                        path=path_cols,
                        values=value_col,
                        title=f"Treemap of {value_col} by {', '.join(path_cols)}"
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"treemap_{value_col}")
                else:
                    st.warning("Please select at least one column for the hierarchy.")
            else:
                st.warning("Need at least 1 categorical column and 1 numeric column for treemaps.")

# Get AI insights using LangChain
def get_ai_insights(df, llm):
    st.subheader("🤖 AI-Generated Insights")
    
    with st.spinner("Generating insights... This may take a moment."):
        try:
            # Create pandas agent
            agent = create_pandas_dataframe_agent(llm, df, agent_type="tool-calling", verbose=False, allow_dangerous_code=True)
            
            # Get insights about the data
            response = agent.invoke(
                "Analyze this dataset comprehensively and provide the following insights:\n"
                "1. Key statistics and patterns\n"
                "2. Unusual or interesting findings\n"
                "3. Potential data quality issues\n"
                "4. Relationships between variables\n"
                "5. Business or practical implications of the data\n"
                "Keep your analysis concise but insightful."
            )
            
            # Create a nice output container
            st.markdown('<div class="output-container">', unsafe_allow_html=True)
            st.markdown("### Key Insights")
            st.markdown(response['output'])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Add Python agent for visualization code generation
            python_agent = create_python_agent(
                llm,
                tool=PythonREPLTool(),
                verbose=False,
                allow_dangerous_code=True  # Opt-in for executing arbitrary code
            )
            
            # Get specialized insights for specific column types
            column_types = {}
            if len(df.select_dtypes(include=['int64', 'float64']).columns) > 0:
                column_types['numeric'] = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if len(df.select_dtypes(include=['object', 'category']).columns) > 0:
                column_types['categorical'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Show detailed insights by data type
            if 'numeric' in column_types and len(column_types['numeric']) > 0:
                with st.expander("📊 Numerical Column Insights"):
                    num_prompt = f"""
                    Analyze the numerical columns in this dataset: {', '.join(column_types['numeric'][:5])}
                    
                    For each column, provide:
                    1. A brief description of the distribution (e.g., normal, skewed, bimodal)
                    2. Potential outliers and their impact
                    3. Key statistical features (mean, median, range, etc.)
                    
                    Focus on the most important insights, be concise and use bullet points.
                    """
                    
                    with st.spinner("Analyzing numerical columns..."):
                        num_response = agent.invoke(num_prompt)
                        st.markdown(num_response['output'])
            
            if 'categorical' in column_types and len(column_types['categorical']) > 0:
                with st.expander("📊 Categorical Column Insights"):
                    cat_prompt = f"""
                    Analyze the categorical columns in this dataset: {', '.join(column_types['categorical'][:5])}
                    
                    For each column, provide:
                    1. Distribution of top categories
                    2. Interesting patterns or imbalances
                    3. Potential issues (e.g., too many categories, missing values)
                    
                    Focus on the most important insights, be concise and use bullet points.
                    """
                    
                    with st.spinner("Analyzing categorical columns..."):
                        cat_response = agent.invoke(cat_prompt)
                        st.markdown(cat_response['output'])
            
            # Generate correlation insights if multiple numeric columns exist
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                with st.expander("📊 Correlation Insights"):
                    corr_prompt = f"""
                    Analyze the correlations between the numerical variables in this dataset.
                    
                    Focus on:
                    1. The strongest positive and negative correlations
                    2. Surprising or counter-intuitive relationships
                    3. Potential causal relationships (be careful not to overstate causation)
                    
                    Be concise and use bullet points.
                    """
                    
                    with st.spinner("Analyzing correlations..."):
                        corr_response = agent.invoke(corr_prompt)
                        st.markdown(corr_response['output'])
            
            # Get visualization recommendations
            with st.expander("🔍 Recommended Visualizations"):
                viz_prompt = f"""
                Based on the following pandas DataFrame information:
                
                Columns: {', '.join(df.columns.tolist())}
                Data types: {dict(df.dtypes.astype(str))}
                
                Suggest 3-5 specific visualizations that would best reveal insights in this dataset.
                For each visualization:
                1. Explain what insight it would reveal
                2. Specify which columns to use and the visualization type
                3. Provide Python code using plotly express to create the visualization
                
                The code should be ready to run and should include only the visualization code (assume the DataFrame is already loaded as 'df').
                """
                
                with st.spinner("Generating visualization recommendations..."):
                    viz_response = python_agent.invoke(viz_prompt)
                    st.markdown(viz_response['output'])
                    
                    # Try to extract and display some of the recommended charts
                    try:
                        # Simple code extraction - this is a very basic approach
                        code_blocks = re.findall(r'```python(.*?)```', viz_response['output'], re.DOTALL)
                        if code_blocks:
                            for i, code in enumerate(code_blocks[:2]):  # Show first 2 charts
                                st.write(f"### Visualization {i+1}")
                                with st.spinner("Generating visualization..."):
                                    # Create a temporary environment to safely execute the code
                                    local_vars = {'df': df, 'px': px, 'np': np, 'pd': pd}
                                    try:
                                        # Try to extract the fig assignment line
                                        fig_line = re.search(r'fig\s*=\s*px\..*\n', code)
                                        if fig_line:
                                            exec(fig_line.group(0), globals(), local_vars)
                                            if 'fig' in local_vars:
                                                st.plotly_chart(local_vars['fig'], use_container_width=True)
                                    except Exception as e:
                                        st.error(f"Could not generate visualization: {str(e)}")
                    except:
                        pass  # Skip if visualization extraction fails
            
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            st.error("Try installing tabulate with: pip install tabulate")
            return None

# Custom query section
def custom_query_section(df, llm):
    st.header("🔍 Custom Query Builder")
    
    # Add unique keys to selectbox elements
    query_type = st.selectbox("Select Query Type", 
        ["Natural Language", "SQL-like", "Advanced Query"], 
        key="query_type_selectbox")
    
    if query_type == "Natural Language":
        st.markdown("**Ask a question about your data in natural language**")
        natural_query = st.text_input("Enter your query", key="natural_query_input")
        
        if st.button("Generate Insights", key="natural_query_button"):
            if natural_query and df is not None:
                # Process natural language query
                insights = process_natural_language_query(df, natural_query, llm)
                st.write(insights)
    
    elif query_type == "SQL-like":
        st.markdown("**Write a query using SQL-like syntax**")
        sql_query = st.text_area("Enter your SQL-like query", key="sql_query_textarea")
        
        if st.button("Execute Query", key="sql_query_button"):
            if sql_query and df is not None:
                # Process SQL-like query
                result = process_sql_like_query(df, sql_query)
                
                # Display result
                st.write("### Query Results")
                st.dataframe(result)
                
                # Visualize result if possible
                if len(result) > 0:
                    st.write("### Visualization of Results")
                    
                    # Determine best visualization based on result shape
                    num_cols = result.select_dtypes(include=['int64', 'float64']).columns
                    cat_cols = result.select_dtypes(include=['object', 'category']).columns
                    
                    # Case 1: One category column and one numeric column - Bar chart
                    if len(cat_cols) == 1 and len(num_cols) == 1:
                        fig = px.bar(
                            result, 
                            x=cat_cols[0], 
                            y=num_cols[0],
                            title=f"{num_cols[0]} by {cat_cols[0]}",
                            text_auto='.2f'
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True, key="sql_bar_chart")
                    
                    # Case 2: One numeric column - Histogram
                    elif len(num_cols) == 1:
                        fig = px.histogram(
                            result, 
                            x=num_cols[0],
                            title=f"Distribution of {num_cols[0]}",
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="sql_histogram")
                    
                    # Case 3: Two numeric columns - Scatter plot
                    elif len(num_cols) >= 2:
                        fig = px.scatter(
                            result, 
                            x=num_cols[0], 
                            y=num_cols[1],
                            title=f"Relationship between {num_cols[0]} and {num_cols[1]}",
                            trendline="ols" if len(result) > 2 else None
                        )
                        st.plotly_chart(fig, use_container_width=True, key="sql_scatter")
                    
                    # Case 4: More complex result - Multiple visualizations
                    elif len(result.columns) > 2:
                        # Offer different visualization options
                        viz_type = st.selectbox(
                            "Select visualization type",
                            ["Table View", "Correlation Heatmap", "Pair Plot"],
                            key="sql_viz_type"
                        )
                        
                        if viz_type == "Correlation Heatmap" and len(num_cols) > 1:
                            corr = result[num_cols].corr()
                            fig = px.imshow(
                                corr,
                                text_auto=True,
                                color_continuous_scale="RdBu_r",
                                title="Correlation Matrix"
                            )
                            st.plotly_chart(fig, use_container_width=True, key="sql_corr_heatmap")
                        
                        elif viz_type == "Pair Plot" and len(num_cols) > 1:
                            fig = px.scatter_matrix(
                                result,
                                dimensions=num_cols[:4],  # Limit to 4 dimensions for readability
                                title="Pair Plot"
                            )
                            st.plotly_chart(fig, use_container_width=True, key="sql_pair_plot")
    
    elif query_type == "Advanced Query":
        st.markdown("**Advanced Data Manipulation**")
        
        # Filtering
        filter_column = st.selectbox("Filter Column", 
            ["None"] + list(df.columns), 
            key="filter_column_selectbox")
        filter_condition = st.text_input("Filter Condition (e.g., > 10)", 
            key="filter_condition_input")
        
        # Sorting
        sort_column = st.selectbox("Sort Column", 
            ["None"] + list(df.columns), 
            key="sort_column_selectbox")
        sort_order = st.selectbox("Sort Order", 
            ["Ascending", "Descending"], 
            key="sort_order_selectbox")
        
        # Aggregation
        agg_column = st.selectbox("Aggregation Column", 
            ["None"] + list(df.select_dtypes(include=['int64', 'float64']).columns), 
            key="agg_column_selectbox")
        agg_method = st.selectbox("Aggregation Method", 
            ["None", "Sum", "Mean", "Median", "Min", "Max"], 
            key="agg_method_selectbox")
        
        # Group by (optional)
        group_by = st.selectbox("Group By (optional)", 
            ["None"] + list(df.select_dtypes(include=['object', 'category']).columns), 
            key="group_by_selectbox")
        
        if st.button("Apply Advanced Query", key="advanced_query_button"):
            result_df = df.copy()
            
            # Apply filtering
            if filter_column != "None" and filter_condition:
                try:
                    result_df = result_df.query(f"{filter_column} {filter_condition}")
                except Exception as e:
                    st.error(f"Error in filtering: {e}")
            
            # Apply grouping and aggregation
            if group_by != "None" and agg_column != "None" and agg_method != "None":
                agg_funcs = {
                    "Sum": "sum",
                    "Mean": "mean", 
                    "Median": "median", 
                    "Min": "min", 
                    "Max": "max"
                }
                try:
                    result_df = result_df.groupby(group_by)[agg_column].agg(agg_funcs[agg_method]).reset_index()
                except Exception as e:
                    st.error(f"Error in aggregation: {e}")
            
            # Apply sorting
            if sort_column != "None":
                ascending = sort_order == "Ascending"
                result_df = result_df.sort_values(by=sort_column, ascending=ascending)
            
            # Display result
            st.write("### Query Results")
            st.dataframe(result_df)
            
            # Visualize result
            if len(result_df) > 0:
                st.write("### Visualization of Results")
                
                # Choose appropriate visualization based on query type
                if group_by != "None" and agg_column != "None" and agg_method != "None":
                    # Aggregation result - Bar chart
                    fig = px.bar(
                        result_df,
                        x=group_by,
                        y=agg_column,
                        title=f"{agg_method} of {agg_column} by {group_by}",
                        text_auto='.2f'
                    )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True, key="advanced_bar_chart")
                
                elif filter_column != "None" and filter_condition:
                    # Filtered result - Show appropriate chart based on columns
                    numeric_cols = result_df.select_dtypes(include=['int64', 'float64']).columns
                    
                    if len(numeric_cols) >= 2:
                        # Scatter plot for two numeric columns
                        col1, col2 = st.columns(2)
                        with col1:
                            x_col = st.selectbox("X-axis", numeric_cols, key="adv_scatter_x")
                        with col2:
                            y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col], key="adv_scatter_y")
                        
                        fig = px.scatter(
                            result_df,
                            x=x_col,
                            y=y_col,
                            title=f"Scatter Plot: {x_col} vs {y_col}",
                            trendline="ols"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="advanced_scatter")
                    
                    elif len(numeric_cols) == 1:
                        # Histogram for one numeric column
                        fig = px.histogram(
                            result_df,
                            x=numeric_cols[0],
                            title=f"Distribution of {numeric_cols[0]}",
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="advanced_histogram")
                
                # For general results, show summary statistics
                if len(result_df.select_dtypes(include=['int64', 'float64']).columns) > 0:
                    st.write("### Summary Statistics")
                    st.dataframe(result_df.describe())
                    
                    # Correlation heatmap for numeric columns if multiple exist
                    num_cols = result_df.select_dtypes(include=['int64', 'float64']).columns
                    if len(num_cols) > 1:
                        st.write("### Correlation Heatmap")
                        corr = result_df[num_cols].corr()
                        fig = px.imshow(
                            corr,
                            text_auto=True,
                            color_continuous_scale="RdBu_r",
                            title="Correlation Matrix"
                        )
                        st.plotly_chart(fig, use_container_width=True, key="advanced_corr_heatmap")

# Generate Python code for analysis
def generate_code_section(df):
    st.subheader("🧪 Generate Python Code")
    
    # Create tabs for different code generation options
    code_tabs = st.tabs(["Data Analysis", "Visualization", "Data Transformation"])
    
    # Data Analysis tab
    with code_tabs[0]:
        st.write("Generate code for data analysis")
        
        analysis_options = [
            "Basic exploratory data analysis",
            "Correlation analysis",
            "Statistical tests",
            "Outlier detection",
            "Custom analysis"
        ]
        
        analysis_type = st.selectbox("Select analysis type", analysis_options)
        
        # Additional options based on analysis type
        if analysis_type == "Correlation analysis":
            corr_cols = st.multiselect("Select columns for correlation", 
                                      df.select_dtypes(include=['int64', 'float64']).columns.tolist())
        
        elif analysis_type == "Statistical tests":
            test_type = st.selectbox("Select test type", ["t-test", "ANOVA", "Chi-square"])
            # Additional test-specific options would go here
        
        elif analysis_type == "Custom analysis":
            custom_desc = st.text_area("Describe the analysis you want to perform")
        
        if st.button("Generate Code", key="analysis_code_btn"):
            with st.spinner("Generating code..."):
                st.code("""
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
# df = pd.read_csv('your_file.csv')  # Replace with your data loading code

# Basic exploratory data analysis
print("Data shape:", df.shape)
print("\\nData types:")
print(df.dtypes)
print("\\nSummary statistics:")
print(df.describe())

# Check for missing values
print("\\nMissing values:")
print(df.isnull().sum())

# Visualize distributions of numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()
    plt.show()

# Correlation analysis
if len(numeric_columns) > 1:
    plt.figure(figsize=(10, 8))
    corr = df[numeric_columns].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
                """, language="python")
    
    # Visualization tab
    with code_tabs[1]:
        st.write("Generate code for data visualization")
        
        viz_options = [
            "Bar chart",
            "Scatter plot",
            "Line chart",
            "Histogram",
            "Box plot",
            "Heatmap",
            "Pie chart",
            "Pair plot"
        ]
        
        viz_type = st.selectbox("Select visualization type", viz_options)
        
        # Additional options based on visualization type
        if viz_type == "Bar chart":
            x_col = st.selectbox("X-axis column", df.columns.tolist())
            y_col = st.selectbox("Y-axis column (optional)", ["None"] + df.select_dtypes(include=['int64', 'float64']).columns.tolist())
        
        elif viz_type == "Scatter plot":
            scatter_x = st.selectbox("X-axis column", df.select_dtypes(include=['int64', 'float64']).columns.tolist())
            scatter_y = st.selectbox("Y-axis column", [c for c in df.select_dtypes(include=['int64', 'float64']).columns if c != scatter_x])
        
        if st.button("Generate Code", key="viz_code_btn"):
            with st.spinner("Generating code..."):
                if viz_type == "Bar chart":
                    if y_col == "None":
                        code = f"""
import plotly.express as px

# Create a simple count-based bar chart
fig = px.bar(df, x='{x_col}', title='Count of {x_col}')
fig.update_layout(
    xaxis_title='{x_col}',
    yaxis_title='Count',
    template='plotly_white'
)
fig.show()
                        """
                    else:
                        code = f"""
import plotly.express as px

# Create a bar chart with values
fig = px.bar(df, x='{x_col}', y='{y_col}', title='{y_col} by {x_col}')
fig.update_layout(
    xaxis_title='{x_col}',
    yaxis_title='{y_col}',
    template='plotly_white'
)
fig.show()
                        """
                    st.code(code, language="python")
                
                elif viz_type == "Scatter plot":
                    code = f"""
import plotly.express as px

# Create a scatter plot
fig = px.scatter(df, x='{scatter_x}', y='{scatter_y}', 
                title='Relationship between {scatter_x} and {scatter_y}',
                opacity=0.7)
fig.update_layout(
    xaxis_title='{scatter_x}',
    yaxis_title='{scatter_y}',
    template='plotly_white'
)

# Add trendline
fig.update_traces(mode='markers')
fig.add_traces(
    px.scatter(df, x='{scatter_x}', y='{scatter_y}', trendline='ols').data[1]
)
fig.show()
                    """
                    st.code(code, language="python")
    
    # Data Transformation tab
    with code_tabs[2]:
        st.write("Generate code for data transformation")
        
        transform_options = [
            "Filtering data",
            "Grouping and aggregation",
            "Handling missing values",
            "Creating new columns",
            "Data type conversion"
        ]
        
        transform_type = st.selectbox("Select transformation type", transform_options)
        
        if st.button("Generate Code", key="transform_code_btn"):
            with st.spinner("Generating code..."):
                if transform_type == "Filtering data":
                    code = """
# Filter data based on a condition
filtered_df = df[df['column_name'] > value]

# Filter data based on multiple conditions
filtered_df = df[(df['column1'] > value1) & (df['column2'] == value2)]

# Filter data based on string matching
filtered_df = df[df['text_column'].str.contains('pattern')]

# Get the top N values by a certain column
top_n = df.nlargest(10, 'value_column')
                    """
                elif transform_type == "Grouping and aggregation":
                    code = """
# Group by one column and calculate mean of other columns
grouped_df = df.groupby('category_column').mean().reset_index()

# Group by multiple columns with different aggregations
grouped_df = df.groupby(['category1', 'category2']).agg({
    'numeric_col1': 'mean',
    'numeric_col2': 'sum',
    'numeric_col3': 'max'
}).reset_index()

# Group and count occurrences
count_df = df.groupby('category_column').size().reset_index(name='count')
                    """
                else:
                    code = """
# Basic data transformation example
# Replace with specific transformation code
                    """
                
                st.code(code, language="python")

# Main app function
def run_app():
    # Initialize session state if needed
    if 'history' not in st.session_state:
        st.session_state['history'] = []
    
    # Create a tabbed interface for different app sections
    main_tabs = st.tabs([
        "📤 Data Upload", 
        "📊 Exploration",
        "🤖 AI Analysis",
        "🔍 Query Builder",
        "📝 Code Generation"
    ])
    
    # Data Upload tab
    with main_tabs[0]:
        st.header("Upload Your Data")
        
        # File uploader with multiple file type support
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=["csv", "xlsx", "xls", "json", "txt"],
            help="Upload your data file (CSV, Excel, JSON, or TXT)"
        )
        
        # URL input as an alternative
        url_input = st.text_input(
            "Or enter a URL to a data file:",
            placeholder="https://example.com/data.csv"
        )
        
        # Sample dataset selection
        st.markdown("### Or choose a sample dataset:")
        
        demo_options = [
            "Health Smartwatch Data (clean)",
            "Health Smartwatch Data (unclean)",
            "Iris Flower Dataset",
            "Titanic Passenger Data",
            "Boston Housing Prices",
            "Medical Insurance Costs"
        ]
        demo_selection = st.selectbox("Select a sample dataset:", ["None"] + demo_options)
        
        # Process data source
        data_loaded = False
        
        if uploaded_file is not None:
            # Show loading spinner while processing
            with st.spinner("Loading and analyzing data..."):
                # Load data from uploaded file
                df = load_data(uploaded_file)
                
                if df is not None:
                    st.success(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                    st.session_state['df'] = df
                    st.session_state['data_source'] = f"Uploaded file: {uploaded_file.name}"
                    data_loaded = True
                else:
                    st.error("Failed to load data. Please check your file and try again.")
        
        elif url_input:
            # Try to load data from URL
            with st.spinner("Loading data from URL..."):
                try:
                    if url_input.endswith('.csv'):
                        df = pd.read_csv(url_input)
                    elif url_input.endswith(('.xls', '.xlsx')):
                        df = pd.read_excel(url_input)
                    elif url_input.endswith('.json'):
                        df = pd.read_json(url_input)
                    else:
                        st.error("Unsupported file type. Please provide a URL to a CSV, Excel, or JSON file.")
                        df = None
                    
                    if df is not None:
                        st.success(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
                        st.session_state['df'] = df
                        st.session_state['data_source'] = f"URL: {url_input}"
                        data_loaded = True
                except Exception as e:
                    st.error(f"Error loading data from URL: {str(e)}")
        
        elif demo_selection != "None" and st.button("Load Sample Data"):
            with st.spinner("Loading sample data..."):
                if demo_selection == "Health Smartwatch Data (clean)":
                    try:
                        demo_data = pd.read_csv("sample_data/unclean_smartwatch_health_data_cleaned.csv")
                        data_source = "Health Smartwatch Data (clean)"
                    except:
                        st.error("Sample data file not found. Falling back to default dataset.")
                        demo_data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
                        data_source = "Default Tips Dataset"
                
                elif demo_selection == "Health Smartwatch Data (unclean)":
                    try:
                        demo_data = pd.read_csv("sample_data/unclean_smartwatch_health_data.csv")
                        data_source = "Health Smartwatch Data (unclean)"
                    except:
                        st.error("Sample data file not found. Falling back to default dataset.")
                        demo_data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
                        data_source = "Default Tips Dataset"
                
                elif demo_selection == "Iris Flower Dataset":
                    demo_data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
                    data_source = "Iris Flower Dataset"
                
                elif demo_selection == "Titanic Passenger Data":
                    demo_data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
                    data_source = "Titanic Passenger Data"
                
                elif demo_selection == "Medical Insurance Costs":
                    demo_data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
                    data_source = "Medical Insurance Costs"
                
                elif demo_selection == "Boston Housing Prices":
                    # Load Boston Housing dataset
                    from sklearn.datasets import load_boston
                    boston = load_boston()
                    demo_data = pd.DataFrame(boston.data, columns=boston.feature_names)
                    demo_data['PRICE'] = boston.target
                    data_source = "Boston Housing Prices"
                
                else:
                    # Fallback to tips dataset
                    demo_data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv")
                    data_source = "Default Tips Dataset"
                
                st.success(f"Sample data loaded successfully: {demo_data.shape[0]} rows, {demo_data.shape[1]} columns")
                st.session_state['df'] = demo_data
                st.session_state['data_source'] = f"Sample dataset: {data_source}"
                data_loaded = True
        
        # Data loaded section
        if data_loaded or 'df' in st.session_state:
            df = st.session_state['df']
            data_source = st.session_state.get('data_source', 'Unknown source')
            
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Add data to history if it's new
            if data_loaded and 'data_source' in st.session_state:
                # Add to history only if it's a new source
                if not any(h['source'] == st.session_state['data_source'] for h in st.session_state['history']):
                    st.session_state['history'].append({
                        'source': st.session_state['data_source'],
                        'shape': df.shape,
                        'columns': df.columns.tolist()
                    })
            
            # Show history in an expander
            with st.expander("Data Loading History"):
                for i, entry in enumerate(st.session_state['history']):
                    st.write(f"{i+1}. **{entry['source']}** - {entry['shape'][0]} rows, {entry['shape'][1]} columns")
        
        # Show a message if no data is loaded
        if not data_loaded and 'df' not in st.session_state:
            st.info("Please upload a file, enter a URL, or select a sample dataset to begin analysis.")
    
    # Only show the other tabs if data is loaded
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        # Exploration tab
        with main_tabs[1]:
            # Show data overview
            display_data_summary(df)
            
            # Create visualizations
            create_visualizations(df)
        
        # AI Analysis tab
        with main_tabs[2]:
            # Initialize LLM
            llm = initialize_llm()
            
            # Get AI insights
            get_ai_insights(df, llm)
        
        # Query Builder tab
        with main_tabs[3]:
            # Custom query section
            custom_query_section(df, llm)
        
        # Code Generation tab
        with main_tabs[4]:
            # Code generation section
            generate_code_section(df)
    
    # If no data is loaded, disable the other tabs
    else:
        for i in range(1, 5):
            with main_tabs[i]:
                st.info("Please load data in the 'Data Upload' tab to enable this feature.")

def process_natural_language_query(df, query, llm):
    """
    Process a natural language query about the dataframe using the LLM.
    
    Args:
        df (pd.DataFrame): The dataframe to query
        query (str): Natural language query
        llm: Language model to use
        
    Returns:
        str: Response from the LLM
    """
    try:
        # Create pandas agent with verbose=False to suppress detailed logs
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            agent_type="tool-calling", 
            verbose=False, 
            allow_dangerous_code=True
        )
        
        # Enhanced prompt for better results
        prompt = f"""
        Here is a natural language query about a pandas DataFrame: "{query}"
        
        Columns in the DataFrame: {', '.join(df.columns.tolist())}
        
        Please:
        1. Understand the user's intent
        2. Perform the necessary analysis or query on the DataFrame
        3. If appropriate, suggest a visualization
        4. Provide a clear, concise answer with relevant insights
        5. If possible, generate Python code for the analysis
        
        Be specific and use the DataFrame's actual columns and data.
        """
        
        # Invoke the agent
        response = agent.invoke(prompt)
        
        # Extract the output
        result = response.get('output', 'No insights generated.')
        
        # Try to generate a visualization if possible
        try:
            # Look for Python code blocks in the response
            code_blocks = re.findall(r'```python(.*?)```', result, re.DOTALL)
            if code_blocks:
                with st.expander("Generated Visualization", expanded=True):
                    for i, code in enumerate(code_blocks[:1]):  # Just show first viz
                        # Create a temporary environment to safely execute the code
                        local_vars = {
                            'df': df, 
                            'px': px, 
                            'np': np, 
                            'pd': pd, 
                            'plt': plt, 
                            'sns': sns
                        }
                        try:
                            # Execute the code
                            exec(code, globals(), local_vars)
                            if 'fig' in local_vars:
                                st.plotly_chart(local_vars['fig'], use_container_width=True, key=f"nl_query_viz_{i}")
                            elif 'plt' in locals():
                                st.pyplot(plt)
                        except Exception as e:
                            st.warning(f"Could not generate visualization: {str(e)}")
        except:
            pass  # Skip if visualization extraction fails
        
        return result
    
    except Exception as e:
        st.error(f"Error processing natural language query: {str(e)}")
        return f"Error: {str(e)}"

if __name__ == "__main__":
    run_app()
