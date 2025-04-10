# app.py
import streamlit as st
import pandas as pd
import numpy as np
import json
import io
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.formula.api import ols
import openai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
import base64

st.set_page_config(page_title="Smart Hypothesis Tester", layout="wide")
st.title("🧪 Smart Statistical Test Runner")

# Initialize session state variables if they don't exist
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'step' not in st.session_state:
    st.session_state['step'] = 1
if 'test_results' not in st.session_state:
    st.session_state['test_results'] = None
if 'assumptions' not in st.session_state:
    st.session_state['assumptions'] = None
if 'visualization' not in st.session_state:
    st.session_state['visualization'] = None

# Function to check assumptions
def check_assumptions(df, columns, test_type):
    results = {}
    
    if test_type == "t-test" or test_type == "ANOVA":
        # Check normality (Shapiro-Wilk test)
        for col in columns:
            if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                # Only run on numeric columns with sufficient data
                if len(df[col].dropna()) >= 3 and len(df[col].dropna()) <= 5000:  # Shapiro has limits
                    stat, p = stats.shapiro(df[col].dropna())
                    results[f"Normality ({col})"] = {
                        "test": "Shapiro-Wilk",
                        "statistic": stat,
                        "p-value": p,
                        "interpretation": "Normal distribution" if p > 0.05 else "Not normally distributed"
                    }
                
        # Check homogeneity of variance (Levene test) for multiple groups
        if test_type == "ANOVA" and len(columns) >= 2:
            groups = [df[col].dropna() for col in columns if col in df.columns]
            if all(len(g) > 0 for g in groups):
                stat, p = stats.levene(*groups)
                results["Homogeneity of Variance"] = {
                    "test": "Levene's test",
                    "statistic": stat,
                    "p-value": p,
                    "interpretation": "Equal variances" if p > 0.05 else "Unequal variances"
                }
                
    elif test_type == "chi-square":
        # Check minimum expected frequency
        if len(columns) == 2 and all(col in df.columns for col in columns):
            contingency_table = pd.crosstab(df[columns[0]], df[columns[1]])
            chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
            min_expected = expected.min()
            results["Minimum Expected Frequency"] = {
                "value": min_expected,
                "interpretation": "Adequate" if min_expected >= 5 else "Too small, results may be unreliable"
            }
    
    return results

# Function to create visualizations
def create_visualization(df, columns, test_type, results=None):
    if test_type == "t-test":
        if len(columns) == 2 and all(col in df.columns for col in columns):
            fig = px.box(df, y=columns, title="Box Plot Comparison")
            return fig
            
    elif test_type == "ANOVA":
        if len(columns) >= 2 and all(col in df.columns for col in columns):
            fig = px.box(pd.melt(df[columns]), x="variable", y="value", title="ANOVA Box Plot Comparison")
            return fig
            
    elif test_type == "chi-square":
        if len(columns) == 2 and all(col in df.columns for col in columns):
            contingency_table = pd.crosstab(df[columns[0]], df[columns[1]], normalize='index')
            fig = px.imshow(contingency_table, 
                           labels=dict(x=columns[1], y=columns[0], color="Proportion"),
                           title="Heatmap of Proportions")
            return fig
            
    elif test_type == "correlation":
        if len(columns) == 2 and all(col in df.columns for col in columns):
            fig = px.scatter(df, x=columns[0], y=columns[1], trendline="ols",
                           title=f"Correlation between {columns[0]} and {columns[1]}")
            return fig
    
    # Default case - just show columns in a table
    fig = go.Figure(data=[go.Table(
        header=dict(values=columns),
        cells=dict(values=[df[col] for col in columns if col in df.columns])
    )])
    fig.update_layout(title="Data Preview")
    return fig

# Function to generate a downloadable report
def generate_report(test_results, assumptions, df, user_prompt):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    title = Paragraph("Statistical Test Report", styles['Title'])
    elements.append(title)
    elements.append(Spacer(1, 12))
    
    # User query
    elements.append(Paragraph("User Query:", styles['Heading2']))
    elements.append(Paragraph(user_prompt, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Test information
    elements.append(Paragraph("Test Information:", styles['Heading2']))
    elements.append(Paragraph(f"Test Name: {test_results.get('test_name', 'N/A')}", styles['Normal']))
    elements.append(Paragraph(f"Description: {test_results.get('description', 'N/A')}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Results section
    elements.append(Paragraph("Test Results:", styles['Heading2']))
    if 'result' in test_results:
        result_str = str(test_results['result'])
        elements.append(Paragraph(result_str, styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Assumption checks
    if assumptions:
        elements.append(Paragraph("Assumption Checks:", styles['Heading2']))
        for name, details in assumptions.items():
            elements.append(Paragraph(f"{name}: {details.get('interpretation', 'N/A')}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Interpretation
    elements.append(Paragraph("Interpretation Guide:", styles['Heading2']))
    elements.append(Paragraph(test_results.get('interpretation_help', 'N/A'), styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Step 1 - Upload
if st.session_state['step'] == 1:
    st.subheader("Step 1: Upload Your Data")
    
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx", "json"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            
            st.session_state['df'] = df
            st.success("File uploaded successfully!")
            st.write("Preview of your data:")
            st.dataframe(df.head())
            
            if st.button("Continue to Step 2"):
                st.session_state['step'] = 2
                # Instead of rerunning, just update the state and let Streamlit handle the rest

        except Exception as e:
            st.error(f"Error loading file: {e}")

# Step 2 - Natural Language Prompt
elif st.session_state['step'] == 2:
    st.subheader("Step 2: Describe Your Analysis")
    
    df = st.session_state['df']
    st.write("Preview of your data:")
    st.dataframe(df.head())
    
    user_prompt = st.text_area("🔍 Describe the comparison you're interested in", 
                              height=150, 
                              placeholder="e.g. Compare math scores between male and female students")
    
    if st.button("Run Statistical Test"):
        if not user_prompt.strip():
            st.error("Please describe your comparison.")
        else:
            st.session_state['user_prompt'] = user_prompt
            st.session_state['step'] = 3
            # Instead of rerunning, just update the state and let Streamlit handle the rest
            
    if st.button("Back to Step 1"):
        st.session_state['step'] = 1
        # Instead of rerunning, just update the state and let Streamlit handle the rest

# Step 3 - LLM Analysis & Assumption Checks
elif st.session_state['step'] == 3:
    st.subheader("Step 3: Analyzing Your Request")
    
    df = st.session_state['df']
    user_prompt = st.session_state['user_prompt']
    
    # Include OpenAI API key from environment variable or Streamlit secrets
    openai.api_key = st.secrets.get("OPENAI_API_KEY", "your_openai_api_key")
    
    columns_list = ', '.join(df.columns)
    
    analysis_prompt = f"""
    You are a statistical assistant. A user has uploaded a dataset with these columns: {columns_list}.
    The first few rows of the data are: {df.head().to_json()}
    
    The user has described their comparison as: "{user_prompt}"
    
    Your task is to:
    1. Determine the most appropriate statistical test.
    2. Output the Python code (using pandas, scipy, and statsmodels) to perform the test.
    3. Explain what the test does and how to interpret the result.
    4. List the columns to use for the test.
    5. Identify the type of test (t-test, chi-square, ANOVA, correlation, etc.)
    
    Return the answer in a JSON format like:
    {{
      "test_name": "...",
      "description": "...",
      "code": "...",
      "interpretation_help": "...",
      "columns": [...],
      "test_type": "..."
    }}
    """
    
    with st.spinner("Analyzing your request and selecting the right test..."):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "You are a statistical analyst."},
                          {"role": "user", "content": analysis_prompt}]
            )
            
            output = json.loads(response.choices[0].message['content'])
            
            # Check assumptions
            assumptions = check_assumptions(df, output.get('columns', []), output.get('test_type', ''))
            
            # Create visualization
            visualization = create_visualization(df, output.get('columns', []), output.get('test_type', ''))
            
            # Store results in session state
            st.session_state['test_results'] = output
            st.session_state['assumptions'] = assumptions
            st.session_state['visualization'] = visualization
            
            st.session_state['step'] = 4
            # Instead of rerunning, just update the state and let Streamlit handle the rest
        except Exception as e:
            st.error(f"Error analyzing request: {e}")
            
    if st.button("Back to Step 2"):
        st.session_state['step'] = 2
        # Instead of rerunning, just update the state and let Streamlit handle the rest

# Step 4 - Results Display
elif st.session_state['step'] == 4:
    st.subheader("Step 4: Statistical Test Results")
    
    df = st.session_state['df']
    output = st.session_state['test_results']
    assumptions = st.session_state['assumptions']
    visualization = st.session_state['visualization']
    
    # Display selected test
    st.subheader(f"🔍 Selected Test: {output['test_name']}")
    st.write(output['description'])
    
    # Display assumption checks
    if assumptions:
        st.subheader("📊 Assumption Checks")
        for name, details in assumptions.items():
            if 'p-value' in details:
                color = "green" if details.get('p-value', 0) > 0.05 else "red"
                st.markdown(f"**{name}**: {details['test']} - p-value: <span style='color:{color}'>{details['p-value']:.4f}</span> ({details['interpretation']})", unsafe_allow_html=True)
            else:
                st.write(f"**{name}**: {details.get('interpretation', 'N/A')}")
                
    # Run the statistical test
    st.subheader("🧪 Test Results")
    
    try:
        # Execute the code safely
        local_vars = {"df": df, "stats": stats, "pd": pd, "np": np, "sm": sm, "ols": ols}
        exec(output["code"], {}, local_vars)
        
        if "result" in local_vars:
            output["result"] = local_vars["result"]
            st.write(local_vars["result"])
            
    except Exception as e:
        st.error(f"Error running the test: {e}")
        
    # Display interpretation guide
    st.subheader("📘 Interpretation Guide")
    st.write(output['interpretation_help'])
    
    # Display visualization
    st.subheader("📈 Visualization")
    if visualization:
        st.plotly_chart(visualization, use_container_width=True)
        
    # Generate report download button
    if st.button("Generate Report"):
        report_buffer = generate_report(output, assumptions, df, st.session_state['user_prompt'])
        report_base64 = base64.b64encode(report_buffer.read()).decode()
        href = f'<a href="data:application/pdf;base64,{report_base64}" download="statistical_report.pdf">Download PDF Report</a>'
        st.markdown(href, unsafe_allow_html=True)
        
    if st.button("Back to Step 3"):
        st.session_state['step'] = 3
        # Instead of rerunning, just update the state and let Streamlit handle the rest
        
    if st.button("Start New Analysis"):
        st.session_state['step'] = 1
        st.session_state['df'] = None
        st.session_state['test_results'] = None
        st.session_state['assumptions'] = None
        st.session_state['visualization'] = None
        # Instead of rerunning, just update the state and let Streamlit handle the rest