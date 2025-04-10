# Data Analyst Agent

An AI-powered data analysis tool that helps you quickly visualize and gain insights from your data files.

## Features

- **Quick Visualization**: Upload your data file and get instant visualizations to understand patterns and trends.
- **Multi-format Support**: Works with CSV, Excel, JSON, and TXT files.
- **Interactive Visualizations**: Explore your data with interactive Plotly charts.
- **AI-Generated Insights**: Get automatic analysis and meaningful insights about your data.
- **Custom Queries**: Ask specific questions about your data and get intelligent answers.
- **Multiple LLM Support**: Choose between OpenAI, Anthropic, or Google AI models.

## Setup Instructions

1. Clone this repository:
```bash
git clone [repository-url]
cd [repository-directory]
```

2. Use the setup script for your platform:

**On Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**On Windows:**
```
setup.bat
```

This will:
- Create a virtual environment
- Install required dependencies
- Set up a template .env file for your API keys

3. Edit the created `.env` file to add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GOOGLE_API_KEY=your_google_api_key
```

4. Run the application:

**On Linux/macOS:**
```bash
source venv/bin/activate
streamlit run data_visualization/data_visualization_agent.py
```

**On Windows:**
```
venv\Scripts\activate
streamlit run data_visualization/data_visualization_agent.py
```

## Usage

1. Open the application in your web browser (usually at http://localhost:8501)
2. Upload your data file using the file uploader or choose one of the provided sample datasets
3. The application will automatically:
   - Show basic information about your data
   - Create visualizations based on your data types
   - Generate AI insights about key patterns and trends
4. Use the interactive elements to explore different aspects of your data
5. Ask custom questions about your data in the query section

## Sample Data

The application comes with sample datasets to help you get started:

- **Health Smartwatch Data (clean)**: A cleaned dataset of health metrics collected from smartwatches
- **Health Smartwatch Data (unclean)**: The raw, uncleaned version of the smartwatch dataset with potential data quality issues

These sample datasets are great for exploring the application's features without having to upload your own data first.

## Requirements

- Python 3.8+
- All packages listed in requirements.txt

## Example Workflow

1. Upload your CSV/Excel/JSON file or select a sample dataset
2. Review the data summary and initial visualizations
3. Explore correlations between variables
4. Check time series patterns if temporal data is available
5. Examine categorical distributions
6. Read AI-generated insights about your data
7. Ask specific questions about your dataset

## Project Structure

- `data_visualization/data_visualization_agent.py`: Main application file
- `sample_data/`: Directory containing sample datasets
- `requirements.txt`: List of required Python packages
- `.env`: Environment variables for API keys (not tracked in git)
- `setup.sh`: Setup script for Linux/macOS
- `setup.bat`: Setup script for Windows

## License

This project is licensed under the MIT License - see the LICENSE file for details. 