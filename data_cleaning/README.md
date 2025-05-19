# Auto Data Cleaner

A Python tool that leverages LLMs (Large Language Models) to automatically analyze and clean datasets with minimal human intervention.

## Overview

This tool uses OpenAI's GPT models to analyze data, identify potential cleaning issues, and generate actionable cleaning steps. It then executes those steps to produce a cleaned dataset ready for analysis.

## Features

- **Automated Data Analysis**: Uses LLMs to identify potential data issues
- **Intelligent Cleaning Operations**: Performs targeted cleaning based on AI recommendations
- **Multiple Cleaning Methods**:
  - Handle missing values (mean, median, mode, custom fill)
  - Fix data types
  - Handle outliers (z-score or IQR methods)
  - Standardize inconsistent values
  - Remove duplicate entries
  - Drop unnecessary columns
- **Interactive Mode**: Allows user verification and customization of cleaning steps
- **Detailed Reporting**: Provides before/after statistics and cleaning impact assessment