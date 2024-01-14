# DisasterResponsePipeline

## Table of Contents

- [Project Summary](#project-summary)
- [Web App Screenshots](#web_app_screenshots)
- [Project Implementation](#project_implementation)
  - [Project Structure](#project_structure)
  - [Data Processing](#data_preprocessing)
  - [Model Training](#model_training)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installing Dependencies](#installing-dependencies)
  - [Running Python Scripts](#running-python-scripts)
  - [Running the Web App](#running-the-web-app)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Summary

The project aims to develop a robust API model designed for classifying disaster messages. Through the accompanying web application, emergency responders can input new messages and promptly receive classification results across various categories. This functionality provides insights into the nature of assistance required, such as "water," "food," "medical help," and more.

Additionally, the web app includes features to visualize relevant data, enhancing the user's ability to interpret and respond effectively to emergency situations.

## Web App Screenshots

## Project Implementation

### Project Structure
app

| - template

| - master.html # main page of web app

| - go.html # classification result page of web app

|- run.py # Flask file that runs the app

<br>
data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py

|- InsertDatabaseName.db # database to save clean data to

<br>
models

|- train_classifier.py

|- classifier.pkl # saved model

<br>
README.md



### Data Processing

- **process_data.py:** This script extracts and processes data from two CSV files, `messages.csv` (containing message data) and `categories.csv` (containing classes of messages). It merges and cleans the data, creating an SQLite database with consolidated information.

### Model Training

- **train_classifier.py:** This script takes the SQLite database produced by `process_data.py` as input, training and fine-tuning a machine learning model for categorizing messages. The output is a pickle file containing the trained model. Evaluation metrics are printed during the training process.

## Getting Started

### Prerequisites

Before running the project, ensure you have the required dependencies installed on your machine. You can do this by running the following command:

```bash
pip install -r requirements.txt
