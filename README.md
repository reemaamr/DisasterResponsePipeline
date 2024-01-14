# DisasterResponsePipeline

## Project Summary

The project aims to develop a robust API model designed for classifying disaster messages. Through the accompanying web application, emergency responders can input new messages and promptly receive classification results across various categories. This functionality provides insights into the nature of assistance required, such as "water," "food," "medical help," and more.

Additionally, the web app includes features to visualize relevant data, enhancing the user's ability to interpret and respond effectively to emergency situations.

## Web App Screenshots
![webappscreenshot1](https://github.com/reemaamr/DisasterResponsePipeline/assets/103683491/e744d3c3-d288-46f8-bf14-df9217662542)
![webappscreenshot3](https://github.com/reemaamr/DisasterResponsePipeline/assets/103683491/32b69fd6-d345-4454-8dd9-3b77afb2bdc3)
![webaapscreenshot2](https://github.com/reemaamr/DisasterResponsePipeline/assets/103683491/2b3ea8ed-3620-4c5a-b417-e8852f5fb8d6)

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


### Web Application
- **template:** Contains HTML templates for the web app.
  - **master.html:** The main page of the web app.
  - **go.html:** The classification result page of the web app.

- **run.py:** Flask file that runs the web app.
 
### Data Processing

- **process_data.py:** This script extracts and processes data from two CSV files, `messages.csv` (containing message data) and `categories.csv` (containing classes of messages). It merges and cleans the data, creating an SQLite database with consolidated information.

### Model Training

- **train_classifier.py:** This script takes the SQLite database produced by `process_data.py` as input, training and fine-tuning a machine learning model for categorizing messages. The output is a pickle file containing the trained model. Evaluation metrics are printed during the training process.

## Getting Started

### Prerequisites

Before running the project, ensure you have the required dependencies installed on your machine. You can do this by running the following command:

```bash
pip install -r requirements.txt
```

### Application Setup
To set up and run the project, follow these steps in your terminal from the top-level project directory (where this README is located):

1. Process the data by executing the following command:

    ```bash
    python data/process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
    ```

   This command will preprocess and merge the specified CSV files into a SQLite database named `DisasterResponse.db`.

2. Train the machine learning model by running the following command:

    ```bash
    python models/train_classifier.py data/DisasterResponse.db classifier.pkl
    ```

   This command utilizes the data from the processed database to train a model and saves it as `classifier.pkl`.

3. Launch the web application with the command:

    ```bash
    python run.py
    ```

   This will start the Flask web server.

4. Open your web browser and navigate to [http://0.0.0.0:3001/](http://0.0.0.0:3001/) (or try [http://localhost:3001/](http://localhost:3001/) if you encounter any issues).

In the web app, you can input any English text message, and it will be categorized among 35 classes.

## Project Origin

This application is a culmination of work completed during the Udacity Data Scientist Nanodegree. Code templates and initial datasets were generously provided by Udacity. The primary dataset, sourced from Figure Eight, was originally compiled by Udacity for use in this project.
