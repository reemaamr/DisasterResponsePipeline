import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','omw-1.4'])
import pickle
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


def load_data(database_filepath):
    '''
    load_data
     this function load the dataset that has been processed using process_data.py
     
    Input:
    database_filepath  the file path of the processed data
    
    Return:
    X              a dataframe contains messages
    Y              a datafframe contains the labels
    category_list  a list of the 36 categories (labels)
    '''
    #import processed data
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_query("SELECT * FROM DisasterResponse", engine)
    
    #split columns
    X = df.message.values
    Y = df.drop(['id','message','original','genre'],axis=1).values
    category_list = list(df.columns[4:])
    
    #return text, labels, category list
    return X, Y, category_list


def tokenize(text):
    '''
    tokenize
     this function process the text by removing URLs, tokenize text into words, lemmatizes 
     each word, converts it to lowercase, and strips leading/trailing whitespaces.
     
    Input:
    text  the load text using the CountVectorizer from the built pipeline model
    
    Return:
    clean_tokens  list of cleaned and lemmatized tokens
    '''
    
    #find URLs in each text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    #Remove URLs from texts
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    '''
    build_model
     create a Pipeline with the folowing component:
     - CountVectorizer
     - TfidfTransformer
     - RandomForestClassifier with MultiOutputClassifier
     
     then use GridSearchCV to find the best parameters
     
    Return:
    cv   the built model
    '''
    
    # bulid model pipeline
    pipeline = pipeline=Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    #assign model's parameters
    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    #apply GridSearch to find the best model's hyperparameters
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    #return best model
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate_model
     after training the model, this function evaluate the model's performance using test set and print 
     the classification report of the model for each category
     
    Input:
    model   model that has been trained
    X_test          the text of the test set
    Y_test          test labels
    category_names  the list of the categories
    '''
    
    #model prediction
    y_pred = model.predict(X_test)
    
    #model's perofrmance evaluation
    report = classification_report(Y_test, y_pred, target_names=category_names)
    print(report)


def save_model(model, model_filepath):
    '''
    save_model
     save model as pkl file for deployment
    '''
    
    #save model for deployment
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()