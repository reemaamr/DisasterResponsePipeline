import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np

def load_data(messages_filepath, categories_filepath):
    '''
    load_data
    load data from csv files and merge them into one dataframe
    
    Input:
    messages_filepath   file path to messages csv
    categories_filepath file path to categories csv
    
    Return:
    df  a dataframe with messages and categories merged togather
    '''
    #load messages
    messages = pd.read_csv(messages_filepath)
    
    #load categories
    categories = pd.read_csv(categories_filepath)
    
    #merge the imported data using 'id'
    df = pd.merge(categories, messages, on="id")
    
    return df


def clean_data(df):
    '''
    clean_data
    clean the merged dataframe 
    
    Input:
    df   a dataframe returned from load_data
    
    Return:
    df   a dataframe that has been processed and cleaned
    '''
    
    # create a dataframe of the 36 individual category columns
    categories_list = df['categories'].str.split(";",expand=True)
    
    # select the first row of the categories dataframe
    row = categories_list.iloc[0]
    
    #extracting the columns' name from first row
    category_colnames = [sub[: -2] for sub in row]
    
    #assign the category_colnames to categories_list
    categories_list.columns = category_colnames
    
    for column in categories_list:
        # set each value to be the last character of the string
        categories_list[column] = categories_list[column].str[-1:]
    
        # convert column from string to numeric
        categories_list[column] = pd.to_numeric(categories_list[column])
        
    # drop the original categories column from `df`
    df=df.drop('categories', axis='columns')
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories_list], axis=1)
    
    # drop duplicates
    df=df.drop_duplicates()
    
    df['related']=df['related'].replace([2], 1)
    
    return df


def save_data(df, database_filename):
    '''
    save_data
    save the cleaned dataframe into an SQL database
    
    Input:
    df                 a dataframe returned from load_data
    database_filename  a name of the database file
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()