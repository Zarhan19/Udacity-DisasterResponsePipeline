# Disaster Response Pipeline Project

### Project Components:

#### 1. process_data.py: Data cleaning pipeline
        This script does the following :
            Loads the messages and categories datasets
            Merges the two datasets
            Cleans the data
            Stores it in a SQLite database

#### 2.train_classifier.py: Machine learning pipeline
        This script does the following :
            Loads data from the SQLite database
            Splits the dataset into training and test sets
            Builds a text processing and machine learning pipeline
            Trains and tunes a model using GridSearchCV
            Outputs results on the test set
            Exports the final model as a pickle file
        
#### 3. run.py: Flask app 
        This script is contains the app and the user interface used to predict results and display them.
    
    
#### 4. templates: folder containing the html templates


### Instructions:

#### Run the following commands in the project's root directory to set up your database and model.

         To run ETL pipeline that cleans data and stores in database
            python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

         To run ML pipeline that trains classifier and saves
            python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

         Run the following command in the app's directory to run your web app. python run.py

         Go to http://0.0.0.0:3001/
