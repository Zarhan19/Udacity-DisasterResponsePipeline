import sys
import nltk
import numpy as np
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sqlalchemy import create_engine
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

def load_data(database_filepath):
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('Disasters', con=engine)
    
    X = df['message'].values
    #Y = df[df.columns[5:]]
    Y = df.iloc[:,5:]
    #print(Y.keys())
    #print(df.head())
    return X, Y


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(X_train,Y_train):
    print("Building pipeline")
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])    
    parameters = {'clf__estimator__min_samples_split': [2, 7]}
    
    model_pipeline = GridSearchCV(estimator=pipeline, param_grid=parameters)
    model_pipeline.fit(X_train,Y_train)
    
    return model_pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    print("Model Evaluation")
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=Y_test.keys()))
    
def train_model(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
    print('Building model...')
    model = build_model(X_train,Y_train)
         
    print('Fitting model')
    model.fit(X_train, Y_train)       
    print('Evaluating model')
          
    evaluate_model(model,X_test, Y_test,Y.keys())
    return model 

def export_model(model,model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
   

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        model = train_model(X, Y)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        export_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()