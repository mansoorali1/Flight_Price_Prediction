import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle

def preprocess_data(df):
    """
    Objectives:
    1. Dropping Null values
    2. Dropping Duplicates
    3. Dropping unnecessary columns
    """
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df.drop(columns=['Route', 'Additional_Info', 'Arrival_Time', 'Duration'], inplace=True)
    return df

class date_splitter(BaseEstimator, TransformerMixin):
    """
    Objective:
    Created a custom transformer date_splitter to Extract Day, Month, Day of the week, Day of the year
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X['Date_of_Journey'] = pd.to_datetime(X['Date_of_Journey'], format='%d/%m/%Y')
        X['Date'] = X['Date_of_Journey'].dt.day
        X['Month'] = X['Date_of_Journey'].dt.month
        X['Day_of_week'] = X['Date_of_Journey'].dt.weekday
        X['Day_of_year'] = X['Date_of_Journey'].dt.dayofyear
        X.drop(['Date_of_Journey'], axis=1, inplace=True)
        return X[['Date', 'Month', 'Day_of_week', 'Day_of_year']].values

class tod_departure(BaseEstimator, TransformerMixin):
    """
    Objectives:
    1. Created a custom transformer tod_departure
    2. Extract Departure hour
    3. Extract Departure minutes
    4. Depending on departure hour and minute, generate new feature TOD = 'time of day'
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Extract hour and minute
        X['Dep_hour'] = X['Dep_Time'].apply(lambda x: int(x.split(':')[0]))
        X['Dep_min'] = X['Dep_Time'].apply(lambda x: int(x.split(':')[1]))

        # Map hour to time of day
        tod = lambda x: 'early morning' if 0 < x <= 6 else ('morning' if 6 < x <= 12 else ('noon' if 12 < x <= 16 else ('evening' if 16 < x <= 20 else 'night')))
        X['TOD'] = X['Dep_hour'].apply(tod)
        X.drop(['Dep_Time'], axis=1, inplace=True)

        return X[['Dep_hour', 'Dep_min', 'TOD', 'Airline']].values

class stops(BaseEstimator, TransformerMixin):
    """
    Objective:
    Created a Custom transformer stops to convert Total_Stops to numerical values.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        stops_map = {
            'non-stop': 0,
            '1 stop': 1,
            '2 stops': 2,
            '3 stops': 3,
            '4 stops': 4
        }
        X['Total_Stops'] = X['Total_Stops'].map(stops_map)
        return X[['Total_Stops']].values

def build_feature_union():
    """
    Objective:
    Build a FeatureUnion to combine different transformer objects.
    """
    features = FeatureUnion(transformer_list=[
        ('date_splitter', date_splitter()),
        ('time_of_departure', tod_departure()),
        ('stops', stops())
    ])
    return features

def build_pipeline():
    """
    Objective:
    1. Building a pipeline to preprocess data
    2. Encode categorical variables using OneHotEncoder
    3. Combining Feature Union and Encoder into the pipeline
    """
    oh_encoder = OneHotEncoder(drop='first')
    encoder = ColumnTransformer([('Airline_TOD', oh_encoder, [6, 7])], remainder='passthrough')

    features = build_feature_union()

    pipe = Pipeline(steps=[
        ('feature_union', features),
        ('encoder', encoder)
    ])
    return pipe

def save_data(df, trainset, testset):
    """
    Objective:
    Saving datasets to CSV files
    """
    df.to_csv(r'../data/dataset.csv', index=False)
    trainset.to_csv(r'../data/trainset.csv', index=False)
    testset.to_csv(r'../data/testset.csv', index=False)

def save_pipeline(pipeline, features, encoder):
    """
    Objective:
    Saving the pipeline, features, and encoder to pickle files
    """
    with open(r'../bin/features.pkl', 'wb') as f1:
        pickle.dump(features, f1)

    with open(r'../bin/encoder.pkl', 'wb') as f2:
        pickle.dump(encoder, f2)

    with open(r'../bin/pipe.pkl', 'wb') as f3:
        pickle.dump(pipeline, f3)


if __name__ == "__main__":
    # Load and preprocess data
    df = pd.read_excel("Data_Train.xlsx")
    df = preprocess_data(df)

    # Split data
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

    # Build and fit the pipeline
    pipeline = build_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Create transformed datasets
    trainset = pd.DataFrame(X_train_transformed)
    testset = pd.DataFrame(X_test_transformed)

    # Save datasets and pipeline
    save_data(df, trainset, testset)
    save_pipeline(pipeline, build_feature_union(), pipeline.named_steps['encoder'])