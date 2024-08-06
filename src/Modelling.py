# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from Preprocessing import date_splitter, tod_departure, stops
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained pipeline
def load_pipeline(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# Load and preprocess the dataset
def load_and_preprocess_data(file_path, pipe):
    df = pd.read_csv(file_path)
    X, y = df.iloc[:, :-1], df.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    x_train1 = pipe.fit_transform(x_train)
    x_test1 = pipe.transform(x_test)
    return X,y,x_train1, x_test1, np.array(y_train), np.array(y_test)

# Define a function to evaluate model performance
def evaluate_clf(true, predicted):
    mse = mean_squared_error(true, predicted)
    mae = mean_absolute_error(true, predicted)
    r2 = r2_score(true, predicted)
    rmse = mean_squared_error(true, predicted, squared=False)
    mape = mean_absolute_percentage_error(true, predicted)
    return mse, mae, r2, rmse, mape

# Create a function to evaluate multiple models
def evaluate_models(x_train1, x_test1, y_train, y_test, models):
    models_list = []
    mse, mae, r2, rmse, mape = [], [], [], [], []

    for name, model in models.items():
        model.fit(x_train1, y_train)  # Train model

        # Make predictions
        y_train_pred = model.predict(x_train1)
        y_test_pred = model.predict(x_test1)

        # Evaluate performance
        model_train_metrics = evaluate_clf(y_train, y_train_pred)
        model_test_metrics = evaluate_clf(y_test, y_test_pred)

        # Append metrics to lists
        models_list.append(name)
        mse.append(float(model_test_metrics[0]))
        mae.append(float(model_test_metrics[1]))
        r2.append(float(model_test_metrics[2]))
        rmse.append(float(model_test_metrics[3]))
        mape.append(float(model_test_metrics[4]))



    report = pd.DataFrame(list(zip(models_list, mse, mae, r2, rmse, mape)), columns=['Model Name', 'MSE', 'MAE', 'R2', 'RMSE', 'MAPE']).sort_values(by=['R2'], ascending=False)
    return report


# Hyperparameter tuning for models
def perform_random_search(x_train1, y_train1, randomcv_models):
    model_param = {}
    for name, model, params in randomcv_models:
        random = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=100, cv=3, verbose=2, n_jobs=-1)
        random.fit(x_train1, y_train1)
        model_param[name] = random.best_params_
    return model_param

# Train final model
def train_final_model(X, y, best_params, pipe):
    X_final = pipe.fit_transform(X)
    y_final = np.array(y)
    final_model = XGBRegressor(min_child_weight=best_params['min_child_weight'], max_depth=best_params['max_depth'])
    final_model.fit(X_final, y_final)
    return final_model

# Save model to a file
def save_model(model, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

# Main execution
if __name__ == "__main__":
    # Load pipeline and data
    pipe = load_pipeline('..//bin/pipe.pkl')
    X, y, x_train1, x_test1, y_train1, y_test1 = load_and_preprocess_data(r'../data/dataset.csv', pipe)

    # Hyperparameter tuning
    xgboost_params = {'max_depth': range(3, 7, 2), 'min_child_weight': range(1, 6, 2)}
    rf_params = {"max_depth": [10, None, 15], "max_features": ['sqrt', 'log2', None], "n_estimators": [10, 100, 200]}
    catboost_params = {'depth': [6, 8, 10], 'learning_rate': [0.01, 0.05, 0.1], 'iterations': [30, 50, 100]}

    randomcv_models = [
        ('XGBoost', XGBRegressor(), xgboost_params),
        ("RF", RandomForestRegressor(), rf_params),
        ("CatBoost", CatBoostRegressor(), catboost_params)
    ]

    model_param = perform_random_search(x_train1, y_train1, randomcv_models)

    # Print the best parameters found
    for model_name in model_param:
        print(f"---------------- Best Params for {model_name} -------------------")
        print(model_param[model_name])

    # Evaluate best models
    best_models = {
        "Random Forest Regressor": RandomForestRegressor(**model_param['RF']),
        "CatBoostRegressor": CatBoostRegressor(**model_param['CatBoost']),
        "XGBRegressor": XGBRegressor(**model_param['XGBoost'], n_jobs=-1),
    }

    tuned_report = evaluate_models(x_train1=x_train1, x_test1=x_test1, y_train=y_train1, y_test=y_test1, models=best_models)

    print(tuned_report)

    # Train and save final model
    final_model = train_final_model(X, y, model_param['XGBoost'], pipe)
    save_model(final_model, r'../bin/model.pkl')