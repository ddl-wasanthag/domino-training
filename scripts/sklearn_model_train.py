# Comment out the paths as needed
# GBP - /mnt/artifacts
# DFS - /mnt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import json
import os
import mlflow
import pickle
from datetime import datetime

# Read in data
# For Domino File system projects where dataset path is /domino/datasets/local/
path = str('/domino/datasets/local/{}/WineQualityData.csv'.format(os.environ.get('DOMINO_PROJECT_NAME')))
# For Git-based projects where dataset path is /mnt/data/
#path = str('/mnt/data/{}/WineQualityData.csv'.format(os.environ.get('DOMINO_PROJECT_NAME')))

df = pd.read_csv(path)
print('Read in {} rows of data'.format(df.shape[0]))

# Rename columns to remove spaces
df.columns = df.columns.str.replace(' ', '_')

# Create is_red variable to store red/white variety as int    
df['is_red'] = (df['type'] == 'red').astype(int)

# Find all Pearson correlations of numerical variables with quality
corr_values = df.corr(numeric_only=True)['quality'].drop('quality')

# Keep all variables with above a 0.08 Pearson correlation
important_feats = corr_values[abs(corr_values) > 0.08]

# Drop NA rows
df = df.dropna(how='any', axis=0)

# Split df into inputs and target
X = df[important_feats.index]
y = df['quality'].astype(float)

# Create a new MLFlow experiment
experiment_name = os.environ.get('DOMINO_PROJECT_NAME') + " " + os.environ.get('DOMINO_STARTING_USERNAME')
mlflow.set_experiment(experiment_name)

# Generate a unique run name based on the current timestamp
run_name = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

with mlflow.start_run(run_name=run_name):
    mlflow.set_tag("Model_Type", "sklearn")
    
    # Create 70/30 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initiate and fit Gradient Boosted Regressor
    print('Training model...')
    gbr = GradientBoostingRegressor(loss='squared_error', learning_rate=0.15, n_estimators=75)
    gbr.fit(X_train, y_train)

    # Predict test set
    print('Evaluating model on test data...')
    preds = gbr.predict(X_test)

    # View performance metrics and save them to MLFlow
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    print("R2 Score:", round(r2, 3))
    print("MSE:", round(mse, 3))

    #Code to write R2 value and MSE to dominostats value for population in Domino Jobs View
    with open('dominostats.json', 'w') as f:
        f.write(json.dumps({"R2": r2,
                           "MSE": mse}))
    
    # Log metrics to MLFlow
    mlflow.log_metric("R2", round(r2, 3))
    mlflow.log_metric("MSE", round(mse, 3))

    # Write results to dataframe for visualizations
    results = pd.DataFrame({'Actuals': y_test, 'Predictions': preds})

    print('Creating visualizations...')
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    sns.regplot(data=results, x='Actuals', y='Predictions', order=3)
    plt.title('Sklearn Actuals vs Predictions Scatter Plot')
    #plt.savefig('/mnt/artifacts/visualizations/actual_v_pred_scatter.png')
    plt.savefig('/mnt/visualizations/actual_v_pred_scatter.png')
    #mlflow.log_artifact('/mnt/artifacts/visualizations/actual_v_pred_scatter.png')
    mlflow.log_artifact('/mnt/visualizations/actual_v_pred_scatter.png')

    # Create histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(results, bins=6, multiple='dodge', palette='coolwarm')
    plt.title('Sklearn Actuals vs Predictions Histogram')
    plt.xlabel('Quality')
    #plt.savefig('/mnt/artifacts/visualizations/actual_v_pred_hist.png')
    plt.savefig('/mnt/visualizations/actual_v_pred_hist.png')
    #mlflow.log_artifact('/mnt/artifacts/visualizations/actual_v_pred_hist.png')
    mlflow.log_artifact('/mnt/visualizations/actual_v_pred_hist.png')
    
    # Save trained model using pickle
    #model_path = '/mnt/artifacts/models/sklearn_gbr.pkl'
    model_path = '/mnt/models/sklearn_gbr.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(gbr, f)
    mlflow.log_artifact(model_path)

print('Script complete!')
