import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
import json
import os
import mlflow

#Read in data
path = str('/domino/datasets/local/{}/WineQualityData.csv'.format(os.environ.get('DOMINO_PROJECT_NAME')))
df = pd.read_csv(path)
print('Read in {} rows of data'.format(df.shape[0]))

#rename columns to remove spaces
for col in df.columns:
    df.rename({col: col.replace(' ', '_')}, axis =1, inplace = True)

#Create is_red variable to store red/white variety as int    
df['is_red'] = df.type.apply(lambda x : int(x=='red'))

#Find all pearson correlations of numerical variables with quality
corr_values = df.corr(numeric_only=True).sort_values(by = 'quality')['quality'].drop('quality',axis=0)

#Keep all variables with above a 8% pearson correlation
important_feats=corr_values[abs(corr_values)>0.08]

#Get data set up for model training and evaluation

#Drop NA rows
df = df.dropna(how='any',axis=0)
#Split df into inputs and target
X = df[important_feats.keys()]
y = df['quality'].astype('float64')

# create a new MLFlow experiemnt
mlflow.set_experiment(experiment_name=os.environ.get('DOMINO_PROJECT_NAME') + " " + os.environ.get('DOMINO_STARTING_USERNAME'))

with mlflow.start_run():
    # Set MLFlow tag to differenciate the model approaches
    mlflow.set_tag("Model_Type", "sklearn")
    
    #Create 70/30 train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #initiate and fit Gradient Boosted Classifier
    print('Training model...')
    gbr = GradientBoostingRegressor(loss='ls',learning_rate = 0.15, n_estimators=75, criterion = 'mse')
    gbr.fit(X_train,y_train)

    #Predict test set
    print('Evaluating model on test data...')
    preds = gbr.predict(X_test)

    #View performance metrics and save them to domino stats!
    print("R2 Score: ", round(r2_score(y_test, preds),3))
    print("MSE: ", round(mean_squared_error(y_test, preds),3))
    
    # Save the metrics in MLFlow
    mlflow.log_metric("R2", round(r2_score(y_test, preds),3))
    mlflow.log_metric("MSE", round(mean_squared_error(y_test,preds),3))

    #Code to write R2 value and MSE to dominostats value for population in experiment manager
    with open('dominostats.json', 'w') as f:
        f.write(json.dumps({"R2": round(r2_score(y_test, preds),3),
                           "MSE": round(mean_squared_error(y_test,preds),3)}))

    #Write results to dataframe for visualizations
    results = pd.DataFrame({'Actuals':y_test, 'Predictions':preds})

    print('Creating visualizations...')
    #Add visualizations and save for inspection
    fig1, ax1 = plt.subplots(figsize=(10,6))
    plt.title('Sklearn Actuals vs Predictions Scatter Plot')
    sns.regplot( 
        data=results,
        x = 'Actuals',
        y = 'Predictions',
        order = 3)
    plt.savefig('/mnt/visualizations/actual_v_pred_scatter.png')
    mlflow.log_figure(fig1, 'actual_v_pred_scatter.png')

    fig2, ax2 = plt.subplots(figsize=(10,6))
    plt.title('Sklearn Actuals vs Predictions Histogram')
    plt.xlabel('Quality')
    sns.histplot(results, bins=6, multiple = 'dodge', palette = 'coolwarm')
    plt.savefig('/mnt/visualizations/actual_v_pred_hist.png')
    mlflow.log_figure(fig2, 'actual_v_pred_hist.png')
    
mlflow.end_run()

#Saving trained model to serialized pickle object 

import pickle 

# save best model
file = '/mnt/models/sklearn_gbm.pkl'
pickle.dump(gbr, open(file, 'wb'))

print('Script complete!')