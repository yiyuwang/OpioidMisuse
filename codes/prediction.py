#!/usr/bin/env python

'''
functions for running ml predictions

Yiyu Wang 2025/03/27
'''

PREDICTOR_COLUMNS = ['Age', 'Hispanic', 'Income', 'Education', 'Disability',
'Pain_duration',  'Average_pain_intensity', 'Worst_pain_intensity','Current_pain_intensity',
'Opioid_years', 'PainInterference', 'Anger', 'Anxiety', 'Depression',
'Fatigue', 'GlobalPhysical', 'GlobalMental', 'PhysicalFunction',
'SleepDisturbance', 'PCS_total', 'PCS_helplessness',
'PCS_rumination', 'PCS_magnification',  'DAST_total',
'AEQ', 'CTQ_EmotionalAbuse','AUDIT_total', 
'CTQ_PhysicalAbuse', 'CTQ_EmotionalNeglect', 'CTQ_PhysicalNeglect',
'CTQ_SexualAbuse', 'CTQ_total', 'White', 'Female', 'Married',
'Employed']

SEED = 42

import glob
from sklearn.metrics import roc_auc_score, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.metrics import r2_score





def split_data(df, split_data = 'split', outcome = 'ouddx', predictor_columns = PREDICTOR_COLUMNS, test_size = 0.2, SEED = SEED):
    if split_data == 'cohort':

        y_train, y_test, y_val = df.loc[df['cohort'] == 'train', outcome], df.loc[df['cohort'] == 'test', outcome], df.loc[df['cohort'] == 'val', outcome]
        X_train, X_test, X_val = df.loc[df['cohort'] == 'train', predictor_columns], df.loc[df['cohort'] == 'test', predictor_columns], df.loc[df['cohort'] == 'validation', predictor_columns]

        print(X_train.shape, X_test.shape, X_val.shape)
    elif split_data == 'split':
        X = df[predictor_columns]
        y = df[outcome]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=SEED)
        
    else:
        X_train, X_test, y_train, y_test = [], [], [], []
        print('split_data must be either "split" or "cohort"')
    return X_train, X_test, y_train, y_test
    # return {'X_train': X_train,'X_test':X_test, 'y_train': y_train, 'y_test':y_test}



def monte_carlo_cv(model, df, split, outcome, n_splits=100, test_size=0.3, seed_start=SEED, predictor_columns=PREDICTOR_COLUMNS):
    auc_scores, acc_scores, rmse_scores, r_scores = [], [], [], []
    seeds = np.arange(seed_start, seed_start + n_splits)  # Define seeds for repeatability
    models = []
    for seed in seeds:
        # Split the data
        X_train, X_test, y_train, y_test = split_data(df, split_data = split, outcome = outcome, predictor_columns=predictor_columns, test_size = test_size, SEED = seed)
        
         # set up model
        if model == None:
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=200,          # Number of trees in the forest
                max_depth=4,              # Maximum depth of the trees
                min_samples_split=4,       # Minimum number of samples required to split an internal node
                min_samples_leaf=2,        # Minimum number of samples required to be at a leaf node
                random_state=seed,            # Ensures a deterministic outcome for reproducibily
            )
        else:
            model = model
            model.random_state = seed 

        model.fit(X_train, y_train)
        models.append(model)
        
        y_pred = model.predict(X_test)
        # Evaluate the model
        if outcome == 'commtot':
            rmse_score = root_mean_squared_error(y_test, y_pred)
            rmse_scores.append(rmse_score)

            r_score = np.corrcoef(y_test, y_pred)[0, 1]
            r_scores.append(r_score)

        elif outcome == 'ouddx':
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc_score)

            accuracy = np.mean(y_pred == y_test)
            acc_scores.append(accuracy)

        else:
            raise ValueError(f'Invalid outcome: {outcome}')    

    # collect the results   
    if outcome == 'commtot':
        # Calculate mean mse and 95% CI
        mean_rmse = np.mean(rmse_scores)
        rmse_ci_lower, rmse_ci_upper = np.percentile(rmse_scores, [2.5, 97.5])
        mean_r = np.mean(r_scores)
        r_ci_lower, r_ci_upper = np.percentile(r_scores, [2.5, 97.5])
        # Calculate mean r and 95% CI
        r2_scores = [r**2 for r in r_scores]
        mean_r2 = np.nanmean(r2_scores)
        result = {
            'models': models,
            'mean_rmse': mean_rmse,
            'rmse_ci_lower': rmse_ci_lower,
            'rmse_ci_upper': rmse_ci_upper,
            'mean_r': mean_r,
            'median_r': np.nanmedian(r_scores), 
            'r_ci_lower': r_ci_lower,
            'r_ci_upper': r_ci_upper,
            'rmse_scores': rmse_scores,
            'r_scores': r_scores,
            'r2_scores': r2_scores,
            'mean_r2': mean_r2,
            'median_r2': np.nanmedian(r2_scores),
            'seeds': seeds,
        }
      
    elif outcome == 'ouddx':
        # Calculate mean AUC and 95% CI
        mean_auc = np.mean(auc_scores)
        auc_ci_lower, auc_ci_upper = np.percentile(auc_scores, [2.5, 97.5])

        mean_accuracy = np.mean(acc_scores)
        accuracy_ci_lower, accuracy_ci_upper = np.percentile(acc_scores, [2.5, 97.5])

        # mse
        rmse_scores = [root_mean_squared_error(y_test, model.predict(X_test)) for model in models]
        # Calculate mean mse and 95% CI
        mean_rmse = np.mean(rmse_scores)
        rmse_ci_lower, rmse_ci_upper = np.percentile(rmse_scores, [2.5, 97.5])
        
        result = {
            'models': models,
            'mean_auc': mean_auc,
            'auc_ci_lower': auc_ci_lower,
            'auc_ci_upper': auc_ci_upper,
            'mean_accuracy': mean_accuracy,
            'accuracy_ci_lower': accuracy_ci_lower,
            'accuracy_ci_upper': accuracy_ci_upper,
            'auc_scores': auc_scores,
            'acc_scores': acc_scores,
            'mse_scores': rmse_scores,
            'mean_rmse': mean_rmse,
            'rmse_ci_lower': rmse_ci_lower,
            'rmse_ci_upper': rmse_ci_upper,
            'seeds': seeds,
        }   
           
    return result


def mc_cv_ml_pipeline(model, df, split, outcome, test_size=0.3, n_split=100, seed_start=0, predictor_columns = PREDICTOR_COLUMNS):
    # Split the dataframe into X (features) and y (target)
    
    result = monte_carlo_cv(model, df, split, outcome, n_splits=n_split, test_size=test_size, seed_start=seed_start, predictor_columns=predictor_columns)
    if outcome == 'ouddx':
        print(f"Mean AUC: {result['mean_auc']:.3f}, 95% CI: [{result['auc_ci_lower']:.3f}, {result['auc_ci_upper']:.3f}]")
        print(f"Mean Accuracy: {result['mean_accuracy']:.3f}, 95% CI: [{result['accuracy_ci_lower']:.3f}, {result['accuracy_ci_upper']:.3f}]")
    elif outcome == 'commtot':
        print(f"Mean RMSE: {result['mean_rmse']:.3f}, 95% CI: [{result['rmse_ci_lower']:.3f}, {result['rmse_ci_upper']:.3f}]")
        print(f"Mean R: {result['mean_r']:.3f}, 95% CI: [{result['r_ci_lower']:.3f}, {result['r_ci_upper']:.3f}]")
        print(f"Median R: {np.median(result['r_scores']):.3f}")    
    return result


