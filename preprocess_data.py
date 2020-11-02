'''
GT User ID: sbiswas67	         Assignment 1 (Supervised Learning)

Dataset A: Breast Cancer Wisconsin (Diagnostic) Dataset
Dataset B:  Statlog (Vehicle Silhouettes) Dataset
'''

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler    #Feature Scaling
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.compose import ColumnTransformer
import matplotlib.mlab as mlab

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def breast_cancer_diagnostic ():

    ############# Dataset A ############################

    df_cancer = pd.read_csv('wdbc.data')

    labelencoder = LabelEncoder()
    df_cancer['diagnosis'] = labelencoder.fit_transform(df_cancer['diagnosis'])

    '''###### Plot Correlation
    sns.set(style='ticks', color_codes=True)
    plt.figure(figsize=(30, 18))
    sns.heatmap(df_cancer.corr(), linewidths=0.1, linecolor='white', annot=True, cbar=True)
    plt.savefig('breast_cancer_diagnostic.png')

    corr_matrix = df_cancer.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    print (to_drop)'''

    # Drop "id" as it is not a feature and "Diagnosis" as it is the label/class
    X = df_cancer.drop(columns=['id', 'diagnosis'])
    # Standardize data
    # X = scale(X)

    # Y includes our labels/classes
    Y = df_cancer.diagnosis  # malignant or benign

    ## pd.plotting.scatter_matrix(X, alpha = 0.3, figsize = (14,8), diagonal = 'kde')
    return X, Y

def vehicle_silhouettes (filename):

    ############# Dataset B ############################
    """
    Loads the Breast Cancer Wisconsin (Original) Data Set
    Input: filename
    Returns: X (features) and Y (classes/labels)
    """

    df_data = pd.read_csv(filename, na_values = ['?'])


    # Encode the Categorical/text value to numeric
    labelencoder = LabelEncoder()
    df_data['CLASS'] = labelencoder.fit_transform(df_data['CLASS'])

    '''
    
    ###### Plot Correlation
    sns.set(style='ticks', color_codes=True)
    plt.figure(figsize=(30, 18))
    sns.heatmap(df_data.corr(), linewidths=0.1, linecolor='white', annot=True, cbar=True)
    plt.savefig('vehicle_silhouettes_corr.png')
    
    corr_matrix = df_data.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.9
    to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
    '''


    # Drop "CLASS" as it is not a feature.
    X = df_data.drop(columns=['CLASS'])
    #X = X.drop(columns=to_drop)
    # Standardize data
    # X = scale(X)

    # Y includes our labels/classes
    Y = df_data.CLASS  # Encoded using LabelEncoding

    return X, Y


