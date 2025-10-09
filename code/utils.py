import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def clean_data(X, y):
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna(how='any')

    target_col = y.columns[0]
    
    X_clean = combined.drop(columns=[target_col])
    y_clean = combined[target_col]
    return X_clean, y_clean


def scale_data(X_train, X_test, y_train, y_test):
   
    scaler = StandardScaler()

    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test


def split_data(X, y, test_size=0.2, random_state=0):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)




    