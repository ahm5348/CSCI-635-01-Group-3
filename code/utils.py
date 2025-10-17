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


def normalize_data(X, y, random_state=0):
    import imblearn.over_sampling as over   # optional part so import separate
    # simple implementation that uses SMOTE
    smote = over.SMOTE(random_state=random_state)
    X_smote, y_smote = smote.fit_resample(X, y)

    return X_smote, y_smote


def shap_processing(model, X_test):
    import shap
    explainer = shap.Explainer(model)
    explained_values = explainer(X_test)
    shap.initjs()       # for JavaScript support in notebooks

    shap.summary_plot(explained_values, X_test)

    return


def scale_data(X_train, X_test, y_train, y_test):
   
    scaler = StandardScaler()

    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

    X_test = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)

    return X_train, X_test, y_train, y_test


def split_data(X, y, test_size=0.2, random_state=0):
    # stratification for handling imbalanced data
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)





    