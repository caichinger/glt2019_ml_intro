"""Utilities and constants."""
import pandas as pd
from sklearn.metrics import confusion_matrix as confusion_matrix


DATA_PATH_RAW = 'data/heart.csv.gz'
DATA_PATH_PREPROCESSED = 'data/heart_preprocessed.csv.gz'

TARGET = 'AHD'
TARGET_VALUES = ['No', 'Yes']
FEATURES = [
    'Age',
    'Ca',
    'ChestPain',
    'Chol',
    'ExAng',
    'Fbs',
    'MaxHR',
    'Oldpeak',
    'RestBP',
    'RestECG',
    'Sex',
    'Slope',
    'Thal',
]


def confusion_matrix_as_df(y_true, y_pred, labels=None):
    labels = labels or sorted(set(y_true).union(y_pred))
    return pd.DataFrame(confusion_matrix(y_true, y_pred),
                        columns=pd.Series(labels, name='pred'),
                        index=pd.Series(labels, name='true'))


def cv_results_to_df(results):
    df = pd.DataFrame.from_records(results['params'])
    df['mean_test_score'] = results['mean_test_score']
    df['mean_train_score'] = results['mean_train_score']
    return df.sort_values('mean_test_score', ascending=False)
