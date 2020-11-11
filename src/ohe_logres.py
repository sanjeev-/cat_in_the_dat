"""
One of the simplest models we can build is a logistic regression where we one-hot encode all
of the features
"""
import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

import config


def run(fold):
    # Load in full training set
    df = pd.read_csv(config.KFOLD_TRAINING_DATA_FILEPATH)

    COLS_TO_EXCLUDE = ("id", "target", "kfold")

    # All columns are features except for the id, target, and kfold features
    features = [col for col in df.columns if col not in COLS_TO_EXCLUDE]

    # fill all NaN values with NONE
    # we convert all columns to strings, because
    # they are all categorical so type doesnt matter

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Get training data using fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # Get validation data using fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Initialize OneHotEncoder from sklearn
    ohe = preprocessing.OneHotEncoder()

    # Fit ohe on training + validation features
    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    # Transform training data
    x_train = ohe.transform(df_train[features])

    # Transform validation data
    x_validation = ohe.transform(df_valid[features])

    # Initialize logistic regression model
    model = linear_model.LogisticRegression()

    # Fit model on OHE training data
    model.fit(x_train, df_train.target.values)

    # Predict on validation data, we need to use the
    # probability values because we are using AUC
    # we will use the probabilities of 1s

    validation_preds = model.predict_proba(x_validation)[:, 1]

    # Get the ROC AUC score
    auc = metrics.roc_auc_score(df_valid.target.values, validation_preds)

    # Print fold and AUC
    print(f"fold: {fold} | AUC: {auc}")


if __name__ == "__main__":
    run(0)

