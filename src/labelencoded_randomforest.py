import pandas as pd
from sklearn import ensemble, metrics, preprocessing

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

    # Now we label encode the features
    for col in features:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(df[col])
        df.loc[:, col] = lbl.transform(df[col])

    # Get training data using fold
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # Get validation data using fold
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # Get features records for train and validation data
    x_train = df_train[features]
    x_valid = df_valid[features]

    # Initialize random forest model
    model = ensemble.RandomForestClassifier(n_jobs=-1)

    # Fit model on training data
    model.fit(x_train, df_train.target.values)

    # Predict on validation data, we need to use the
    # probability values because we are using AUC
    # we will use the probabilities of 1s

    validation_preds = model.predict_proba(x_valid)[:, 1]

    # Get the ROC AUC score
    auc = metrics.roc_auc_score(df_valid.target.values, validation_preds)

    # Print fold and AUC
    print(f"fold: {fold} | AUC: {auc}")


if __name__ == "__main__":
    for fold in range(5):
        run(fold)
