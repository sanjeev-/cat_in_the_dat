import pandas as pd
from sklearn import model_selection
import config


def create_folds(num_folds):

    # Read in training data
    df = pd.read_csv(config.TRAINING_DATA_FILEPATH)

    # Create a new column called KFold and fill it with -1
    df.loc[:, "kfold"] = -1

    # Randomize the rows of data
    df.sample(frac=1).reset_index(drop=True)

    # Fetch labels
    y = df.target.values

    # Initiate kfold class from model selection
    kf = model_selection.StratifiedKFold(n_splits=num_folds)

    # Fill in new kfold column
    for fold, (_, validation_indices) in enumerate(kf.split(X=df, y=y)):
        df.loc[validation_indices, "kfold"] = fold

    # Save the new csv with kfold column
    df.to_csv(config.KFOLD_TRAINING_DATA_FILEPATH, index=False)


if __name__ == "__main__":
    create_folds(num_folds=5)

