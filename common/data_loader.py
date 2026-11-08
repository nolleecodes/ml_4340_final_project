import pandas as pd

#loading in trainnig,validation, and testing data.
def load_data():
    train = pd.read_csv("data/p1_swansf.csv")
    val = pd.read_csv("data/p2_swansf.csv")
    test = pd.read_csv("data/p3_swansf.csv")

    #seperate features from target and exclude start/end
    X_train = train.drop(columns=["label", "start", "end"])
    y_train = train["label"]

    X_val = val.drop(columns=["label", "start", "end"])
    y_val = val["label"]

    X_test = test.drop(columns=["label", "start", "end"])
    y_test = test["label"]

    return X_train, y_train, X_val, y_val, X_test, y_test
