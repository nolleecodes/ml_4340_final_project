import pandas as pd 
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

#pull in training dataset and then seperate the predictors and true class labels
train_df = pd.read_csv("data/p1_swansf.csv")
X_train = train_df.drop(columns=["label","start", "end"])
y_train = train_df["label"]

#create dt model and set a random depth
dt_model = DecisionTreeClassifier(max_depth=3,random_state=42)
#train model
dt_model.fit(X_train, y_train)
print("dt trained")
#plot trained model decision tree
plt.figure(figsize=(12,6))
plot_tree(dt_model, feature_names=X_train.columns, class_names=["No Flare", "Flare"],filled=True, fontsize=9)
plt.title("Decision Tree for Solar Flare Prediction")
plt.show()