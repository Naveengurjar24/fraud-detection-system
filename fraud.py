import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score


data = {
    "Amount": [100,200,150,5000,300,10000,250,7000,
               120,8000,400,15000,180,9000,350,20000,
               220,6000,450,12000],

    "Transactions": [1,2,1,10,2,15,1,12,
                     1,14,2,18,1,13,2,20,
                     2,11,3,16],

    "Previous_Fraud": [0,0,0,1,0,1,0,1,
                       0,1,0,1,0,1,0,1,
                       0,1,0,1],

    "Distance": [2,5,3,500,4,800,6,600,
                 3,700,5,900,2,650,4,1000,
                 5,550,3,750],

    "Fraud": [0,0,0,1,0,1,0,1,
              0,1,0,1,0,1,0,1,
              0,1,0,1]
}

df = pd.DataFrame(data)

X = df[["Amount", "Transactions", "Previous_Fraud", "Distance"]]
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


decision_tree = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

decision_tree.fit(X_train, y_train)
random_forest.fit(X_train, y_train)

dt_prediction = decision_tree.predict(X_test)
rf_prediction = random_forest.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_prediction)
rf_accuracy = accuracy_score(y_test, rf_prediction)

print("DECISION TREE ACCURACY:", round(dt_accuracy * 100, 2), "%")
print("RANDOM FOREST ACCURACY:", round(rf_accuracy * 100, 2), "%")


joblib.dump(random_forest, "model.pkl")

print("\nRandom Forest model saved!")


anomaly_model = IsolationForest(
    contamination=0.2,
    random_state=42
)

anomaly_model.fit(X_train)


transactions = [
    [150, 2, 0, 5],
    [9000, 14, 1, 700],
    [300, 2, 0, 4],
    [15000, 18, 1, 900]
]

print("\nREAL TIME TRANSACTIONS")

for t in transactions:

    new = pd.DataFrame(
        [t],
        columns=X.columns
    )

    prediction = random_forest.predict(new)[0]
    probability = random_forest.predict_proba(new)[0][1] * 100
    anomaly = anomaly_model.predict(new)[0]

    result = "FRAUD" if prediction == 1 else "NORMAL"

    if probability >= 70:
        risk = "HIGH"
    elif probability >= 40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    unusual = "UNUSUAL" if anomaly == -1 else "NORMAL"

    print("\nTransaction:", t)
    print("Result:", result)
    print("Fraud Probability:", round(probability, 2), "%")
    print("Risk:", risk)
    print("Anomaly:", unusual)


print("\nPROJECT READY")