import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------
# Load Dataset
# ---------------------------------

iris = load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("Dataset Shape:", X.shape)

# ---------------------------------
# Train Test Split
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------
# Feature Scaling
# ---------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------
# Models
# ---------------------------------

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Support Vector Machine": SVC()
}

results = {}

print("=" * 60)

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"\n{name}")
    print("-" * 40)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, predictions))

# ---------------------------------
# Comparison
# ---------------------------------

plt.figure(figsize=(7,5))
plt.bar(results.keys(), results.values())

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8,1.0)

plt.show()