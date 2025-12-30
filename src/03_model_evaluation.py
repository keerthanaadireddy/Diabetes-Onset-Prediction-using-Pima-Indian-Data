import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
DATA_PATH = os.path.join(BASE_DIR, "data", "pima-indians-diabetes.csv")
os.makedirs(RESULT_DIR, exist_ok=True)

# Load Models and Data
svm = joblib.load(os.path.join(MODEL_DIR, "svm_model.pkl"))
dt = joblib.load(os.path.join(MODEL_DIR, "dt_model.pkl"))
scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
df = pd.read_csv(DATA_PATH)

# Pre-process same as training
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_scaled = scaler.transform(X)

# Predict & Evaluate
y_pred = dt.predict(X_scaled)
acc = accuracy_score(y, y_pred)

print(f"Decision Tree Accuracy: {acc*100:.2f}%")
print(classification_report(y, y_pred))

# Save Confusion Matrix Plot
plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y, y_pred), annot=True, fmt='d', cmap='Greens')
plt.title("Confusion Matrix")
plt.savefig(os.path.join(RESULT_DIR, "confusion_matrix.png"))
plt.show()

print(f"Evaluation finished. Charts saved in: {RESULT_DIR}")