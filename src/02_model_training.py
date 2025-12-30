import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "pima-indians-diabetes.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Data Cleaning: Replace 0 with Median (where 0 is physically impossible)
cols_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_fix] = df[cols_fix].replace(0, np.nan)
df[cols_fix] = df[cols_fix].fillna(df[cols_fix].median())

# Split
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train, y_train)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)

# Save
joblib.dump(svm, os.path.join(MODEL_DIR, "svm_model.pkl"))
joblib.dump(dt, os.path.join(MODEL_DIR, "dt_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl")) # Save scaler for GUI use later

print(f"✅Training complete. Models saved in: {MODEL_DIR}")