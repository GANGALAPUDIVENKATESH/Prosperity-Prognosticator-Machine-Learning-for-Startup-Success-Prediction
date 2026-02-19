
# 1️⃣ Install required libraries
!pip install flask joblib scikit-learn pandas numpy matplotlib seaborn

# 2️⃣ Import libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 3️⃣ Upload CSV file
from google.colab import files
uploaded = files.upload()

# Get uploaded filename
filename = list(uploaded.keys())[0]

# 4️⃣ Load dataset
# Use pd.read_excel for .xlsx files
if filename.endswith('.xlsx'):
    data = pd.read_excel(filename)
elif filename.endswith('.csv'):
    data = pd.read_csv(filename)
else:
    raise ValueError("Unsupported file format. Please upload a .csv or .xlsx file.")

print("Dataset Shape:", data.shape)
data.head()

# 5️⃣ Basic EDA
print("\nMissing Values:\n", data.isnull().sum())
print("\nStatistical Summary:\n", data.describe())

# Drop non-numeric columns for correlation calculation
numeric_data = data.select_dtypes(include=[np.number])

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation")
plt.show()

# 6️⃣ Feature & Target split
# Identify columns to drop (IDs, text, dates that aren't processed, and redundant/unnamed)
cols_to_drop = [
    'Unnamed: 0', 'id', 'object_id', 'name', 'zip_code', 'city',
    'state_code', 'state_code.1', 'category_code', 'status',
    'founded_at', 'closed_at', 'first_funding_at', 'last_funding_at',
    'Unnamed: 6'
]
# Filter out columns that might not exist in the dataframe after initial cleaning or if the dataset changes.
cols_to_drop = [col for col in cols_to_drop if col in data.columns]

X = data.drop(columns=cols_to_drop + ["labels"], errors='ignore')
y = data["labels"]

# 7️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8️⃣ Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 9️⃣ Random Forest + GridSearchCV
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "bootstrap": [True, False]
}

rf = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters Found:")
print(grid.best_params_)

# 🔟 Model Evaluation
train_pred = best_model.predict(X_train_scaled)
test_pred = best_model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print("\nTraining Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

print("\nClassification Report:\n")
print(classification_report(y_test, test_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 1️⃣1️⃣ Save Model & Scaler
joblib.dump(best_model, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel & Scaler Saved Successfully!")

# 1️⃣2️⃣ Download Saved Files
files.download("random_forest_model.pkl")
files.download("scaler.pkl")

# 1️⃣3️⃣ Generate Flask App Code (Optional Deployment)
flask_code = """
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Startup Success Prediction App"

@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_data = scaler.transform([data])
    prediction = model.predict(final_data)[0]
    result = "Acquired / Successful" if prediction == 1 else "Closed / Failed"
    return result

if __name__ == "__main__":
    app.run(debug=True)
"""

with open("app.py", "w") as f:
    f.write(flask_code)

files.download("app.py")

print("\nFlask app.py file generated!")
