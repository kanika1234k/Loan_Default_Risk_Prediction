import matplotlib.pyplot as plt
import seaborn as sns
# Check import sucess
print("Libraries imported successfully")
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Dataset
df = pd.read_csv("data/Loan_default.csv")

# Features and Target
X = df.drop("Default", axis=1)
y = df["Default"]

# Drop LoanID (ID column)
if "LoanID" in X.columns:
    X = X.drop("LoanID", axis=1)

# Separate columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# Numeric Pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical Pipeline
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Save Model
joblib.dump(model, "models/model.pkl")

print("Model saved successfully!")

# Loan Default Distribution Graph
plt.figure(figsize=(6,4))

sns.countplot(x="Default", data=df)

plt.title("Loan Default Distribution")
plt.savefig("Images/loan_default_distribution.png")

plt.show()
#Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(df["Age"],bins=10)
plt.title("Age Distribution")

plt.savefig("Images/loan_age_distribution.png")
plt.show()
import joblib 
model.fit(X_train,y_train)
joblib.dump(model,"model/models.pkl")
print("Model saved successfully!")