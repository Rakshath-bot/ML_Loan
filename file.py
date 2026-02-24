import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# 1. LOAD DATA
df = pd.read_csv('loan_status_data.csv')

# 2. DATA CLEANING
# Replace '?' with NaN and convert types
df.replace('?', np.nan, inplace=True)
df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Remove outliers
df = df[df['age'] < 100].copy()
df = df[df['emp_length'] < 60].copy()

# Impute missing values with median
df['emp_length'] = df['emp_length'].fillna(df['emp_length'].median())
df['rate'] = df['rate'].fillna(df['rate'].median())

# 3. FEATURE ENGINEERING
# Create a precise loan-to-income ratio
df['loan_percent_ratio'] = df['loan_amount'] / df['income']
df.drop(columns=['loan_percent_income'], inplace=True)

# 4. ENCODING
# Ordinal encoding for grade
grade_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
df['loan_grade_numeric'] = df['loan_grade'].map(grade_mapping)

# Binary encoding for default history
df['person_default_numeric'] = df['person_default'].map({'Y': 1, 'N': 0})

# One-hot encoding for categorical variables
df_processed = pd.get_dummies(df, columns=['home', 'loan_intent'], prefix=['home', 'intent'], drop_first=True)

# Drop original text columns
df_processed.drop(columns=['loan_grade', 'person_default'], inplace=True)

# 5. SCALING
scaler = StandardScaler()
num_cols = ['age', 'income', 'emp_length', 'loan_amount', 'rate', 'person_cred_hist_length', 'loan_percent_ratio', 'loan_grade_numeric']
df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])

# 6. MODEL TRAINING
X = df_processed.drop(columns=['loan_status'])
y = df_processed['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 7. EVALUATION
y_pred = rf_model.predict(X_test)
print("--- Model Performance ---")
print(classification_report(y_test, y_pred))

# Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n--- Top 5 Important Features ---")
print(importances.head(5))
import matplotlib.pyplot as plt

# 1. Show EDA Plots
# (Assuming 'df' is your cleaned dataframe)
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.countplot(data=df, x='loan_grade', hue='loan_status')
plt.title('Loan Status by Loan Grade')
# ... (add other subplots here)
plt.tight_layout()
plt.show()  # <--- This displays the plot on your screen

# 2. Show Confusion Matrix and ROC Curve
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
# ... (add matrix and roc code here)
plt.tight_layout()
plt.show()  # <--- This displays the evaluation plots

# 3. Show Feature Importance
plt.figure(figsize=(10, 8))
sns.barplot(x=importances, y=importances.index)
plt.title('Feature Importances')
plt.show()  # <--- This displays the importance chart