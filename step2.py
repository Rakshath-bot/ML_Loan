import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD AND CLEAN DATA
df = pd.read_csv('loan_status_data.csv')

# Handling missing values and outliers
df.replace('?', np.nan, inplace=True)
df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')
df['rate'] = pd.to_numeric(df['rate'], errors='coerce')

# Filter realistic ages and employment lengths
df_clean = df[(df['age'] < 100) & (df['emp_length'] < 60)].copy()

# Impute remaining missing values with median
df_clean['emp_length'] = df_clean['emp_length'].fillna(df_clean['emp_length'].median())
df_clean['rate'] = df_clean['rate'].fillna(df_clean['rate'].median())

# 2. VISUALIZATION: KEY FEATURE DISTRIBUTIONS
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Income Distribution
sns.histplot(df_clean[df_clean['income'] < 200000]['income'], bins=30, kde=True, ax=axes[0], color='skyblue')
axes[0].set_title('Distribution of Income (<$200k)')

# Loan Amount Distribution
sns.histplot(df_clean['loan_amount'], bins=30, kde=True, ax=axes[1], color='salmon')
axes[1].set_title('Distribution of Loan Amount')

# Loan Grade (Credit Proxy) Distribution
sns.countplot(data=df_clean, x='loan_grade', order=sorted(df_clean['loan_grade'].unique()), ax=axes[2], palette='viridis')
axes[2].set_title('Distribution of Loan Grade')
plt.tight_layout()
plt.show()

# 3. VISUALIZATION: PATTERN ANALYSIS (Demographic & Financial)
plt.figure(figsize=(16, 10))

# Home Ownership vs Approval Proportion
plt.subplot(2, 2, 1)
home_approval = pd.crosstab(df_clean['home'], df_clean['loan_status'], normalize='index')
home_approval.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#66c2a5', '#fc8d62'])
plt.title('Approval Rate by Home Ownership')
plt.ylabel('Proportion (0=Approved, 1=Default)')

# Loan Intent vs Approval Proportion
plt.subplot(2, 2, 2)
intent_approval = pd.crosstab(df_clean['loan_intent'], df_clean['loan_status'], normalize='index')
intent_approval.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#66c2a5', '#fc8d62'])
plt.title('Approval Rate by Loan Intent')

# Income vs Status
plt.subplot(2, 2, 3)
sns.boxplot(data=df_clean, x='loan_status', y='income', palette='Set2')
plt.yscale('log')
plt.title('Income vs Loan Status (Log Scale)')

# Loan Grade vs Status
plt.subplot(2, 2, 4)
grade_approval = pd.crosstab(df_clean['loan_grade'], df_clean['loan_status'], normalize='index')
grade_approval.plot(kind='bar', stacked=True, ax=plt.gca(), color=['#66c2a5', '#fc8d62'])
plt.title('Default Rate by Loan Grade')
plt.tight_layout()
plt.show()

# 4. PRINTED OUTPUT (STATISTICS)
print("--- SUMMARY STATISTICS BY LOAN STATUS (0=Approved, 1=Default) ---")
stats = df_clean.groupby('loan_status').agg({
    'income': 'mean',
    'loan_amount': 'mean',
    'rate': 'mean',
    'age': 'mean'
}).round(2)
print(stats)

print("\n--- APPROVAL RATE BY HOME OWNERSHIP ---")
approval_by_home = (1 - df_clean.groupby('home')['loan_status'].mean()).sort_values(ascending=False).round(4) * 100
print(approval_by_home.apply(lambda x: f"{x:.2f}%"))

print("\n--- APPROVAL RATE BY LOAN INTENT ---")
approval_by_intent = (1 - df_clean.groupby('loan_intent')['loan_status'].mean()).sort_values(ascending=False).round(4) * 100
print(approval_by_intent.apply(lambda x: f"{x:.2f}%"))