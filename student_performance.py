# Student Performance Prediction
# Author: Shaik Aman

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Encode categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Create target variable (pass/fail)
df['average_score'] = (df['math score'] + df['reading score'] + df['writing score'])/3
df['result'] = df['average_score'].apply(lambda x: 1 if x >= 50 else 0)

# Features & Labels
X = df.drop(['average_score','result'], axis=1)
y = df['result']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Model
model = RandomForestClassifier()
model.fit(X_train,y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

# Visualization
sns.heatmap(df.corr(),annot=False,cmap='coolwarm')
plt.title("Feature Correlation")
plt.savefig("correlation.png")
plt.show()
