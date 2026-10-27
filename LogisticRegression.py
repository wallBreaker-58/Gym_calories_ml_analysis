
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    log_loss, roc_auc_score
)

# Step 1: Load dataset
df = pd.read_csv('gym_dataset_cleaned.csv')

# Step 2: Create a binary target
# High = calories burned at or above the median
median_calories = df['Calories_Burned'].median()

df['Calorie_Category'] = (
    df['Calories_Burned'] >= median_calories
).astype(int)

# 0 = Low calories, 1 = High calories

# Step 3: Select input features
X = df[
    [
        'Session_Duration (hours)',
        'Weight (kg)',
        'Age',
        'Avg_BPM',
        'Workout_Frequency (days/week)'
    ]
]

y = df['Calorie_Category']

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Create and train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

