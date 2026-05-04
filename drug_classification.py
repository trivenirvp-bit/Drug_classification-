import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
df = pd.read_csv('drug200.csv')
df.head()

# ---- 1. Age distribution vs Drug type ----
fig, ax = plt.subplots(figsize=(8, 6))
sns.swarmplot(
    x='Drug',
    y='Age',
    data=df,
    palette='Set2',
    ax=ax
)
ax.set_title('Drug Distribution by Age')
ax.set_xlabel('Drug Type')
ax.set_ylabel('Age')
ax.tick_params(axis='x', rotation=45)
st.pyplot(fig)

# ---- 2. Correlation heatmap ----
fig, ax = plt.subplots(figsize=(8, 6))

corr = df.select_dtypes(include='number').corr()
sns.heatmap(
    corr,
    annot=True,
    cmap='coolwarm',
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

ax.set_title('Feature Correlation Heatmap')
st.pyplot(fig)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---- Encoding categorical features ----
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()

df['Sex'] = le_sex.fit_transform(df['Sex'])
df['BP'] = le_bp.fit_transform(df['BP'])
df['Cholesterol'] = le_chol.fit_transform(df['Cholesterol'])

# ---- Define Features (X) and Target (y) ----
X = df.drop('Drug', axis=1)
y = df['Drug']

# Optional: encode target if it's categorical
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# ---- Train-Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- Feature Scaling ----
scaler = StandardScaler()

# Fit ONLY on training data
X_train = scaler.fit_transform(X_train)

# Transform test data
X_test = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier()
model.fit(X_train, y_train)

# ---- Initialize & Train Model ----
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ---- Predictions ----
predictions = model.predict(X_test)

# ---- Evaluation ----
st.write("Classification Report:\n")
st.write(classification_report(y_test, predictions))

# ---- Confusion Matrix ----
cm = confusion_matrix(y_test, predictions)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=set(y_test),
    yticklabels=set(y_test),
    ax=ax
)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix')
st.pyplot(fig)

