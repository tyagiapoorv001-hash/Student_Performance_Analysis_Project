# ===============================
# Student Performance Prediction
# ===============================

# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 2️⃣ Load Dataset
df = pd.read_csv("student_data.csv")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())


# 3️⃣ Data Cleaning & Encoding
# Convert Gender to numerical
df['gender'] = df['gender'].map({'M': 0, 'F': 1})

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())


# 4️⃣ Exploratory Data Analysis (EDA)

# Study hours vs Pass
plt.figure()
sns.boxplot(x='pass', y='study_hours', data=df)
plt.title("Study Hours vs Pass")
plt.savefig("study_hours_vs_pass.png")
plt.show()

# Attendance vs Pass
plt.figure()
sns.boxplot(x='pass', y='attendance', data=df)
plt.title("Attendance vs Pass")
plt.savefig("attendance_vs_pass.png")
plt.show()

# Correlation Heatmap
numeric_df = df.select_dtypes(include=['int64','float64'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()


# 5️⃣ Feature Selection
X = df[['study_hours', 'attendance', 'previous_score', 'sleep_hours', 'internet']]
y = df['pass']


# 6️⃣ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 7️⃣ Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 8️⃣ Train Machine Learning Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)


# 9️⃣ Model Prediction
y_pred = model.predict(X_test_scaled)


# 🔟 Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 1️⃣1️⃣ Predict New Student (Manual Input)
print("\n--- Predict New Student Result ---")
study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance percentage: "))
previous_score = float(input("Enter previous score: "))
sleep_hours = float(input("Enter sleep hours: "))
internet = int(input("Internet access (1 = Yes, 0 = No): "))

new_data = np.array([[study_hours, attendance, previous_score, sleep_hours, internet]])
new_data_scaled = scaler.transform(new_data)

prediction = model.predict(new_data_scaled)

if prediction[0] == 1:
    print("✅ Prediction: Student will PASS")
else:
    print("❌ Prediction: Student will FAIL")
