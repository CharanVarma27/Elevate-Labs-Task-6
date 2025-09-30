import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. Import Data and Explore ---
df = pd.read_csv('Iris.csv')

# Drop irrelevant 'Id' column
df = df.drop('Id', axis=1)

# Encode the target variable ('Species')
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Define Features (X) and Target (y)
X = df.drop('Species', axis=1)
y = df['Species']

# --- 2. Scaling and Splitting Data ---
# Scaling is mandatory for KNN!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
print("Data preparation complete.")

# --- 3. Elbow Method: Finding the Optimal K ---
error_rate = []
# Test K values from 1 to 20
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    # Calculate the error rate (1 - accuracy)
    error_rate.append(np.mean(pred_i != y_test))

# Plotting the Elbow Method result
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), error_rate, color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value (Elbow Method)')
plt.xlabel('K Value (Number of Neighbors)')
plt.ylabel('Error Rate')
plt.grid(True)
plt.show() # 

# Based on the plot, select the K where the error rate minimizes or plateaus
optimal_k = 5 # Example: often K=5 is a good default or visual minimum

# --- 4. Train and Evaluate KNN with Optimal K ---
print(f"\nTraining KNN Model with Optimal K = {optimal_k}...")
final_knn = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn.fit(X_train, y_train)
final_pred = final_knn.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, final_pred)
print(f"Final Model Accuracy (K={optimal_k}): {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, final_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, final_pred))

print("\nKNN implementation complete.")
