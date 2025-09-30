# Elevate Labs - Task 6: K-Nearest Neighbors (KNN) Classification 

### **Objective**
The goal of this task was to understand, implement, and correctly tune the **K-Nearest Neighbors (KNN)** classification algorithm. This project used the classic Iris dataset to classify flower species.

### **Workflow & Key Steps**

1.  **Preprocessing**:
    * The categorical target variable (`Species`) was converted to numerical labels (0, 1, 2).
    * **Crucially, the features (sepal and petal dimensions) were scaled using `StandardScaler`**. This was necessary because KNN is a distance-based algorithm, and scaling prevents features with larger ranges from dominating the distance calculation.
2.  **Model Tuning (The Elbow Method)**:
    * KNN's main hyperparameter, **K** (the number of neighbors), was tuned by running the model for K values from 1 to 20.
    * An **Elbow Method plot** was generated to visualize the error rate versus K. The optimal K value was selected at the point where the error rate began to plateau, which was determined to be **K=[Insert Optimal K Value]**.
3.  **Final Model Evaluation**:
    * The final `KNeighborsClassifier` was trained using the selected optimal K.
    * The model achieved an **Accuracy Score** of **[Insert Final Accuracy Score]**.
    * The **Classification Report** and **Confusion Matrix** confirmed the model's strong performance across all three Iris classes.

### **Conclusion**
The project successfully implemented a well-tuned KNN classifier. The process emphasized the necessity of feature scaling for distance-based models and the importance of hyperparameter tuning using the Elbow Method.
