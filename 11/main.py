from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics 

# 1. Load the Iris dataset 
iris = load_iris() 
X = iris.data 
y = iris.target

#Split data 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=1) 

# 3. Initialize the KNN classifier with 3 neighbors 
classifier_knn = KNeighborsClassifier(n_neighbors=3) 

# 4. Train the classifier using the training data 
classifier_knn.fit(X_train, y_train) 

# 5. Predict the labels for the test set 
y_pred = classifier_knn.predict(X_test) 

# 6. Evaluate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, y_pred) 
print("Accuracy:", accuracy) 

# 7. Provide sample data for prediction 
sample = [[5, 5, 3, 2], [2, 4, 3, 5]] 

# 8. Make predictions on the sample data 
preds = classifier_knn.predict(sample) 

# 9. Map the predicted labels to species names 
pred_species = [iris.target_names[p] for p in preds] 

# 10. Display the predictions 
print("Predictions:", pred_species)