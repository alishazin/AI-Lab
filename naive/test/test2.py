from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
iris = pd.read_csv('data.csv')

X = iris.iloc[:, 1:-1]
y = iris.iloc[:, -1]

# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

print(X_train.shape)
print(X_test.shape)

# print(y_train.shape)
# print(y_test.shape)

gnb=GaussianNB()

gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)

pred_df = pd.read_csv("predict.csv")

y_pred=gnb.predict(pred_df)
print(y_pred)

# pred_species = [{'1': "Yes", '2': "No"}[str(p)] for p in y_pred] 
# print("Predictions:", pred_species)

# # confusion metric
# from sklearn.metrics import confusion_matrix 
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# # accuracy score
# from sklearn.metrics import accuracy_score 
# ac = accuracy_score(y_test, y_pred)
# print(ac)

# Overcast = 1
# Sunny = 2
# Rain = 3

# Hot = 1
# Mild = 2
# Cool = 3

# High = 1
# Normal = 2

# Weak = 1
# Strong = 2

# Yes = 1
# No = 2