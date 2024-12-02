from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, load_iris, load_digits, load_breast_cancer
from sklearn.metrics import confusion_matrix, accuracy_score

wine = load_breast_cancer()

X = wine.data
print(wine.feature_names)
print(wine.target_names)
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: ")
print(cm)

ac = accuracy_score(y_test, y_pred)
print("Accuracy: ", ac)
