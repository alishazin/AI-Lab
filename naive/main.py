
import pandas as pd
from naive_bayes import MultinomialNaiveBayesClassifier, GaussianNaiveBayesClassifier
from testing import k_fold_cross_validation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

""" Salary Data """

# df = pd.read_csv('datasets/salary.csv')

# mnb = MultinomialNaiveBayesClassifier()

# X = df.iloc[:, 0:-1]
# Y = df.iloc[:, -1]

# mnb.fit(X, Y)

# y_test = pd.DataFrame(
#     [
#         ["facebook","computer programmer","bachelors"],
#         ["abc pharma","business manager","bachelors"],
#     ], 
#     columns=["company","job","degree"]
# )


# y_pred =  mnb.predict(y_test)
# print(y_pred)

""" PlayTennis Data """

# df = pd.read_csv('datasets/data.csv')

# mnb = MultinomialNaiveBayesClassifier(alpha=1)

# X = df.iloc[:, 1:-1]
# Y = df.iloc[:, -1]

# mnb.fit(X, Y)

# y_test = pd.DataFrame(
#     [
#         ["Sunny", "Cool", "High", "Strong"],
#         ["Overcast", "Hot", "High", "Strong"],
#     ], 
#     columns=["Outlook", "Temperature", "Humidity", "Wind"]
# )

# y_pred =  mnb.predict(y_test)
# print(y_pred)

""" Gender  Data """

# df = pd.read_csv('datasets/gender.csv')

# gnb = GaussianNaiveBayesClassifier()

# X = df.iloc[:, 0:-1]
# Y = df.iloc[:, -1]

# gnb.fit(X, Y)

# y_test = pd.DataFrame(
#     [
#         [6, 130, 8],
#     ], 
#     columns=["Height", "Weight", "Foot Size"]
# )

# y_pred =  gnb.predict(y_test)
# print(y_pred)

""" Iris Dataset """
from sklearn.datasets import load_iris 
df = load_iris()

X = pd.DataFrame(df.data)
y = pd.Series(df.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

gnb=GaussianNaiveBayesClassifier()

gnb.fit(X_train,y_train)

y_pred=gnb.predict(X_test)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=df.target_names)
cm_display.plot()
plt.show()

# accuracy score
from sklearn.metrics import accuracy_score 
ac = accuracy_score(y_test, y_pred)
print(ac)

# score = k_fold_cross_validation(gnb.fit, gnb.predict, gnb.reset, X, y, 10)
# print(score)