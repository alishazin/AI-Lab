
import pandas as pd
from naive_bayes import GaussianNaiveBayesClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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