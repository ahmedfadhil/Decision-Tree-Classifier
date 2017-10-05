import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier

dataset = load_digits()

X, y = dataset.data, dataset.target
for class_name, class_count in zip(dataset.target_names, np.bincount(dataset.target)):
    print(class_name, class_count)

y_binary_imbalanced = y.copy()
y_binary_imbalanced[y_binary_imbalanced != 1] = 0
print('original label:', y[1:30])
print('new label:', y_binary_imbalanced[1:30])

new_class = np.bincount(y_binary_imbalanced)
print(new_class)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary_imbalanced, random_state=0)
svm = SVC(kernel='linear', C=1).fit(X_train, y_train)
# svm = SVC(kernel='rbf', C=1).fit(X_train, y_train)

svm.score(X_test, y_test)

# Now create the dummy version of the algorithm
dummy_majority = DummyClassifier(strategy='major_frequent').fit(X_train, y_train)

y_dummy_predictions = dummy_majority.predict(X_test)
y_dummy_predictions

dummy_majority.score(X_test, y_test)
dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test, y_majority_predicted)

print('most freq class dummy classifier \n', confusion)




dummy_majority = DummyClassifier(strategy='stratified').fit(X_train, y_train)
y_majority_predicted = dummy_majority.predict(X_test)
confusion = confusion_matrix(y_test, y_majority_predicted)

print('random class proportion  \n', confusion)



svm = SVC(kernel='linear',C=1).fit(X_train,y_train)
svm_predicted = svm.predict(X_test)
confusion = confusion_matrix(y_test,svm_predicted)

lr = LogisticRegression().fit(X_train,y_train)
lr_predict = lr.predict(X_test)

confusion = confusion_matrix(y_test,lr_predict)



dt = DecisionTreeClassifier(max_depth=2).fit(X_train,y_train)

tree_predictor = dt.predict(X_test)
cpnfi = confusion_matrix(y_test,tree_predictor)

print('the decision tree classfier \n', cpnfi)





















