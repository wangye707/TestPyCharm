import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import sklearn.svm as svm

digits = load_digits()

# print(digits)
x, y = digits.data, digits.target

print(max(y), min(y))
# y=label_binarize(y,classes=list(range(10)))

x_train, x_test, y_train, y_test = train_test_split(x, y)
model = OneVsRestClassifier(svm.SVC(kernel='linear'))
clf = model.fit(x_train, y_train)
# clf.score()
