import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# take data

X_train, X_test, y_train, y_test = np.load("../data/5obj.npy", allow_pickle = True)

X_train = X_train.reshape(37,96100)
X_test = X_test.reshape(13,96100)
X_train.shape

# try RF - k-fold cross val (cv = k)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
rf = cross_val_score(clf, X_train, y_train, cv=3, scoring='accuracy') #k-fold cross validation
print(rf)

# input data scaling

scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)

clf.fit(X_train_scale, y_train)
ss = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy")
print(ss)

# evaluation

X_test_scale = scaler.fit_transform(X_test)
prediction = clf.predict(X_test_scale)
result = (prediction == y_test).mean()
print(result)