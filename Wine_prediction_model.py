import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.naive_bayes import GaussianNB,MultinomialNB



 
l = load_wine()
x = l.data
y = l.target
model = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
model.fit(X_train,y_train)
p = model.score(X_test,y_test)
model1 = MultinomialNB()
model1.fit(X_train,y_train)
p1 = model1.score(X_test,y_test)
print(p,p1)