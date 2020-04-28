import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#import dataset 

dataset=pd.read_csv('credit.csv')
dataset=pd.read_csv('bank-full.csv')
dataset=dataset["y","job","duration","month","balance","poutcome","age","day"]

#encoding categorical data
job=pd.get_dummies(dataset['job'],prefix="job")
marital=pd.get_dummies(dataset['marital'],prefix="marital")
education=pd.get_dummies(dataset['education'],prefix="education")
default=pd.get_dummies(dataset['default'],drop_first=True,prefix="default")
housing=pd.get_dummies(dataset['housing'],drop_first=True,prefix="housing")
loan=pd.get_dummies(dataset['loan'],drop_first=True,prefix="loan")
contact=pd.get_dummies(dataset['contact'],prefix="contact")
month=pd.get_dummies(dataset['month'],prefix="month")
poutcome=pd.get_dummies(dataset['poutcome'],prefix="poutcome")
y=pd.get_dummies(dataset['y'],drop_first=True,prefix="y")
dataset=pd.concat([dataset,job,marital,education,default,housing,loan,contact,month,poutcome,y],axis=1)
dataset=dataset.drop(['job','marital','education','default','housing','loan','contact','month','poutcome','y'],axis=1)

#spliting data
X=dataset.drop("y_yes",axis=1)
Y=dataset["y_yes"]
#test and train data
type(X)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


#Random Forest

model=RandomForestRegressor(100,oob_score=True,n_jobs=1,random_state=0)
model.fit(X_train,Y_train)
Y_predict=model.predict(X_test)




#logistic Regression

log_model=LogisticRegression()
log_model.fit(X_train,Y_train)
Y_predict=log_model.predict(X_test)


#Decision alogorithm

classifier=DecisionTreeClassifier()
classifier.fit(X_train,Y_train)
Y_predict=classifier.predict(X_test)


#confusion matrix

classification_report(Y_test,Y_predict)

cm=confusion_matrix(Y_test,Y_predict)
TP=cm[1,1]
TN=cm[0,0]
FP=cm[0,1]
FN=cm[1,0]

accuracy_score(Y_test,Y_predict)

from sklearn import metrics
metrics.roc_auc_score(Y_test,Y_predict)
