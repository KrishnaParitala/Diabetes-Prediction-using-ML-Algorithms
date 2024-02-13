import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r'C:\Users\rakes\OneDrive\Desktop\Projects\diabetes.csv')
df
df.head()
df.info()
df.info()
df.describe()
df.describe()
df.isnull()
df[df.duplicated()]
sns.pairplot(df,hue='Outcome')
plt.show()
sns.histplot(df['Glucose'],bins=15,kde=True)
plt.xlabel('Glucose')
plt.ylabel('Distribution')
plt.title('Glucose level Distribution')
plt.show()
df.corr()
sns.heatmap(df.corr(),annot=True)
x=df.drop(['Outcome'],axis=1)
y=df['Outcome']
print(x)
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=615)
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('-----Logistic Regression------\n')
print('Accuracy:',accuracy_score(y_test,y_pred))
print('confusion_matrix:',confusion_matrix(y_test,y_pred))
print('classification_report:',classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
Rfc=RandomForestClassifier()
Rfc.fit(x_train,y_train)
y1_pred=Rfc.predict(x_test)
print('-----Randon Forest Classifier------\n')
As1=accuracy_score(y_test,y1_pred)
print('Accuracy:',accuracy_score(y_test,y1_pred))
print('confusion_matrix:',confusion_matrix(y_test,y1_pred))
print('classification_report:',classification_report(y_test,y1_pred))
predicted1=Rfc.predict([[5,130,80,20,80,43.5,0.856,50]])
for i in predicted1:
    predicted11=i
print(predicted11)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
y2_pred=knn.predict(x_test)
print('-----KNeighborsClassifier------\n')
As2=accuracy_score(y_test,y2_pred)
print('Accuracy:',accuracy_score(y_test,y2_pred))
print('confusion_matrix:',confusion_matrix(y_test,y2_pred))
print('classification_report:',classification_report(y_test,y2_pred))
predicted2=knn.predict([[5,130,80,20,80,43.5,0.856,50]])
for i in predicted2:
    predicted12=i
print(predicted12)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
y3_pred=nb.predict(x_test)
print('-----Naive Bayes------\n')
As3=accuracy_score(y_test,y3_pred)
print('Accuracy:',accuracy_score(y_test,y3_pred))
print('confusion_matrix:',confusion_matrix(y_test,y3_pred))
print('classification_report:',classification_report(y_test,y3_pred))
predicted3=nb.predict([[5,130,80,20,80,43.5,0.856,50]])
for i in predicted3:
    predicted13=i
print(predicted13)
from sklearn.tree import DecisionTreeClassifier
DT=DecisionTreeClassifier()
DT.fit(x_train,y_train)
y4_pred=DT.predict(x_test)
print('-----DecisionTreeClassifier------\n')
As4=accuracy_score(y_test,y4_pred)
print('Accuracy:',accuracy_score(y_test,y4_pred))
print('confusion_matrix:',confusion_matrix(y_test,y4_pred))
print('classification_report:',classification_report(y_test,y4_pred))
predicted4=DT.predict([[5,130,80,20,80,43.5,0.856,50]])
for i in predicted4:
    predicted14=i
print(predicted14)
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(max_iter=700)
LR.fit(x_train,y_train)
y5_pred=LR.predict(x_test)
print('-----Logistic Regression------\n')
As5=accuracy_score(y_test,y5_pred)
print('Accuracy:',accuracy_score(y_test,y5_pred))
print('confusion_matrix:',confusion_matrix(y_test,y5_pred))
print('classification_report:',classification_report(y_test,y5_pred))
predicted5=LR.predict([[5,130,80,20,80,43.5,0.856,50]])
for i in predicted5:
    predicted15=i
print(predicted15)
result=pd.DataFrame({
    'Models':['RandomForest','KNeighbours','GaussianNB','DecisionTree','LogisticRegression',],
    'Score': [As1*100 , As2*100 ,As3*100 , As4*100, As5*100]})
result
import matplotlib.pyplot as plt
result.plot.bar(x='Models',y='Score')
plt.title('Models vs Accuracy_Scores')
plt.ylabel('Score')
plt.show()
pre=pd.DataFrame({
    'Models':['RandomForest','KNeighbours','GaussianNB','DecisionTree','LogisticRegression',],
    'Predicted_Diabetes': [predicted11, predicted12 ,predicted13 , predicted14, predicted15]})
pre
