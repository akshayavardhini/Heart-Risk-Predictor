#!/usr/bin/env python
# coding: utf-8

# In[175]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[158]:


file=r"C:\Users\Akshaya\OneDrive\Desktop\Internship\Task3\heart.csv"
df=pd.read_csv(file)


# In[159]:


df.shape


# In[160]:


print(df.columns.tolist())


# In[161]:


df.isnull().sum()


# In[162]:


df.duplicated().sum()


# In[163]:


df=df.drop_duplicates()


# In[164]:


df.shape


# In[165]:


x=df.drop('target', axis=1)
y=df['target']


# In[166]:


scal=StandardScaler()
x_scal=scal.fit_transform(x)


# In[167]:


x_train, x_test, y_train, y_test= train_test_split(x_scal, y, test_size=0.2, random_state=45)


# In[170]:


mod=RandomForestClassifier(random_state=42)
mod.fit(x_train, y_train)
y_pred=mod.predict(x_test)
print("Random Forest")
print("Accuracy", accuracy_score(y_test, y_pred))
print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
print("Classification Matrix\n", classification_report(y_test, y_pred))


# In[ ]:


model = HistGradientBoostingClassifier(random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Hist Gradradient")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
print("Classification Matrix\n", classification_report(y_test, y_pred))


# In[ ]:


voting_model = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=1000)),
    ('rf', RandomForestClassifier()),
    ('svc', SVC(probability=True))
], voting='soft')

voting_model.fit(x_train, y_train)
y_pred = voting_model.predict(x_test)
print("Ensambling")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
print("Classification Matrix\n", classification_report(y_test, y_pred))


# In[172]:


svc_model = SVC(kernel='rbf', probability=True)
svc_model.fit(x_train, y_train)
svc_pred = svc_model.predict(x_test)
print("SVC\nAccuracy:", accuracy_score(y_test, svc_pred))
print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
print("Classification Matrix\n", classification_report(y_test, y_pred))


# In[173]:


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)
print("KNN\nAccuracy:", accuracy_score(y_test, knn_pred))
print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
print("Classification Matrix\n", classification_report(y_test, y_pred))


# In[174]:


nb_model = GaussianNB()
nb_model.fit(x_train, y_train)
nb_pred = nb_model.predict(x_test)
print("Naive Bayes\nAccuracy:", accuracy_score(y_test, nb_pred))
print("Confusion Matrix\n", confusion_matrix(y_test, y_pred))
print("Classification Matrix\n", classification_report(y_test, y_pred))


# In[177]:


import joblib
joblib.dump(knn_model, "knn_model.pkl")
joblib.dump(scal, "scaler.pkl")
print("✅ Model and scaler saved successfully.")

