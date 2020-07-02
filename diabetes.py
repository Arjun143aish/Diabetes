import os
import pandas as pd
import numpy as np

os.chdir("C:\\Users\\user\\Documents\\Python\\Heroku-Demo-master\\Diabetic")

FullRaw = pd.read_csv("diabetes.csv")

FullRaw.isnull().sum()

FullRaw['Outcome'].value_counts()/FullRaw.shape[0]*100

#FullRaw['Outcome'] = np.where(FullRaw['Outcome'] == 1,'Yes','No')

from sklearn.model_selection import train_test_split

Train,Test = train_test_split(FullRaw,test_size = 0.3, random_state =123)

Train_X = Train.drop(['Outcome'], axis =1)
Train_Y = Train['Outcome'].copy()
Test_X = Test.drop(['Outcome'], axis =1)
Test_Y = Test['Outcome'].copy()

from sklearn.linear_model import LogisticRegression

M1 = LogisticRegression(random_state=123).fit(Train_X,Train_Y)

Test_pred = M1.predict(Test_X)

from sklearn.metrics import confusion_matrix,f1_score,recall_score,precision_score
from sklearn.metrics import roc_auc_score,roc_curve

Con_Mat = confusion_matrix(Test_pred,Test_Y)

sum(np.diag(Con_Mat))/Test_Y.shape[0]*100

roc_auc_score(Test_pred,Test_Y)*100

f1_score(Test_pred,Test_Y)*100
recall_score(Test_pred,Test_Y)*100
precision_score(Test_pred,Test_Y)*100

from sklearn.ensemble import RandomForestClassifier

RF_Model = RandomForestClassifier(random_state=123).fit(Train_X,Train_Y)

RF_Pred = RF_Model.predict(Test_X)

RF_Con = confusion_matrix(RF_Pred,Test_Y)

sum(np.diag(RF_Con))/Test_Y.shape[0]*100
roc_auc_score(RF_Pred,Test_Y)*100
f1_score(RF_Pred,Test_Y)*100

import pickle

pickle.dump(RF_Model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

model.predict([[4,141,74,0,0,27.6,0.244,40]])




