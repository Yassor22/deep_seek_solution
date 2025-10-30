import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib as jb
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df= pd.read_excel('augmented_rectal_cancer_data.xlsx')
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
df['scaled_age']= ss.fit_transform(df[['Age']])
df['scaled_distance from verge by Cm']= ss.fit_transform(df[['distance']])
df['scaled_dimensions']= ss.fit_transform(df[['dimensions']])
df['saled_Anal_canal']= ss.fit_transform(df[['Anal_canal']])
from sklearn.preprocessing import LabelEncoder
lb= LabelEncoder()

df['Encoded_gender']= lb.fit_transform(df['gender'])
df['Encoded_stageT']= lb.fit_transform(df['stageT'])
df['Encoded_stageN']= lb.fit_transform(df['stageN'])
df['Encoded_sphincter']= lb.fit_transform(df['sphincter'])
df['Encoded_Biopsy']= lb.fit_transform(df['Biopsy'])
df['Encoded_TNT']= lb.fit_transform(df['TNT'])
df['Encoded_Course']= lb.fit_transform(df['Course'])
df['Encoded_Response']= lb.fit_transform(df['Response'])

X=df[['scaled_age','scaled_distance from verge by Cm','scaled_dimensions','saled_Anal_canal','Encoded_gender', 'Encoded_stageT','Encoded_stageN', 'Encoded_sphincter', 'Encoded_Biopsy', 'Encoded_TNT', 'Encoded_Course']]
y = df['Response']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC
svc= SVC(C=10000, degree= 30)
