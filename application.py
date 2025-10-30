import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb
from sklearn.preprocessing import StandardScaler

df= pd.read_csv('augmented_rectal_cancer_data.csv')
st.sidebar.header('This is a predictive model for locally advanced Rectal cancer to TNT')
st.sidebar.image('https://tse4.mm.bing.net/th/id/OIP.j2TJw0vapJPgHsqrhGYUfgHaHa?pid=ImgDet&w=185&h=185&c=7&dpr=1.1&o=7&rm=3')
st.sidebar.write('This application has been built to predicit the reponse of locally advanced Rectal cancer to the types of total neoadjuvant therapy')
st.sidebar.write('The used model is RandomForestClassifier model with accuracy_score 78')
st.sidebar.write('Limitation : small-sized data\n If you can share deindentifiable data please Contact me')
st.sidebar.write('Created by ')
st.sidebar.markdown(' "Yasser Ali Okasha"')
st.sidebar.write('Supervisied by ')
st.sidebar.markdown('"professor.khaled Madbouly"')
st.sidebar.write('Contact details ')
st.sidebar.write("Email: yasser.okasha@alexmed.edu.eg")
st.title('Prediction of locally advanced rectal cancer response to TNT')
a1, a2,a3 = st.columns(3)
a1.image('cancer.JPG')
a2.image('radiotherapy.JPG')
a2.image('Capture.JPG')
a3.image('surgery.JPG')

st.text('Please Fill the following parameters about your ptient to predict the Response')
st.write('Demographic Data')
gender= st.selectbox('Gender',['Male','Female'])
Age= st.slider('Age',10,108)
bmi= st.number_input('BMI')
st.write('DRE')
length= st.number_input('Anal Canal length in CM')
distance= st.number_input('Distance from anal verge in CM')
quadrants_involved= st.slider('Quadrants involved', 0,4)
antorp= st.selectbox('Site',['Anterior','posterior','lateral','All'])
invasion= st.selectbox('Invsion of surrounding sturctures',['Yes', 'No'])

st.write('Pretreatment MRI findings')
stageT= st.selectbox('T stage',['T1','T2','T3a','T3b','T3c','T3d','T4a','T4b'])
StagN= st.selectbox('N stage',['N0','N1a','N1b','N1c','N2a','N2b','N2c','N3a','N3b','N3c'])
dimensions= st.number_input('tumour dimensions')
sphincter= st.selectbox('Sphincters involvment', ['Yes','No'])

st.write('Colonoscopic Biopsy')
biopsy=st.selectbox('Histopatholoical result of the Biopsy',['Well differentiated adenocarcinoma','Moderately differentiated adenocarcinoma','poorly differentiated adenocarcinoma','Mucoid adeoncarcinoma'])

st.write('TNT details')
tnt_c= st.selectbox('TNT Radiation', ['Short course','long course'])
tnt= st.selectbox('TNT Chemotherapy', ['induction','Consolidation'])

st.write('Postreatment MRI findings')
course=st.selectbox('Radiological Response',['Regression', 'Stationary','Progression'])
stagepT= st.selectbox('Post T stage',['T1','T2','T3a','T3b','T3c','T4a','T4b'])
StagpN= st.selectbox('Post N stage',['N0','N1','N2a','N2b','N2c','N3a','N3b','N3c'])

sphincterp= st.selectbox('Sphincters involvment', ['Involved','Spared'])

btn=st.button('Submit')

if btn== True:
    scaler= jb.load('Scaler.pkl')
    scaled_age = scaler.transform([[Age]])[0][0]
    scaled_distance = scaler.transform([[distance]])[0][0]
    scaled_quadrants_involved = scaler.transform([[quadrants_involved]])[0][0]
    scaled_dimensions = scaler.transform([[dimensions]])[0][0]
    scaled_length = scaler.transform([[length]])[0][0]
    model= jb.load('svc_model.pkl')
    gender_mapping={'Female':0, 'Male':1}
    gender_encoded= gender_mapping[gender]
    stageT_mapping={ 'T1':0,'T2':1, 'T3a':2,'T3b':3,'T3c':4,'T4a':5,'T4b':6}
    stageT_encoded= stageT_mapping[stageT]
    stageN_mapping={'N0':0,'N1':1,'N2a':2,'N2b':3,'N2c':4, 'N3a':5,'N3b':6,'N3c':7}
    stageN_encoded= stageN_mapping[StagN]
    sphincter_mapping={'Yes':0, 'No':1}
    sphincter_encoded= sphincter_mapping[sphincter]
    biopsy_mapping={'Well differentiated adenocarcinoma':8,'Moderately differentiated adenocarcinoma':6,'poorly differentiated adenocarcinoma':4,'Mucoid adeoncarcinoma':1}
    biopsy_encoded= biopsy_mapping[biopsy]
    TNT_mapping={'Short course':1,'long course':0}
    TNT_encoded= TNT_mapping[tnt_c]
    course_mapping={'Regression':0, 'Stationary':1,'Progression':2}
    course_encoded= course_mapping[course]

    input_data=np.array([[scaled_age,scaled_length,scaled_distance,scaled_dimensions,scaled_quadrants_involved,gender_encoded,stageT_encoded,stageN_encoded,sphincter_encoded,biopsy_encoded,TNT_encoded,course_encoded]])
   
    prediction_encoded = model.predict(input_data)[0]
    if prediction_encoded == 0:
            st.success('Your patient mostly will get Complete pathological response')
    elif prediction_encoded == 1:
            st.error('Unfortunately, Your patient mostly will not get pathological response')
    else:
            st.warning('Your patient mostly will get partial pathological response')
