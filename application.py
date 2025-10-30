import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb
from sklearn.preprocessing import StandardScaler

# Load your augmented data to understand the structure
df = pd.read_csv('augmented_rectal_cancer_data.csv')

st.sidebar.header('This is a predictive model for locally advanced Rectal cancer to TNT')
st.sidebar.image('https://tse4.mm.bing.net/th/id/OIP.j2TJw0vapJPgHsqrhGYUfgHaHa?pid=ImgDet&w=185&h=185&c=7&dpr=1.1&o=7&rm=3')
st.sidebar.write('This application has been built to predict the response of locally advanced Rectal cancer to the types of total neoadjuvant therapy')
st.sidebar.write('The used model is RandomForestClassifier model with accuracy_score 78')
st.sidebar.write('Limitation : small-sized data\n If you can share deidentified data please Contact me')
st.sidebar.write('Created by ')
st.sidebar.markdown(' "Yasser Ali Okasha"')
st.sidebar.write('Supervised by ')
st.sidebar.markdown('"professor.khaled Madbouly"')
st.sidebar.write('Contact details ')
st.sidebar.write("Email: yasser.okasha@alexmed.edu.eg")

st.title('Prediction of locally advanced rectal cancer response to TNT')
a1, a2, a3 = st.columns(3)
a1.image('cancer.JPG')
a2.image('radiotherapy.JPG')
a3.image('surgery.JPG')

st.text('Please Fill the following parameters about your patient to predict the Response')

# Input fields that match your augmented data
st.write('Demographic Data')
gender = st.selectbox('Gender', ['Male', 'Female'])
Age = st.slider('Age', 10, 108, 50)

st.write('DRE')
distance = st.number_input('Distance from anal verge in CM', min_value=0.0, value=3.0, step=0.1)
quadrants_involved = st.slider('Quadrants involved', 1, 4, 2)
length = st.number_input('Anal Canal length in CM', min_value=0.0, value=3.0, step=0.1)

st.write('Pretreatment MRI findings')
stageT = st.selectbox('T stage', ['T1', 'T2', 'T3a', 'T3b', 'T3c', 'T4a', 'T4b'])
StagN = st.selectbox('N stage', ['N0', 'N1', 'N2a', 'N2b', 'N2c', 'N3a', 'N3b', 'N3c'])
dimensions = st.number_input('Tumour dimensions', min_value=0.0, value=2.0, step=0.1)
sphincter = st.selectbox('Sphincters involvement', ['No', 'Yes'])

st.write('Colonoscopic Biopsy')
biopsy = st.selectbox('Histopathological result of the Biopsy', [
    'Well differentiated adenocarcinoma',
    'Moderately differentiated adenocarcinoma',
    'poorly differentiated adenocarcinoma',
    'Mucoid adeoncarcinoma'
])

st.write('TNT details')
tnt_c = st.selectbox('TNT Radiation', ['Short course', 'Long course'])

st.write('Post-treatment MRI findings')
course = st.selectbox('Radiological Response', ['Regression', 'Stationary', 'Progression'])

btn = st.button('Submit')

if btn:
    try:
        # Load model and scaler
        model = jb.load('svc_model.pkl')
        scaler = jb.load('Scaler.pkl')
        
        # Apply mappings (consistent with your augmented data)
        gender_mapping = {'Female': 0, 'Male': 1}
        gender_encoded = gender_mapping[gender]
        
        stageT_mapping = {'T1': 1, 'T2': 2, 'T3a': 3, 'T3b': 4, 'T3c': 5, 'T4a': 6, 'T4b': 7}
        stageT_encoded = stageT_mapping[stageT]
        
        stageN_mapping = {'N0': 0, 'N1': 1, 'N2a': 2, 'N2b': 3, 'N2c': 4, 'N3a': 5, 'N3b': 6, 'N3c': 7}
        stageN_encoded = stageN_mapping[StagN]
        
        sphincter_mapping = {'No': 0, 'Yes': 1}
        sphincter_encoded = sphincter_mapping[sphincter]
        
        biopsy_mapping = {
            'Well differentiated adenocarcinoma': 2,
            'Moderately differentiated adenocarcinoma': 1,
            'poorly differentiated adenocarcinoma': 0,
            'Mucoid adeoncarcinoma': 0
        }
        biopsy_encoded = biopsy_mapping[biopsy]
        
        TNT_mapping = {'Short course': 0, 'Long course': 1}
        TNT_encoded = TNT_mapping[tnt_c]
        
        course_mapping = {'Regression': 0, 'Stationary': 1, 'Progression': 2}
        course_encoded = course_mapping[course]

        # Create input array in the correct order
        input_data = np.array([[
            Age, gender_encoded, distance, quadrants_involved, length,
            stageT_encoded, stageN_encoded, sphincter_encoded, dimensions,
            biopsy_encoded, TNT_encoded, course_encoded
        ]])
        
        st.write(f"🔍 Input data shape: {input_data.shape}")
        st.write(f"📊 Sample values: {input_data[0]}")
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction_encoded = model.predict(input_scaled)[0]
        
        # Display result
        st.write("---")
        st.write("## 🎯 Prediction Result:")
        
        if prediction_encoded == 2:
            st.success('✅ Your patient mostly will get Complete pathological response')
        elif prediction_encoded == 1:
            st.warning('⚠️ Your patient mostly will get Partial pathological response')
        else:
            st.error('❌ Unfortunately, Your patient mostly will not get pathological response')
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.write("Please make sure:")
        st.write("- Your model files (svc_model.pkl, Scaler.pkl) are in the correct directory")
        st.write("- The model was trained with the same features as in this app")
