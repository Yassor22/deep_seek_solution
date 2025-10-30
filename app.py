import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb

# Load the improved model
@st.cache_resource
def load_model():
    model = jb.load('improved_model.pkl')
    scaler = jb.load('improved_scaler.pkl')
    encodings = jb.load('encodings.pkl')
    return model, scaler, encodings

model, scaler, encodings = load_model()

# Your existing UI code remains the same...
st.sidebar.header('This is a predictive model for locally advanced Rectal cancer to TNT')
st.sidebar.image('https://tse4.mm.bing.net/th/id/OIP.j2TJw0vapJPgHsqrhGYUHgHaHa?pid=ImgDet&w=185&h=185&c=7&dpr=1.1&o=7&rm=3')
st.sidebar.write('This application has been built to predict the response of locally advanced Rectal cancer to the types of total neoadjuvant therapy')
st.sidebar.write('The used model is Improved RandomForestClassifier with balanced classes')
st.sidebar.write('Limitation : small-sized data\n If you can share deidentified data please Contact me')
st.sidebar.write('Created by ')
st.sidebar.markdown(' "Yasser Ali Okasha"')
st.sidebar.write('Supervised by ')
st.sidebar.markdown('"Professor.khaled Madbouly"')
st.sidebar.write(' Assisted By " Dr Shahed"')
st.sidebar.write('Contact details ')
st.sidebar.write("Email: yasser.okasha@alexmed.edu.eg")

st.title('Prediction of locally advanced rectal cancer response to TNT')
a1, a2, a3 = st.columns(3)
a1.image('cancer.JPG')
a2.image('radiotherapy.JPG')
a2.image('Capture.JPG')
a3.image('surgery.JPG')

st.text('Please Fill the following parameters about your patient to predict the Response')
st.write('Demographic Data')
gender = st.selectbox('Gender', ['Male', 'Female'])
Age = st.slider('Age', 10, 108)
st.write('DRE')
length = st.number_input('Anal Canal length in CM', min_value=0.0, value=3.0)
distance = st.number_input('Distance from anal verge in CM', min_value=0.0, value=3.0)
quadrants_involved = st.slider('Quadrants involved', 0, 4, 2)
antorp = st.selectbox('Site', ['Anterior', 'posterior', 'lateral', 'All'])
invasion = st.selectbox('Invasion of surrounding structures', ['Yes', 'No'])

st.write('Pretreatment MRI findings')
stageT = st.selectbox('T stage', ['T1', 'T2', 'T3a', 'T3b', 'T3c', 'T4a', 'T4b'])
StagN = st.selectbox('N stage', ['N0', 'N1', 'N2a', 'N2b', 'N2c', 'N3a', 'N3b', 'N3c'])
dimensions = st.number_input('Tumour dimensions', min_value=0.0, value=2.0)
sphincter = st.selectbox('Sphincters involvement', ['Not involved', 'Involved'])

st.write('Colonoscopic Biopsy')
biopsy = st.selectbox('Histopathological result of the Biopsy', 
                      ['Well differentiated adenocarcinoma', 
                       'Moderately differentiated adenocarcinoma', 
                       'Poorly differentiated adenocarcinoma',
                       'Mucoid adenocarcinoma'])

st.write('TNT details')
tnt_c = st.selectbox('TNT Radiation', ['Short course', 'Long course'])
tnt = st.selectbox('TNT Chemotherapy', ['induction', 'Consolidation'])

st.write('Post-treatment MRI findings')
course = st.selectbox('Radiological Response', ['Regression', 'Stationary', 'Progression'])

btn = st.button('Submit')

if btn:
    try:
        # Apply consistent encodings
        gender_encoded = encodings['gender'][gender.lower()]
        stageT_encoded = encodings['stageT'][stageT.lower()]
        stageN_encoded = encodings['stageN'][StagN.lower()]
        sphincter_encoded = encodings['sphincter'][sphincter.lower()]
        
        biopsy_lower = biopsy.lower()
        if 'well' in biopsy_lower:
            biopsy_encoded = 2
        elif 'moderate' in biopsy_lower:
            biopsy_encoded = 1
        else:  # poor or mucoid
            biopsy_encoded = 0
            
        TNT_encoded = encodings['TNT'][tnt_c.split()[0].lower()]
        course_encoded = encodings['Course'][course.lower()]

        # Create input array (using only the features we trained on)
        input_data = np.array([[
            Age, gender_encoded, distance, quadrants_involved, length,
            stageT_encoded, stageN_encoded, sphincter_encoded, dimensions,
            biopsy_encoded, TNT_encoded, course_encoded
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Get prediction and probabilities
        prediction_encoded = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        # Debug information
        with st.expander("Debug Information"):
            st.write(f"Input shape: {input_data.shape}")
            st.write(f"Prediction probabilities: {probabilities}")
            st.write(f"Class probabilities: No Response: {probabilities[0]:.2f}, Partial: {probabilities[1]:.2f}, Complete: {probabilities[2]:.2f}")
            st.write(f"Raw prediction: {prediction_encoded}")
        
        # Display result
        st.write("---")
        st.write("## 🎯 Prediction Result:")
        
        if prediction_encoded == 2:
            st.success('✅ **Your patient mostly will get Complete pathological response**')
            st.write(f"Probability: {probabilities[2]:.1%}")
        elif prediction_encoded == 1:
            st.warning('⚠️ **Your patient mostly will get Partial pathological response**')
            st.write(f"Probability: {probabilities[1]:.1%}")
        else:
            st.error('❌ **Unfortunately, Your patient mostly will not get pathological response**')
            st.write(f"Probability: {probabilities[0]:.1%}")
            
        # Show confidence
        max_prob = max(probabilities)
        if max_prob > 0.7:
            st.info("🔄 **High confidence prediction**")
        elif max_prob > 0.5:
            st.info("🔄 **Moderate confidence prediction**")
        else:
            st.info("🔄 **Low confidence prediction - consider clinical judgment**")
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
