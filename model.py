import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib as jb

# Load your original data
df = pd.read_excel('master_sheet for machine learning.xlsx')

# Data cleaning and standardization
def clean_and_prepare_data(df):
    # Standardize column names
    df.columns = ['Age', 'gender', 'distance', 'Quadrants', 'Anal_canal', 'stageT', 'stageN', 
                  'sphincter', 'dimensions', 'Biopsy', 'TNT', 'Course', 'Response']
    
    # Clean and standardize values
    df['gender'] = df['gender'].str.strip().str.lower()
    df['sphincter'] = df['sphincter'].str.strip().str.lower()
    df['Biopsy'] = df['Biopsy'].str.strip().str.lower()
    df['TNT'] = df['TNT'].str.strip().str.lower()
    df['Course'] = df['Course'].str.strip().str.lower()
    df['Response'] = df['Response'].str.strip().str.lower()
    
    # Fix sphincter values
    df['sphincter'] = df['sphincter'].replace({'not': 'not involved', 'involved': 'involved'})
    
    return df

df_clean = clean_and_prepare_data(df)

# Create consistent encoding mappings
def create_consistent_encodings():
    gender_mapping = {'female': 0, 'male': 1}
    sphincter_mapping = {'not involved': 0, 'involved': 1}
    
    # T stage mapping (consistent with medical staging)
    stageT_mapping = {'1': 1, '2': 2, '3a': 3, '3b': 4, '3c': 5, '4a': 6, '4b': 7}
    
    # N stage mapping
    stageN_mapping = {'0': 0, '1': 1, '2a': 2, '2b': 3, '2c': 4, '3a': 5, '3b': 6, '3c': 7}
    
    biopsy_mapping = {
        'well differentiated adenocarcinoma': 2,
        'moderately differentiated adenocarcinoma': 1,
        'poorly differentiated adenocarcinoma': 0,
        'mucoid carcinoma with signet ring': 0,
        'invasive adenocarcinoma': 1
    }
    
    TNT_mapping = {'short': 0, 'long': 1}
    course_mapping = {'regressive': 0, 'stationary': 1, 'progressive': 2}
    response_mapping = {'complete': 2, 'partial': 1, 'no response': 0}
    
    return {
        'gender': gender_mapping,
        'sphincter': sphincter_mapping,
        'stageT': stageT_mapping,
        'stageN': stageN_mapping,
        'Biopsy': biopsy_mapping,
        'TNT': TNT_mapping,
        'Course': course_mapping,
        'Response': response_mapping
    }

encodings = create_consistent_encodings()

# Apply encodings to cleaned data
def encode_data(df, encodings):
    df_encoded = df.copy()
    
    for column, mapping in encodings.items():
        if column in ['Response']:  # Target variable
            continue
        df_encoded[column] = df_encoded[column].map(mapping)
    
    # Encode target separately
    df_encoded['Response_encoded'] = df_encoded['Response'].map(encodings['Response'])
    
    return df_encoded

df_encoded = encode_data(df_clean, encodings)

# Data augmentation function
def augment_data(df, augmentation_factor=5):
    augmented_data = [df]
    
    for i in range(augmentation_factor):
        # Create slightly modified copies
        df_aug = df.copy()
        
        # Add small random noise to numerical features
        numerical_cols = ['Age', 'distance', 'Quadrants', 'Anal_canal', 'dimensions']
        for col in numerical_cols:
            noise = np.random.normal(0, 0.1, size=len(df_aug))  # 10% noise
            df_aug[col] = df_aug[col] * (1 + noise)
            # Ensure values stay in reasonable ranges
            if col == 'Age':
                df_aug[col] = df_aug[col].clip(18, 80)
            elif col in ['distance', 'Anal_canal', 'dimensions']:
                df_aug[col] = df_aug[col].clip(0.5, 10)
            elif col == 'Quadrants':
                df_aug[col] = df_aug[col].round().clip(1, 4)
        
        # Occasionally swap categorical values with similar ones
        if np.random.random() > 0.7:
            swap_indices = np.random.choice(len(df_aug), size=len(df_aug)//3, replace=False)
            for idx in swap_indices:
                # Swap between similar biopsy types
                current_biopsy = df_aug.iloc[idx]['Biopsy']
                if current_biopsy in [0, 1]:  # poor/moderate
                    df_aug.iloc[idx, df_aug.columns.get_loc('Biopsy')] = 1 - current_biopsy
                # Occasionally change TNT course
                if np.random.random() > 0.8:
                    df_aug.iloc[idx, df_aug.columns.get_loc('TNT')] = 1 - df_aug.iloc[idx]['TNT']
        
        augmented_data.append(df_aug)
    
    return pd.concat(augmented_data, ignore_index=True)

# Augment the data
print(f"Original data size: {len(df_encoded)}")
df_augmented = augment_data(df_encoded, augmentation_factor=10)
print(f"Augmented data size: {len(df_augmented)}")

# Prepare features and target
feature_columns = ['Age', 'gender', 'distance', 'Quadrants', 'Anal_canal', 'stageT', 'stageN', 
                   'sphincter', 'dimensions', 'Biopsy', 'TNT', 'Course']

X = df_augmented[feature_columns]
y = df_augmented['Response_encoded']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with balanced class weights
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=3,
    class_weight='balanced',  # Important for imbalanced data
    random_state=42
)

model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Response', 'Partial', 'Complete']))

print("\nFeature Importances:")
for feature, importance in zip(feature_columns, model.feature_importances_):
    print(f"{feature}: {importance:.3f}")

# Save the model and scaler
jb.dump(model, 'improved_model.pkl')
jb.dump(scaler, 'improved_scaler.pkl')
jb.dump(encodings, 'encodings.pkl')

print("\n✅ Model and scaler saved successfully!")
