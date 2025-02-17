import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuration
st.set_page_config(page_title="Forest Fire Prediction", layout="wide")

# Title of the app
st.title("Forest Fire Prediction using Random Forest")

# Embed ArcGIS map using iframe HTML
st.markdown("""
    <iframe src="https://firemap.live/" 
            width="100%" height="600px">
    </iframe>
""", unsafe_allow_html=True)

# Define dataset URL from GitHub
dataset_url = "https://raw.githubusercontent.com/SiddharthaSomalinga/FireShield/refs/heads/main/dataset.csv"

# Function to download and cache dataset
@st.cache_data
def download_dataset(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        return pd.read_csv(StringIO(response.text))
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download dataset: {str(e)}")
        st.stop()

# Load dataset
df = download_dataset(dataset_url)

# Display the dataset preview
st.subheader("Dataset Preview")

# Ensure 'Year' column is displayed as integers if it exists
if "Year" in df.columns:
    df["Year"] = df["Year"].astype(int)

st.dataframe(df.head())

# Strip extra spaces from column names
df.columns = df.columns.str.strip()

# Select the features to use for prediction
features = ['Temperature', 'RH (Relative Humidity)', 'WS (Wind Speed)', 'Rain']

# Ensure all the selected features exist in the dataset
for feature in features:
    if feature not in df.columns:
        st.error(f"Missing required feature: {feature}")
        st.stop()

# Automatically set target column to "Result"
target_column = "Result"
if target_column not in df.columns:
    st.error(f"Target column '{target_column}' not found in the dataset.")
    st.stop()

# Clean and process the target column
df[target_column] = df[target_column].astype(str).str.strip().str.lower()  # Standardize labels
df = df.dropna(subset=[target_column])  # Drop rows with NaN values in the target column

# Convert all numeric columns
df = df.apply(lambda x: pd.to_numeric(x, errors="coerce") if x.name not in [target_column] else x)

# Handle missing values
numeric_cols = df[features + [target_column]].select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].apply(lambda col: col.fillna(col.mean()))

# Define feature and target variables
X = df[features]
y = df[target_column]

# Split dataset
test_size = st.sidebar.slider("Test Size (Fraction)", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Standardize feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize model_trained flag and rf_model in session_state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None

# Train Random Forest Classifier
n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 50, 500, 100)
rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

if st.sidebar.button("Train Model"):
    rf_model.fit(X_train, y_train)
    st.session_state.rf_model = rf_model  # Save the trained model in session state
    st.session_state.model_trained = True  # Set the model_trained flag to True
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    st.subheader("Model Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")

    # Feature Importance
    st.subheader("Feature Importance")
    feature_importance = rf_model.feature_importances_
    importance_df = pd.DataFrame({"Feature": features, "Importance": feature_importance}).sort_values(
        by="Importance", ascending=False)

    # Display Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", ax=ax)
    ax.set_title("Feature Importance in Forest Fire Prediction")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

# Input form for prediction
st.header("Predict Forest Fire")
st.subheader("Enter Feature Values for Prediction")
user_input = {}

# Create number input fields for each feature on the main page
for feature in features:
    user_input[feature] = st.number_input(f"Enter {feature}", value=float(X[feature].mean()))

# Prediction button on the main page
if st.button("Predict"):
    if not st.session_state.model_trained:
        st.error("The model is not trained yet. Please train the model before making predictions.")
    else:
        # Retrieve the trained model
        rf_model = st.session_state.rf_model

        # Create a DataFrame for the user input
        input_df = pd.DataFrame([user_input])

        # Check if there are any missing values and handle them
        input_df = input_df.apply(lambda x: x.fillna(x.mean()) if x.name != target_column else x)

        # Apply the same scaling transformation as used on the training data
        input_scaled = scaler.transform(input_df)

        try:
            # Get prediction probabilities
            probas = rf_model.predict_proba(input_scaled)

            # Get class labels and their order
            class_labels = rf_model.classes_

            # Combine probabilities for all "fire" variations
            prob_fire = sum(probas[0][i] for i, label in enumerate(class_labels)
                            if label.strip() == "fire")

            # Combine probabilities for all "not fire" variations
            prob_no_fire = sum(probas[0][i] for i, label in enumerate(class_labels)
                               if label.strip() == "not fire")

            # Display prediction percentages
            fire_percentage = prob_fire * 100
            no_fire_percentage = prob_no_fire * 100

            # Map prediction to fire/not fire based on the class with the higher probability
            fire_prediction = "fire" if prob_fire > prob_no_fire else "not fire"

            # Display the prediction and probabilities
            st.subheader("Prediction Results")
            st.write(f"Prediction: **{fire_prediction}**")
            st.write(f"**Fire**: {fire_percentage:.2f}%")
            st.write(f"**Not Fire**: {no_fire_percentage:.2f}%")

        except NotFittedError:
            st.error("The model is not trained yet. Please train the model before making predictions.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            import traceback
            st.write("Traceback:", traceback.format_exc())
