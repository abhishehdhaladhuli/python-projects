import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# App title
st.write("""
# Iris Flower Prediction App 🌸  
This app predicts the type of iris flower based on user input features.
""")

# Sidebar title
st.sidebar.header("User Input Features")

# Function to get user input
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 6.8, 1.5)
    petal_width = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)
    
    # Create DataFrame with the correct feature names
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    
    dataset = pd.DataFrame([data])  
    return dataset

# Get user input
df = user_input_features()

# Display user input
st.subheader("User Input Parameters")
st.write(df)

# Load Iris dataset
iris = load_iris()

# Convert `X` to a DataFrame with feature names
X = pd.DataFrame(iris.data, columns=iris.feature_names)  
Y = iris.target

# Train RandomForest model
clf = RandomForestClassifier()
clf.fit(X, Y)

# Make predictions
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Display class labels
st.subheader("Class Labels and their corresponding index number")
st.write(iris.target_names)

# Display prediction
st.subheader("Prediction")
st.write(f"Predicted Flower Type: **{iris.target_names[prediction[0]]}**")

# Display prediction probabilities
st.subheader("Prediction Probability")
st.write(prediction_proba)
