import streamlit as st
# st.title('hello world')

import numpy as np  # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import joblib
#Load the model to disk
model = joblib.load("model2.sav")

from PIL import Image

#Import python scripts
from preprocessing import preprocess

def main():
    # Setting Application title
    st.title('Telecom Churn Prediction App')

    # Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a ficitional telecom company use case. 
     The application is functional for batch data prediction.. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Setting Application sidebar default
    image = Image.open('PyCoders.jpeg')
    # add_selectbox = st.sidebar.selectbox(
    #     "How would you like to predict?", ("Online","Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

st.subheader("Dataset upload")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Get overview of data
    st.write(data.head())
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    # Preprocess inputs
    preprocess_df = preprocess(data)
    if st.button('Predict'):
        # Get batch prediction
        prediction = model.predict(preprocess_df)
        prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
        prediction_df = prediction_df.replace({1: 'Yes, the customer will terminate the Company.',
                                               0: 'No, the customer is happy with telecom Company.'})

        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.subheader('Prediction')
        st.write(prediction_df)


if __name__ == '__main__':
    main()
