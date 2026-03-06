import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("../src/lstm_model.h5")

st.title("Stock Price Prediction")

st.write("AI predicts future stock prices using LSTM")

input_data = np.random.rand(1,60,1)

prediction = model.predict(input_data)

st.write("Predicted Price:", prediction[0][0])
