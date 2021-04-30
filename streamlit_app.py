import streamlit as st
import pandas as pd
import tensorflow as tf

header = st.beta_container()
dataset = st.beta_container()
model_training = st.beta_container()
 

with header:
    st.title('Google fit App')
    st.markdown('''
        * In this project I looked into the dataset of my Google Fit App.
        * I use this app everyday to trace my Calories.
        * I like to know How long I walked? What was my Heart beating rate? 
        * So I downloaded my Google fit dataset and used it to check how many Calories did I burnt?
    ''')


with dataset:
    st.header("Google Fit dataset")
    st.markdown('''
        * To train my model I used dataset from kaggle.com
        * But for the testing I used my own dataset from google fit App and used it.''')
    dataset = pd.read_csv('data/GoogleFit.csv')
    st.write(dataset.sample(10))

    # most heighst heart points
    st.subheader('Heart beating point distribution of first 20 heart points')
    heart_points = pd.DataFrame(dataset['Heart_point'].value_counts()).head(20)
    st.bar_chart(heart_points)



with model_training:
    st.header("Model_training")
    st.markdown('''
    * I used Artificial Neural Network to train my model.
    * Model have five hidden layers, with Adamax optimizer and mean_squared_error loss.
    * Model was over-fitting so I used **Regularization** with L1 0.001
    ''')

    # Creating coulmns

    heart_point = st.slider('Select the number of Heart points', min_value=1, max_value=150, value=5, step=2)
    heart_minute = st.slider('Select the number of Heart minutes', min_value=1, max_value=150, value=5, step=2)
    step_count = st.selectbox('select the Number of Steps', options=[500, 1000, 1500, 2000, 2500, 3000])
    distance = st.slider('Select the Distance (meter)', min_value=100, max_value=100000, value=1000, step=500)
    average_speed = st.slider('Your Average speed (meter/second)', min_value=0.1, max_value=10.0, value=0.1, step=0.5)
    move = st.slider('Number of Moves per minute', min_value=5, max_value=300, value=10, step=10)
    walking_duration = st.slider('How long did you walked (min-sec)', min_value=10, max_value=500, value=50, step=10)

    features = [[heart_point, heart_minute, step_count, distance, average_speed, move, walking_duration]]
    # loding the model
    loded_model = tf.keras.models.load_model('my_model_0.9.h5')

    st.subheader('Calories burnt (kCal):')
    st.subheader('\n')
    st.write(
        loded_model.predict(features)
    )
