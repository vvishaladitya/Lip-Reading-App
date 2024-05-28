# Import all of the dependencies
import streamlit as st
import os 
import imageio 

import tensorflow as tf 
from util import load_data, num_to_char
from modelutil import load_model

# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar: 
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Lip Reader')
    st.info('ML application used to read lips') 

st.title('LipReader Application') 

options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose a video', options)


col1, col2 = st.columns(2)

if options: 
    
    with col1: 
        st.info('Input Video')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        
        video, annotations = load_data(tf.convert_to_tensor(file_path))
       

        st.info('Tokens Generated')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Output')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        
