import numpy as np
from keras.models import load_model
from keras.applications.vgg19 import preprocess_input
from PIL import Image

import cv2

import streamlit as st
import base64

import os
from urllib.request import urlopen
import streamlit as st

classes =['happy','sad']

st.markdown('<h1 style="color:black;">Image classification model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> happy,sad </h3>', unsafe_allow_html=True)

# background image to streamlit

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


#data preparation
#img= cv2.imread(r'D:\unet_vgg\seg_test\seg_test\forest\20056.jpg')
#st.markdown('<h4 style="color:white;">Input image</h4>', unsafe_allow_html=True)
upload= st.file_uploader('Insert image for classification', type=['png','jpg'])


my_email = st.secrets['email']
model_weight_file = st.secrets['model_url']

if not os.path.exists('/best.hdf5'):
    u = urlopen(model_weight_file)
    data = u.read()
    u.close()
    with open('best.hdf5', 'wb') as f:
        f.write(data)



c1, c2= st.columns(2)
if upload is not None:
  im= Image.open(upload)
  img= np.asarray(im)
  image= cv2.resize(img,(224, 224))
  img= preprocess_input(image)
  img= np.expand_dims(img, 0)
  c1.header('Input Image')
  c1.image(im)
  c1.write(img.shape)

  #load weights of the trained model.
  input_shape = (224, 224, 3)
  n_classes=2
  vgg_model = load_model('best.hdf5')

  # prediction on model
  vgg_preds = vgg_model.predict(img)
  vgg_pred_classes = np.argmax(vgg_preds, axis=1)
  c2.header('Output')
  c2.subheader('Predicted class :')
  c2.write(classes[vgg_pred_classes[0]] )