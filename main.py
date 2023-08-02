from pyexpat import features
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from PIL import Image

st.title("Electric Motor Temperature")

if st.sidebar.checkbox("About"):
#    img = Image.open("1.png","2.png","3.png")
    st.image("1.png")
    st.image("2.png")
    st.image("3.png")
    st.image("4.png")
    
if st.sidebar.checkbox("Predict"):
    
    st.sidebar.header("User input parameters")



    def user_input_features():
        ambient = st.sidebar.selectbox('ambient',[-0.752143 , -0.047497] )
 #       ambient = st.sidebar.number_input(label="ambient")
#        st.bar_chart(ambient)
        
        coolant = st.sidebar.selectbox('coolant',[-1.118446,  0.341638] )   
 #       coolant = st.sidebar.number_input(label="coolant")       
         
        u_d     = st.sidebar.selectbox('U_d',   [ 0.327935 , 0.331475] )   
 #       u_d = st.sidebar.number_input(label="Voltage d-component")        
        
        u_q     = st.sidebar.selectbox('U_q', [   -1.297858, -1.246114] )  
 #       u_q = st.sidebar.number_input(label="Voltage q-component")        
                  
        
        
        i_d = st.sidebar.selectbox('I_d',[1.029572 , 1.029142] )
#        i_d = st.sidebar.number_input(label="Current d-component")       
        
        i_q = st.sidebar.selectbox('I_q',[-0.245860 , -0.245723	 ])   
#        i_q = st.sidebar.number_input(label="Current q-component")       

        pm = st.sidebar.selectbox('pm',[-2.522071 , 0.429853 ])   
#        pm = st.sidebar.number_input(label="pm")       
        
    
        data = {	'ambient' : ambient,
			'coolant':coolant,
			'Voltage d-component':u_d,
      		'Voltage q-component':u_q,
			'Current d-component':i_d,
			'Current q-component':i_q,
            'pm':pm,
            }
    
    
        features = pd.DataFrame(data ,index = [0])
        return features


    df = user_input_features()
    st.subheader("User input Parameters")
    st.table(df, )

    chart_data = pd.DataFrame(df.values[0],)
    st.bar_chart(chart_data)

#Loading the top 2 best models as a pickle files for prediction
    with open('file.pkl','rb') as f:
        mp_rf = pickle.load(f)
        
    with open('filee.pkl','rb') as g:
        mp_knn = pickle.load(g) 
        
    status = st.radio(("Select model"),("Decisiontree","K-Neighbors","None"))
    if st.button("Predict"):
        if status == "Decisiontree":
            pred_rf = mp_rf.predict(df)
            st.markdown("### motor speed is")
            st.markdown(pred_rf)
        
        elif status == "K-Neighbors":
            pred_knn = mp_knn.predict(df)
            st.markdown("### motorspeed is")
            st.markdown(pred_knn)
            
        
        else:
            st.warning("Choose atleast one Model")



