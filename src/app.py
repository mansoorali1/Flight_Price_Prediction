import pandas as pd
import streamlit as st
from Preprocessing import date_splitter, tod_departure, stops
import pickle
import base64

def set_background(png_file):
    """Sets the background image of the Streamlit app."""
    with open(png_file, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode()

    # Create CSS to set the background image
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_image}");
        background-size: cover;  /* Ensure the image covers the entire screen */
        background-repeat: no-repeat;  /* Prevent the image from repeating */
        background-attachment: fixed;  /* Fix the image position when scrolling */
        background-position: center;  /* Center the image */
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def get_user_input():
    """Collects user input through Streamlit widgets and returns a DataFrame."""
    Airline = st.selectbox('**Select Airline**',(
        'IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
        'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
        'Vistara Premium economy', 'Jet Airways Business',
        'Multiple carriers Premium economy', 'Trujet'))

    Date_of_Journey = st.date_input("**Date of Journey**", format='DD-MM-YYYY')
    Date_of_Journey = str(Date_of_Journey.strftime('%d/%m/%Y'))
    
    Source = st.selectbox('**From**',(
        'Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'))
    
    Destination = st.selectbox('**To**',(
        'New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'))
    
    Dep_Time = st.time_input("**Departure Time**")
    Dep_Time = str(Dep_Time.strftime('%H:%M'))
    
    Total_Stops = st.selectbox('**Stops**',(
        'non-stop', '1 stop', '2 stops', '3 stops', '4 stops'))

    
    data = [Airline, Date_of_Journey, Source, Destination, Dep_Time, Total_Stops]
    df = pd.DataFrame(pd.Series(data, index=[
        'Airline', 'Date_of_Journey', 'Source', 'Destination',
        'Dep_Time', 'Total_Stops'])).transpose()
    return df

def display_gif(gif_file):
    """Displays a GIF below the user input section."""
    gif_placeholder = st.empty()
    with open(gif_file, "rb") as f:
        encoded_gif = base64.b64encode(f.read()).decode()
    
    gif_html = """
    <div style="display: flex; justify-content: center; margin-top: -15px;">
        <img src='data:image/gif;base64,{encoded_gif}' style="width: 200px; opacity: 0.8;"/>
    </div>
    """
    gif_placeholder.markdown(gif_html.format(encoded_gif=encoded_gif), unsafe_allow_html=True)
    return gif_placeholder

def load_model_and_pipeline():
    """Loads the preprocessing pipeline and model from pickle files."""
    with open(r'..//bin/pipe.pkl', 'rb') as f1:
        pipe = pickle.load(f1)
    with open(r'..//bin/model.pkl', 'rb') as f2:
        model = pickle.load(f2)
    return pipe, model

def predict_price(pipe, model, df):
    """Predicts the flight price using the pre-trained model and displays the result."""
    input_data = pipe.transform(df)
    y = model.predict(input_data)[0]
    return round(y)

if __name__ == "__main__":
    st.title("Flight Price Prediction")
    set_background("..//images/flight_back.png")
    user_input_df = get_user_input()
    gif_placeholder = display_gif("..//images/flight_rotate.gif")
    pipe, model = load_model_and_pipeline()

    if st.button("**Predict**"):
        predicted_price = predict_price(pipe, model, user_input_df)
        gif_placeholder.empty()  # Clear the GIF
        st.markdown(f"**The estimated price for the flight is around {predicted_price}**")