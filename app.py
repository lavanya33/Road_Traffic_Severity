import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from prediction import get_prediction, ordinal_encoder

model = joblib.load(r'Model/extree_tuned_classifier.joblib')

st.set_page_config(page_title="Accident Severity Prediction App",
                   page_icon="🚧", layout="wide")


#creating option list for dropdown menu
options_day = ['Sunday', "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
options_age = ['18-30', '31-50', 'Over 51', 'Unknown', 'Under 18']

options_acc_area = ['Other', 'Office areas', 'Residential areas', ' Church areas',
       ' Industrial areas', 'School areas', '  Recreational areas',
       ' Outside rural areas', ' Hospital areas', '  Market areas',
       'Rural village areas', 'Unknown', 'Rural village areasOffice areas',
       'Recreational areas']
       
options_cause = ['No distancing', 'Changing lane to the right',
       'Changing lane to the left', 'Driving carelessly',
       'No priority to vehicle', 'Moving Backward',
       'No priority to pedestrian', 'Other', 'Overtaking',
       'Driving under the influence of drugs', 'Driving to the left',
       'Getting off the vehicle improperly', 'Driving at high speed',
       'Overturning', 'Turnover', 'Overspeed', 'Overloading', 'Drunk driving',
       'Unknown', 'Improper parking']
options_vehicle_type = ['Automobile', 'Lorry (41-100Q)', 'Other', 'Pick up upto 10Q',
       'Public (12 seats)', 'Stationwagen', 'Lorry (11-40Q)',
       'Public (13-45 seats)', 'Public (> 45 seats)', 'Long lorry', 'Taxi',
       'Motorcycle', 'Special vehicle', 'Ridden horse', 'Turbo', 'Bajaj', 'Bicycle']
options_driver_exp = ['5-10yr', '2-5yr', 'Above 10yr', '1-2yr', 'Below 1yr', 'No Licence', 'unknown']
options_lanes = ['Two-way (divided with broken lines road marking)', 'Undivided Two way',
       'other', 'Double carriageway (median)', 'One way',
       'Two-way (divided with solid lines road marking)', 'Unknown']
options_light_conditions=['Daylight','Darkness - lights lit','Darkness - no lighting','Darkness - lights unlit']
options_type_of_collision=['Vehicle with vehicle collision','Collision with roadside objects','Collision with pedestrians','Rollover','Collision with animals','Collision with roadside-parked vehicles','Fall from vehicles','Other','Unknown','With Train']

features = ['hour','Day_of_week','Number_of_casualties','Cause_of_accident','Number_of_vehicles_involved','Type_of_vehicle','Light_conditions','Type_of_collision','Age_band_of_driver','Driving_experience','Area_accident_occured','Lanes_or_Medians']
            


st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        
        hour = st.slider("Pickup Hour: ", 0, 23, value=0, format="%d")
        Day_of_week = st.selectbox("Select Day of Week: ", options=options_day)
        Number_of_casualties = st.slider("Number of Casualties: ", 1, 8, value=0, format="%d") 
        Cause_of_accident = st.selectbox("Select Accident Cause: ", options=options_cause) 
        Number_of_vehicles_involved = st.slider("No.of Vehicles involved: ", 1, 7, value=0, format="%d")
        Type_of_vehicle = st.selectbox("Select Vehicle Type: ", options=options_vehicle_type)
        Light_conditions = st.selectbox("Select Light Conditions: ", options=options_light_conditions)
        Type_of_collision = st.selectbox("Select Type of Collision: ", options=options_type_of_collision)
        
        Age_band_of_driver = st.selectbox("Select Driver Age: ", options=options_age)
        Driving_experience = st.selectbox("Select Driving Experience: ", options=options_driver_exp)
        Area_accident_occured = st.selectbox("Select Accident Area: ", options=options_acc_area)
        Lanes_or_Medians = st.selectbox("Select Lanes: ", options=options_lanes)
        
       
        
        
        submit = st.form_submit_button("Predict")


    if submit:
        Day_of_week = ordinal_encoder(Day_of_week, options_day)
        Cause_of_accident = ordinal_encoder(Cause_of_accident, options_cause)
        Type_of_vehicle = ordinal_encoder(Type_of_vehicle, options_vehicle_type)
        Light_conditions = ordinal_encoder(Light_conditions, options_light_conditions)
        Type_of_collision = ordinal_encoder(Type_of_collision, options_type_of_collision)
        Age_band_of_driver = ordinal_encoder(Age_band_of_driver, options_age)
        Driving_experience = ordinal_encoder(Driving_experience, options_driver_exp)
        Area_accident_occured = ordinal_encoder(Area_accident_occured, options_acc_area)
        Lanes_or_Medians = ordinal_encoder(Lanes_or_Medians, options_lanes)


        data = np.array([hour,Day_of_week,Number_of_casualties,Cause_of_accident,Number_of_vehicles_involved,Type_of_vehicle,Light_conditions,Type_of_collision,Age_band_of_driver,Driving_experience,Area_accident_occured,Lanes_or_Medians]).reshape(1,-1)
        
        
        pred = get_prediction(data=data, model=model)

        st.write(f"The predicted severity is:  {pred[0]}")

if __name__ == '__main__':
    main()