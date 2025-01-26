import streamlit as st
import pickle
from PIL import Image
import matplotlib.pyplot as plt

st.title("Malaria Prediction in Nigeria")
image = Image.open('Medicine-Higher-Life-Foundation.jpg')
st.image(image, caption='Malaria Prediction', use_column_width=True)

st.markdown("""
<style>
.sidebar .sidebar-content {
    background-color: #f5f5f5;
    padding: 20px;
    font-family: Arial, sans-serif;
}
.sidebar .sidebar-header {
    font-size: 1.5em;
    color: #333;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Input Parameters")
    st.markdown("""
    **Pregnancies:** The number of times the individual has been pregnant.
    
    **Treated Net:** Availability of treated nets (Yes/No).
    
    **Environment Sanitization:** Is the environment sanitized (Yes/No)?
    
    **Laboratory Equipments:** Adequate or not?
    
    **Location:** Urban or Rural?
    
    **Additional Inputs:** Includes numerical values for various parameters.
    """)
    
    st.write("**Developed by:** Temitope Atoyebi")

# Load the models
pickle_in = open('Multinomial2.pkl', 'rb')
classifier = pickle.load(pickle_in)

pickle_in2 = open('gradepred.pkl', 'rb')
regressor = pickle.load(pickle_in2)

Name = st.text_input("Name")
Pregnancies = st.number_input("The amount of times the individual got pregnant:")
Availaibility_of_treated_net = st.selectbox("Were treated nets available?", ['No', 'Yes'])
Season_Level_of_Rainfall_Stagnant_water_breeding = st.number_input("The level of rainfall")
Malaria_Parasite_Density_Fever_Rapid_Diagnostic_TestStrip = st.number_input("The malaria Parasite Density fever rapid diagnostic")
Complaints_Symptoms = st.number_input("The complaints or symptoms of the individual")
Age = st.number_input("The age of the individual")
Electricity = st.selectbox('Was there availability of electricity?', ['No', 'Yes'])
Environment_Sanitised_or_not = st.selectbox("Is the environment sanitized or not", ['Yes', 'No'])
Doctor_to_Patient = st.selectbox("Doctor to patient", ['High', 'Low'])
Laboratory_Equipments = st.selectbox("Laboratory Equipments", ['Not Adequate', 'Adequate'])
Location = st.selectbox("Location (Urban or Rural)", ['Urban', 'Rural'])
submit = st.button("Predict")

if submit:
    classification_input_data = [
        Pregnancies, Availaibility_of_treated_net, Season_Level_of_Rainfall_Stagnant_water_breeding,
        Malaria_Parasite_Density_Fever_Rapid_Diagnostic_TestStrip, Complaints_Symptoms, Age, Electricity,
        Environment_Sanitised_or_not, Location, Laboratory_Equipments, Doctor_to_Patient
    ]

    classification_input_data[1] = 1 if Availaibility_of_treated_net == 'Yes' else 0
    classification_input_data[6] = 1 if Electricity == 'Yes' else 0
    classification_input_data[7] = 1 if Environment_Sanitised_or_not == 'Yes' else 0
    classification_input_data[8] = 1 if Location == 'Urban' else 0
    classification_input_data[9] = 1 if Laboratory_Equipments == 'Adequate' else 0
    classification_input_data[10] = 1 if Doctor_to_Patient == 'High' else 0

    Doctor_to_Patient_Low = 1 - classification_input_data[10]  
    classification_input_data.append(Doctor_to_Patient_Low)

    prediction = classifier.predict([classification_input_data])

    if prediction == 0:
        st.write(f'Congratulations, {Name}, you do not have malaria.')
        st.write("Hence the Lab Diagnosis states that the malaria you have is Uncomplicated, indicating it's less than 70%.")

        fig, ax = plt.subplots()
        ax.bar(["Malaria Negative", "Malaria Positive"], [1, 0], color=['#4CAF50', '#F44336'])
        ax.set_ylabel('Prediction')
        ax.set_title('Malaria Prediction Result')
        st.pyplot(fig)

    else:
        st.write(f'{Name}, we are sorry to inform you that you might have malaria.')

        regression_input_data = [
            Pregnancies, Availaibility_of_treated_net, Season_Level_of_Rainfall_Stagnant_water_breeding,
            Malaria_Parasite_Density_Fever_Rapid_Diagnostic_TestStrip, Complaints_Symptoms, Age, Electricity,
            Environment_Sanitised_or_not, Doctor_to_Patient, Location
        ]

        regression_input_data[1] = 1 if Availaibility_of_treated_net == 'Yes' else 0
        regression_input_data[6] = 1 if Electricity == 'Yes' else 0
        regression_input_data[7] = 1 if Environment_Sanitised_or_not == 'Yes' else 0
        regression_input_data[8] = 1 if Doctor_to_Patient == 'High' else 0
        regression_input_data[9] = 1 if Location == 'Urban' else 0

        Laboratory_Equipments_Not_Adequate = 1 if Laboratory_Equipments == 'Not Adequate' else 0
        regression_input_data.append(Laboratory_Equipments_Not_Adequate)

        if len(regression_input_data) != 11:
            st.error(f"Expected 11 features, but got {len(regression_input_data)} features.")
        else:
            lab_diagnosis_prediction = regressor.predict([regression_input_data])[0]
            diagnosis_status = "Complicated" if lab_diagnosis_prediction > 70 else "Uncomplicated"

            st.write(f"The lab diagnosis grading predicts the condition as **{diagnosis_status}**.")
            st.write(f"Malaria Lab Diagnosis: **{lab_diagnosis_prediction:.2f}**.")

            fig, ax = plt.subplots()
            ax.bar(["Malaria Negative", "Malaria Positive"], [0, 1], color=['#4CAF50', '#F44336'])
            ax.set_ylabel('Prediction')
            ax.set_title('Malaria Prediction Result')
            st.pyplot(fig)
