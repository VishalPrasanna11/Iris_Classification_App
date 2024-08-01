import streamlit as st
import joblib  

# Load the model
try:
    model = joblib.load('./iris_model.pkl')
    print("Successfully loaded")
except Exception as e:
    print(f"Failed to load the model: {e}")

# Title
st.title("Iris Classification App")
st.write("This app predicts the **Iris flower** species!")
st.write("Please adjust the input parameters and click on the **Predict** button to get the prediction.")

# Sidebar
st.sidebar.title("User Input Parameters")
sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.5)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 2.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 0.5)

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if prediction[0] == 0:
        st.markdown("<p style='color: green; font-weight: bold;'>Prediction: Iris-setosa</p>", unsafe_allow_html=True)
    elif prediction[0] == 1:
        st.markdown("<p style='color: blue; font-weight: bold;'>Prediction: Iris-versicolor</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: red; font-weight: bold;'>Prediction: Iris-virginica</p>", unsafe_allow_html=True)
# Footer
st.sidebar.markdown("Made with ❤️ by Vishal Prasanna")
                    
