import streamlit as st
import joblib  
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import seaborn as sns 

# Load the Iris dataset
iris = load_iris()
X = iris.data 
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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

#Model Selection from Various Models
st.sidebar.title("Select Model")
model_name = st.sidebar.selectbox("Model", ["Logistic Regression","Decision Tree", "Random Forest", "Support Vector Machine","K-Nearest Neighbors (KNN)"])
if model_name == "Logistic Regression":
    model = joblib.load('./iris_model_lr.pkl')
elif model_name == "Decision Tree":
    model = joblib.load('./iris_model_dt.pkl')
elif model_name == "Random Forest":
    model = joblib.load('./iris_model_rf.pkl')
elif model_name == "Support Vector Machine":
    model = joblib.load('./iris_model_svc.pkl')
else:
    model = joblib.load('./iris_model_knn.pkl')    

# Prediction
if st.sidebar.button("Predict"):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    if prediction[0] == 0:
        st.markdown("<p style='color: green; font-weight: bold;'>Prediction: Iris-setosa</p>", unsafe_allow_html=True)
    elif prediction[0] == 1:
        st.markdown("<p style='color: blue; font-weight: bold;'>Prediction: Iris-versicolor</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: red; font-weight: bold;'>Prediction: Iris-virginica</p>", unsafe_allow_html=True)

# Decision boundaries of the classifiers 
def plot_decision_boundaries():
    X_vis = iris.data[:, 2:] 
    X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y, test_size=0.3, random_state=42)
    X_train_vis = scaler.fit_transform(X_train_vis)
    X_test_vis = scaler.transform(X_test_vis)

    x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
    y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    model_vis = model.fit(X_train_vis, y_train_vis)
    Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolor='k', s=20, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.title(f'Decision Boundary for {model_name}')
    st.pyplot(plt)

if st.sidebar.checkbox("Show Decision Boundaries"):
    if model_name in ["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine", "K-Nearest Neighbors (KNN)"]:
        plot_decision_boundaries()
    else:
        st.write(f"Decision boundaries visualization is not supported for the selected model ({model_name}).")

# Footer
st.sidebar.markdown("Made with ❤️ by Vishal Prasanna")
                    
