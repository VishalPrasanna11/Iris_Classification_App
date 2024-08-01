# Iris_Classification_App

Welcome to the Iris Classification App! This project provides an interactive web application for classifying Iris flower species using various machine learning algorithms.

## Overview

The Iris Classification App allows users to input features of an Iris flower and predict its species (Setosa, Versicolor, or Virginica) using pre-trained machine learning models. The app is built to demonstrate the application of machine learning in a simple, user-friendly interface.

## Features

- **Interactive UI**: Enter flower measurements (sepal length, sepal width, petal length, petal width) to get a prediction.
- **Multiple Algorithms**: Choose between different machine learning models such as:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Nearest Neighbors (KNN)
- **Visualization**: View the decision boundaries of the classifiers

## Installation

To run the app locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/VishalPrasanna11/Iris_Classification_App.git
2. Navigate to the project directory:
    ```bash
    cd Iris_Classification_App
    
3. Create a virtual environment:
    ```bash
    python -m venv venv
    
5. Activate the virtual environment:
   
    On Windows:
      ```bash
      venv\Scripts\activate
      

   On Mac:   
    ```bash
    source venv/bin/activate
    
7. Install the required packages:
    ```bash
    pip install -r requirements.txt

8. Run the app:
    ```bash
    streamlit run app.py
    
9. Open your web browser and go to http://localhost:8501 to interact with the app.


