import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Streamlit App Title
st.title("Salary Prediction Based on Experience")

# File Upload for the dataset
uploaded_file = st.file_uploader("Choose a dataset", type=["xlsx"])

if uploaded_file is not None:
    # Load the dataset
    dataset = pd.read_excel(uploaded_file)
    st.write("Dataset Preview:")
    st.write(dataset.head(5))

    # Split the dataset into independent (x) and dependent (y) variables
    x = dataset.iloc[:, :-1].values 
    y = dataset.iloc[:, -1].values

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)

    # Create and train the Linear Regression model
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Make predictions
    y_pred = regressor.predict(x_test)

    # Visualizing the Training Set
    st.subheader("Salary Vs Experience (Training Set)")
    fig, ax = plt.subplots()
    ax.scatter(x_train, y_train, color='red')
    ax.plot(x_train, regressor.predict(x_train), color='blue')
    ax.set_title('Salary Vs Experience (Training Set)')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary')
    st.pyplot(fig)

    # Visualizing the Testing Set
    st.subheader("Salary Vs Experience (Test Set)")
    fig, ax = plt.subplots()
    ax.scatter(x_test, y_test, color='red')
    ax.plot(x_train, regressor.predict(x_train), color='blue')
    ax.set_title('Salary Vs Experience (Test Set)')
    ax.set_xlabel('Years of Experience')
    ax.set_ylabel('Salary')
    st.pyplot(fig)

    # Predict salary for a given experience
    experience = st.slider("Select Experience (Years)", min_value=0, max_value=30, step=1)
    new_salary_pred = regressor.predict([[experience]])
    st.write(f'The Predicted Salary of a Person with {experience} Years of Experience is: {new_salary_pred[0]:.2f}')
