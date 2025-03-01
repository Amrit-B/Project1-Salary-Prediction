import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
dataset = pd.read_excel(r"S:\TCS\Projects\SALARY PREDICTION\SALARY PREDICTION\Salary_Data.xlsx")
print(dataset.head(5))

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
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Testing Set
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary Vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Predict salary for a given experience
new_salary_pred = regressor.predict([[13]])
print('The Predicted Salary of a Person with 13 Years of Experience is:', new_salary_pred)
