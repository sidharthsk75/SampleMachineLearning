from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
excel = pd.read_csv(r"D:\python\Python_Data\MachineLearningModel\delaney_solubility_with_descriptors.csv")

# Split the data into features (x) and target (y)
y = excel["logS"]
x = excel.drop("logS", axis=1)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# Initialize and train the Linear Regression model
lr = LinearRegression()
lr.fit(x_train, y_train)

# Predict the target values for the training set
y_train_predict = lr.predict(x_train)

# Calculate Mean Square Error and R2 Score for Linear Regression
mean_square_error_lr = mean_squared_error(y_train, y_train_predict)
r2_score_lr = r2_score(y_train, y_train_predict)

# Initialize and train the Random Forest Regressor model
rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

# Predict the target values for the training set using Random Forest
y_train_predict_rf = rf.predict(x_train)

# Calculate Mean Square Error and R2 Score for Random Forest
mean_square_error_rf = mean_squared_error(y_train, y_train_predict_rf)
r2_score_rf = r2_score(y_train, y_train_predict_rf)

# Create a DataFrame to display the results
table = pd.DataFrame(columns=["Algorithm", "Mean_Square_Error", "R_Square"])

# Add results for Linear Regression
table.loc[0] = ["Linear Regression", mean_square_error_lr, r2_score_lr]

# Add results for Random Forest
table.loc[1] = ["Random Forest", mean_square_error_rf, r2_score_rf]

