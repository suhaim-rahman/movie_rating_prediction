import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Update the file path to the correct location on your computer
file_path = 'C:/Users/suhai/IMDb Movies India.csv'

# Load the dataset with the specified encoding
try:
    movie_data = pd.read_csv(file_path, encoding='cp1252')
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(movie_data.head())

# Print column names to verify the exact names
print("Column names in the dataset:")
print(movie_data.columns)

# Check for missing values
print("Missing values in each column:")
print(movie_data.isnull().sum())

# Drop rows with missing target values (assuming 'Rating' is the target column)
movie_data.dropna(subset=['Rating'], inplace=True)

# Fill missing values in other columns with appropriate values or drop columns with too many missing values
movie_data.fillna(method='ffill', inplace=True)

# Check the data types of each column
print("Data types of each column:")
print(movie_data.dtypes)

# Convert 'Duration' to numeric (e.g., extracting numeric value)
try:
    movie_data['Duration'] = movie_data['Duration'].str.extract('(\d+)').astype(float)
    print("Converted 'Duration' to numeric successfully")
except Exception as e:
    print(f"Error converting 'Duration': {e}")

# Convert 'Year' to numeric (assuming 'Year' is a string with potential non-numeric characters)
try:
    movie_data['Year'] = movie_data['Year'].str.extract('(\d+)').astype(float)
    print("Converted 'Year' to numeric successfully")
except Exception as e:
    print(f"Error converting 'Year': {e}")

# Convert 'Votes' to numeric
try:
    movie_data['Votes'] = movie_data['Votes'].str.replace(',', '').astype(float)
    print("Converted 'Votes' to numeric successfully")
except Exception as e:
    print(f"Error converting 'Votes': {e}")

# Drop the 'Name' column as it is not relevant for prediction
movie_data.drop('Name', axis=1, inplace=True)

# Check for remaining non-numeric columns
non_numeric_cols = movie_data.select_dtypes(exclude=[float, int]).columns
print("Non-numeric columns:", non_numeric_cols)

# Convert categorical features to numerical values using one-hot encoding
movie_data = pd.get_dummies(movie_data, columns=non_numeric_cols, drop_first=True)

# Display the first few rows of the processed dataset
print("First few rows of the processed dataset:")
print(movie_data.head())

# Select features and target variable
X = movie_data.drop('Rating', axis=1)
y = movie_data['Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")
