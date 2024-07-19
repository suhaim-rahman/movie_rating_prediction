# Movie Prediction Model

This repository contains code and data for building a machine learning model to predict movie ratings based on various features.

## Dataset

The dataset used for this project is `IMDb Movies India.csv`, which contains the following columns:

- `Name`: Name of the movie
- `Year`: Year of release
- `Duration`: Duration of the movie
- `Genre`: Genre of the movie
- `Rating`: IMDb rating of the movie
- `Votes`: Number of votes
- `Director`: Director of the movie
- `Actor 1`: Lead actor
- `Actor 2`: Supporting actor
- `Actor 3`: Supporting actor

## Requirements

- Python 3.6+
- pandas
- scikit-learn


The script performs the following steps:

Loads the dataset and displays its shape and first few rows.
Preprocesses the data by filling missing values and converting categorical features to numerical values.
Splits the data into training and testing sets.
Trains a Linear Regression model.
Evaluates the model using Mean Squared Error and R2 Score.
Code Overview
The movie_prediction.py script contains the following main sections:

Data Loading: Load the dataset and display basic details.
Missing Values: Handle missing values in the dataset.
Data Preprocessing: Convert categorical features to numerical values.
Model Training: Train a Linear Regression model.
Model Evaluation: Evaluate the model using Mean Squared Error and R2 Score.
Results
After running the script, you will see the evaluation results for the model, including Mean Squared Error and R2 Score.

Contributing
If you would like to contribute to this project, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgements
Special thanks to the creators of the dataset and the developers of the libraries used in this project.
