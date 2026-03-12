# Sales Prediction using Linear Regression

This project demonstrates the implementation of **Linear Regression** to predict product sales based on advertising budgets. The model learns the relationship between advertising channels and sales performance.

The goal of this project is to strengthen core **machine learning fundamentals**, including data preparation, model training, evaluation, and saving trained models for later use.

---

## Project Overview

Businesses invest heavily in advertising across multiple channels such as **TV, Radio, and Newspaper**. This project builds a machine learning model that predicts **sales revenue** based on advertising spend.

Through experimentation, it was observed that **TV and Radio advertising have stronger predictive power**, while Newspaper advertising contributes very little to the model's performance.

---

## Dataset Features

| Feature | Description |
|------|------|
| TV | Advertising budget spent on TV |
| Radio | Advertising budget spent on Radio |
| Newspaper | Advertising budget spent on Newspaper |
| Sales | Product sales generated |

After experimentation, the final model focuses on **TV and Radio** as the primary predictive features.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Pickle

---

## Machine Learning Workflow

1. Data Loading using Pandas
2. Feature Selection
3. Train-Test Split
4. Model Training using Linear Regression
5. Model Evaluation using R² Score
6. Model Serialization using Pickle
7. Custom Predictions using the saved model

---

## Model Training Example

X = df[["TV", "Radio"]]
Y = df["Sales"]

X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_Train, Y_Train)

pred = model.predict(X_Test)

print(r2_score(Y_Test, pred))

Model performance achieved an R² score of approximately 0.90, indicating strong predictive capability.

Saving the Model

The trained model is saved using Pickle, which allows it to be reused without retraining.

import pickle

with open("linear_model.pkl", "wb") as f:
    pickle.dump(model, f)
Loading the Model and Making Predictions
with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)

prediction = model.predict([[150, 30]])
print(prediction)

Example prediction estimates the sales value based on custom TV and Radio advertising budgets.

Key Learning Outcomes

Understanding how Linear Regression works

Implementing train-test split

Evaluating models using R² score

Performing feature importance analysis

Saving and loading models using pickle

Making predictions on custom input values

Future Improvements

Add data visualization for feature relationships

Implement multiple regression comparison

Build a simple web interface using Flask

Deploy the model as an API

Author

Syed Haseeb Shah

GitHub
https://github.com/codingsheep17

---

🔥 Pro tip for your GitHub repo structure (makes it look way more professional):

Sales-Prediction-Linear-Regression
│
├── data
│ └── advertising.csv
│
├── notebooks
│ └── sales_prediction.ipynb
│
├── model
│ └── linear_model.pkl
│
├── README.md
└── requirements.txt
