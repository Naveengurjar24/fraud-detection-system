# Fraud Detection System

This is my Python project for detecting fraudulent transactions.

## About the Project

The project takes transaction details as input and checks whether the transaction is normal or fraud.

The details used are:

- Amount
- Number of transactions
- Previous fraud
- Distance

## Output

The system shows:

- Normal or Fraud
- Fraud probability
- Risk level
- Anomaly status

## Machine Learning

I used two machine learning algorithms:

- Decision Tree
- Random Forest

I compared the accuracy of both models.

I also used Isolation Forest for finding unusual transactions.

## Website

I made a simple website using Flask.

The user enters transaction details and gets the prediction.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Flask
- HTML

## How to Run

Install the libraries:

```bash
pip install pandas scikit-learn flask joblib
