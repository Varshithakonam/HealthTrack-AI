# HealthTrack AI

**Live Demo:** [Open HealthTrack AI](https://healthtrack-ai-dwgk.onrender.com)

# HealthTrack AI

An AI-powered symptom-based health condition prediction
web application developed using Python, Flask and
Machine Learning.

## Features

- Symptom input form
- Input validation
- Machine learning predictions
- Top three potential conditions
- Model probability scores
- CSV result download
- Error handling
- Responsive user interface

## Technologies Used

- Python 3.12
- Flask
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
- HTML
- CSS

## Project Structure

HealthTrack-AI/
- app.py
- train_model.py
- dataset/
- models/
- templates/
- static/
- requirements.txt

## Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate into the project:

cd HealthTrack-AI

Create virtual environment:

python -m venv .venv

Activate environment on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Train the Model

python train_model.py

## Run the Application

python app.py

Open:

http://127.0.0.1:5000

## Machine Learning Approach

The application uses a Scikit-learn pipeline
consisting of:

1. TF-IDF Vectorizer for text feature extraction.
2. Logistic Regression for classification.
3. Predict_proba for model probability scores.

The model is trained using a small synthetic
symptom-condition dataset.

## Example Input

fever headache body pain fatigue

## Example Output

Three potential health conditions with
their corresponding model probability scores.

Actual predictions depend on model training.

## AI Usage

AI assistance was used during development for:

- Project planning and architecture.
- Code generation and debugging assistance.
- HTML and CSS development.
- Documentation preparation.

All generated code and application functionality
were reviewed and tested during development.

AI-generated suggestions were verified and
adapted to the project requirements.

## Disclaimer

This application is an educational demonstration
using synthetic data. It is not a medical diagnostic
tool and must not replace professional medical advice.
