# Heart_Disease_Project
🫀 Heart Disease Prediction using Optimized SVM
This project details the end-to-end process of building, optimizing, and deploying a machine learning model to predict the presence of heart disease based on 12 key clinical features. The final model is an Optimized Support Vector Machine (SVM) packaged for deployment in a Streamlit web application.

🚀 Project Objectives
Data Processing: Clean and prepare the UCI Heart Disease dataset.

Model Optimization: Use Hyperparameter Tuning (Grid Search) to maximize model robustness.

Deployment: Create an interactive Streamlit UI and provide deployment instructions (Ngrok).

📂 Project Structure
Path

Description

data/

Contains the fetched and combined heart_disease.csv dataset.

models/

Contains the final, serialized model pipeline: final_model.pkl.

ui/

Contains the Streamlit web application source code: app.py.

notebooks/

Contains the step-by-step analysis and training process.

deployment/

Contains instructions for public deployment (ngrok_setup.txt).

results/

Contains the final evaluation metrics (evaluation_metrics.txt).

README.md

This document.

requirements.txt

Lists all necessary Python dependencies.

🛠️ Setup and Installation
1. Clone the Repository
git clone [YOUR-REPOSITORY-URL]
cd Heart_Disease_Project

2. Set up Environment
You must create a Python environment and install the dependencies listed in requirements.txt.

# Recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

3. Running the Streamlit App
Once the environment is set up and the model/data are generated:

Navigate to the project root directory.

Run the application using Streamlit:

streamlit run ui/app.py

The application will open in your browser, typically at http://localhost:8501.
