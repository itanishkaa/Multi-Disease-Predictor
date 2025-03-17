# Multi Disease Predictor

Comprehensive machine learning-based system  designed to predict the presence of various health conditions. With this tool, you can make predictions for diseases such as Cancer, Diabetes, Heart Disease, Kidney Disease, and Liver Disease, all in one place. This project uses classification algorithms to build predictive models that can assist in early detection and diagnosis.

## 📁 Dataset
Contains the raw datasets for each disease that are used to train and test the machine learning models.

- **Cancer_prediction:** Contains the dataset for cancer prediction `(data.csv)`.
- **Diabetes_prediction:** Contains the dataset for diabetes prediction `(diabetes.csv)`.
- **Heart_prediction:** Contains the dataset for heart disease prediction `(heart.csv)`.
- **Kidney_prediction:** Contains the dataset for kidney disease prediction `(kidney_disease.csv)`.
- **Liver_prediction:** Contains the dataset for liver disease prediction `(indian_liver_patient.csv)`.

## 📦 Models
This directory houses the pre-trained machine learning models for each disease prediction saved as serialized .pkl files, ready to be loaded and used for making predictions. These models have been trained on the respective datasets and are ready for use in prediction.
- `cancer.pkl`: Trained model for predicting cancer.
- `diabetes.pkl`: Trained model for predicting diabetes.
- `heart.pkl`: Trained model for predicting heart disease.
- `kidney.pkl`: Trained model for predicting kidney disease.
- `liver.pkl`: Trained model for predicting liver disease.

## 📓 Notebook
This directory contains Jupyter Notebooks that demonstrate the workflow for training models and making predictions. These notebooks are organized by disease and include steps for loading data, preprocessing, model training, evaluation, and prediction.
- `Cancer_prediction.ipynb`: Notebook for training and predicting cancer.
- `diabetes_prediction.ipynb`: Notebook for training and predicting diabetes.
- `heart_prediction.ipynb`: Notebook for training and predicting heart disease.
- `kidney_prediction.ipynb`: Notebook for training and predicting kidney disease.
- `Liver_prediction.ipynb`: Notebook for training and predicting liver disease.

## 🚀 How to Get Started
**1. Clone this repository to your local machine or server:**

```bash
https://github.com/itanishkaa/Multi-Disease-Predictor.git
```
**2. Install dependencies:** 

Create a virtual environment and install the necessary Python packages.

Ensure that you have the necessary packages like *pandas, numpy, scikit-learn, and matplotlib* installed.

**3. Run the Notebooks:** 

Launch Jupyter Notebooks to start exploring the project and training/predicting with different disease models:

```bash
jupyter notebook
```
Open any of the disease-specific notebooks (e.g., Cancer_prediction.ipynb) to interact with the models.

**4. Explore and Predict:** 

- Open any disease prediction notebook (e.g., Cancer_prediction.ipynb) to see the end-to-end process from data loading to model evaluation and prediction. 
- Use the pre-trained models and datasets to make predictions for any of the diseases (Cancer, Diabetes, Heart, Kidney, Liver).
- If you prefer to use pre-trained models directly, load them from the `models/` directory
```python
import pickle

with open('models/cancer.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Use the model for predictions
predictions = model.predict(new_data)
```

## ⚡Features
- **Multiple Disease Prediction:** Predict the likelihood of Cancer, Diabetes, Heart Disease, Kidney Disease, and Liver Disease with separate models for each condition.
- **Pre-trained Models:** Use pre-trained .pkl models for quick predictions without needing to retrain the models.
- **Interactive Notebooks:** Step-by-step guides for building, training, and testing models, making the process transparent and easy to follow.
- **Scalable for More Diseases:** Easily extend the system to include additional diseases or conditions by adding new datasets and models.

## 🧑‍💻 Model and Training
Each notebook contains a well-structured pipeline for training and evaluating the models. The general workflow is as follows:
- **Load Dataset:** Import the dataset using pandas and perform basic data inspection.
- **Data Preprocessing:** Handle missing values, feature scaling, and any other necessary transformations.
- **Model Selection:** Select an appropriate classification model (e.g., Logistic Regression, Random Forest, Support Vector Machine).
- **Model Training:** Train the model on the preprocessed dataset.
- **Model Evaluation:** Evaluate the model’s performance using metrics like accuracy, precision, recall, and F1 score.
- **Prediction:** Use the trained model to make predictions on new data.

## 🔧Technologies & Libraries Used
- `Python` for the main programming language.
- `scikit-learn` for machine learning model implementation.
- `pandas` for data manipulation.
- `numpy` for numerical computations.
- `matplotlib & seaborn` for data visualization.
- `Jupyter Notebooks` for interactive development.
- `pickle` for saving models.

## 🔍 Future Enhancements
- **Model Optimization:** Implement hyperparameter tuning and cross-validation to further improve model accuracy.
- **Deep Learning Integration:** Experiment with neural networks or ensemble learning to increase prediction accuracy.
- **Real-time Prediction API:** Deploy the models as a REST API using `Flask` or `FastAPI` for real-time predictions.
- **Explainability:** Incorporate tools like `SHAP` and `LIME` to explain model predictions and enhance trust with healthcare professionals.
