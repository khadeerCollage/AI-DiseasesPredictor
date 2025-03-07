# 🏥 Disease Prediction System

## Project Overview
This project implements a machine learning-based disease prediction system that analyzes symptoms to predict possible diseases. The system utilizes deep learning techniques to provide accurate disease diagnoses based on user-reported symptoms.

## 📁 Repository Structure
- `app.py`: Streamlit web application for disease prediction
- `model2.h5`: Trained deep learning model for disease prediction
- `dataset.csv`: Primary dataset with disease and symptom information
- `symptom_Description.csv`: Contains descriptions of various diseases
- `symptom_precaution.csv`: Lists precautionary measures for each disease
- `symptom_severity.csv`: Contains severity ratings for different symptoms
- `disease_tablets.csv`: Information about medication for different diseases
- `diseases_model.ipynb`: Jupyter notebook with model development and training

## 📊 Dataset Information
The project utilizes several datasets:

1. **Main Dataset (`dataset.csv`)**: Contains disease records with corresponding symptoms (Symptom_1 to Symptom_17 columns)

2. **Symptom Severity (`symptom_severity.csv`)**: 
   - 133 unique symptoms with severity weights ranging from 1-7
   - Higher weights indicate more severe symptoms (e.g., high_fever: 7, chest_pain: 7)
   - Used to prioritize symptoms in the prediction model

3. **Disease Tablets (`disease_tablets.csv`)**:
   - Maps diseases to recommended medications (up to 4 tablets per disease)
   - Covers 42 different diseases with their respective treatment options

## 🧠 Model Development (diseases_model.ipynb)

### Data Preprocessing
- Symptom text normalization (removing spaces, standardizing names)
- One-hot encoding of symptoms (121 unique symptoms)
- Label encoding of disease categories
- Data splitting into training and testing sets

### Models Explored
1. **Deep Learning Model (Final Choice)**
   - Multi-layer neural network with:
     - Input layer: 121 neurons (one per symptom)
     - Hidden layers with dropout for regularization
     - Output layer: Softmax activation for multi-class classification
   - Trained with Adam optimizer and categorical cross-entropy loss
   - Achieved highest accuracy of approximately 95%

2. **Other Models Evaluated**
   - Support Vector Machine (SVM)
   - Random Forest
   - Naive Bayes
   - K-Nearest Neighbors (KNN)
   - Decision Trees

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 💻 Web Application (app.py)

### Features
- **Interactive Symptom Selection**: Users can enter symptoms via text input with auto-completion
- **Multi-symptom Analysis**: Requires at least 2 symptoms for accurate prediction
- **Disease Prediction**: Shows top diseases with confidence scores
- **Detailed Information**: Displays disease descriptions and recommended precautions
- **Result Export**: Generates downloadable reports in PDF and Word formats
- **Professional UI**: Clean, responsive interface with medical theme
- **Debug Information**: Shows model's reasoning process (input encoding, standardization, raw predictions)

### Implementation Details
- **Framework**: Built with Streamlit for rapid web app development
- **Model Integration**: Loads trained TensorFlow model to make predictions
- **Data Processing Pipeline**:
  1. Text input parsing and matching to known symptoms
  2. Binary encoding of selected symptoms
  3. Standardization using pre-fitted StandardScaler
  4. Prediction through neural network model
  5. Result processing and confidence calculation

- **UI Components**:
  - Instruction panel with step-by-step guidance
  - Symptom search with autocomplete
  - Expandable disease information cards
  - Confidence indicators (color-coded)
  - Download options for reports

- **Export Functionality**:
  - Word document generation using python-docx
  - PDF generation using pdfkit with wkhtmltopdf
  - Custom formatted reports with symptoms, predictions, descriptions, and precautions

- **Error Handling**:
  - Graceful fallbacks for PDF generation issues
  - Clear error messages and installation instructions
  - Input validation to ensure sufficient symptoms

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Streamlit
- pandas, numpy, scikit-learn
- python-docx (for Word document generation)
- pdfkit and wkhtmltopdf (for PDF generation)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/khadeeCollege/disease-prediction.git
cd disease-prediction