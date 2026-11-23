# 🚦 Traffic Congestion Prediction System

**Machine Learning-Based Traffic Volume Prediction with Live Traffic & Weather Monitoring**

A machine learning-based traffic prediction system that analyzes historical traffic patterns and estimates traffic volume using time-related and environmental features. The application also provides live traffic and weather monitoring for selected locations in Nagpur.

## 🌐 Live Website

[Click here to open the live project](https://smart-traffic-prediction-2026.streamlit.app/)

---

## 1. Introduction

Traffic congestion is a major challenge in modern urban transportation. Increasing vehicle populations, changing weather conditions, and variations in traffic patterns make traffic management difficult.

This project implements a machine learning-based Traffic Congestion Prediction System using historical traffic data and a Random Forest regression model.

The system provides an interactive Streamlit dashboard where users can explore historical traffic patterns, predict traffic volume, evaluate model performance, and monitor live traffic and weather information.

## 2. Project Objectives

- Predict traffic volume using machine learning.
- Analyze historical traffic patterns and trends.
- Identify important features influencing traffic volume.
- Visualize model performance through interactive charts.
- Monitor live traffic conditions and weather information.
- Provide a user-friendly web application using Streamlit.

## 3. Key Features

### 📊 Interactive Dashboard
- Historical traffic statistics.
- Traffic volume distribution.
- Time-based traffic visualizations.
- Dataset overview.

### 🔮 Traffic Volume Prediction
- Predict traffic volume using user-provided inputs.
- Consider hour, day, month, and weekday information.
- Include weather-related input features.
- Display predicted vehicle volume and congestion category.

### 📈 Model Performance
- Display machine learning evaluation metrics.
- Visualize model performance.
- Explore prediction accuracy and error measurements.

### ⭐ Feature Importance
- Identify the most influential prediction features.
- Visualize feature importance using charts.

### 🌦️ Live Traffic & Weather Monitoring
- Monitor selected Nagpur locations.
- Retrieve live traffic flow information using TomTom API.
- Retrieve weather information using Open-Meteo API.
- Display available live traffic and weather records.

### ℹ️ About Project
- Project overview.
- Objectives and methodology.
- Technologies and limitations.

## 4. Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Prediction Algorithm | Random Forest Regressor |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Altair |
| Web Framework | Streamlit |
| Live Traffic API | TomTom |
| Weather API | Open-Meteo |
| Model Storage | Joblib |
| Development Environment | PyCharm |

## 5. System Architecture

The project follows the workflow below:

```text
       Historical Traffic Dataset
                  |
                  v
         Data Preprocessing
                  |
                  v
          Data Analysis (EDA)
                  |
                  v
          Feature Engineering
                  |
                  v
        Machine Learning Model
                  |
                  v
         Random Forest Model
                  |
                  v
        Model Evaluation
                  |
                  v
        Streamlit Application
                  |
       +----------+-----------+
       |          |           |
       v          v           v
   Dashboard  Prediction  Performance
       |
       v
 Live Traffic & Weather Monitoring
       |
       v
   User Interface & Results
```

## 6. Dataset Information

The project uses historical traffic volume data containing traffic observations and associated time and weather information.

**Dataset Summary:**

- Total Records: 48,187
- Total Columns: 14
- Missing Values: 0
- Duplicate Records: 0

The dataset is based on historical traffic observations from the United States (Minnesota), rather than Nagpur.

The live traffic and weather monitoring component separately retrieves information for selected Nagpur locations.

## 7. Machine Learning Methodology

The project uses a Random Forest Regressor to estimate traffic volume.

**Workflow:**

1. Load historical traffic data.
2. Perform data cleaning and preprocessing.
3. Analyze traffic patterns.
4. Prepare input features and target variable.
5. Train the Random Forest model.
6. Evaluate model performance.
7. Save the trained model.
8. Integrate the model into the Streamlit application.

### Prediction Output

The model estimates traffic volume in terms of vehicle count.

The application also displays a traffic congestion category based on the predicted volume.

## 8. Project Directory Structure

```text
Traffic_Congestion_Prediction/
│
├── app.py
│
├── dataset/
│
├── models/
│   └── traffic_volume_model.pkl
│
├── src/
│   ├── Data preprocessing scripts
│   ├── Model training scripts
│   ├── Model evaluation scripts
│   └── Feature importance scripts
│
├── outputs/
│   ├── cleaned_traffic.csv
│   ├── model_comparison.csv
│   └── feature_importance.png
│
├── .env.example
├── .gitignore
├── .gitattributes
├── requirements.txt
└── README.md
```

## 9. Installation and Setup

### Prerequisites

- Python 3.13
- Git
- PyCharm or another Python IDE

### Step 1: Clone the Repository

```bash
git clone https://github.com/mayurirangari866-glitch/Traffic-Congestion-Prediction.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Traffic-Congestion-Prediction
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 4: Activate the Environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Configure API Key

Create a `.env` file in the project root:

```env
TOMTOM_API_KEY=your_tomtom_api_key_here
```

Keep your actual API key private. Never upload your `.env` file to GitHub.

### Step 7: Run the Application

```bash
streamlit run app.py
```

### Step 8: Open in Browser

```text
http://localhost:8501
```

## 10. Application Modules

| Module | Description |
|---|---|
| Dashboard | Displays historical traffic analysis and statistics |
| Traffic Prediction | Predicts traffic volume from user inputs |
| Model Performance | Displays evaluation metrics and charts |
| Feature Importance | Shows important prediction features |
| Live Traffic & Weather | Displays available live traffic and weather information |
| About Project | Explains the project and methodology |

## 11. Live Data Integration

### TomTom Traffic API

Used to retrieve available traffic flow information for selected monitoring locations.

### Open-Meteo Weather API

Used to retrieve weather information such as temperature and other available weather parameters.

**Note:** Live traffic speed and historical vehicle volume are different measurements. Live traffic speed is not treated as a direct vehicle-volume measurement.

## 12. Project Limitations

- The historical training dataset represents traffic observations from Minnesota, USA.
- The model's predictions are based on the historical dataset and its learned patterns.
- Live monitoring is limited to the selected Nagpur locations and API availability.
- Live traffic speed is not equivalent to actual vehicle count.
- The system does not currently integrate with Google Maps, Waze, social media feeds, or physical IoT traffic sensors.
- Prediction results should be interpreted as estimates, not guaranteed real-world traffic counts.

## 13. Future Enhancements

- Integrate location-specific Nagpur traffic datasets.
- Explore LSTM and other deep learning models.
- Integrate additional traffic data sources.
- Add real-time vehicle-count detection using computer vision.
- Improve prediction accuracy with more diverse historical datasets.
- Expand live monitoring to additional locations.

## 14. Conclusion

The Traffic Congestion Prediction System demonstrates how machine learning, historical traffic analysis, and live API integration can be combined in an interactive application.

The project provides traffic volume estimation, historical data visualization, model evaluation, feature importance analysis, and live traffic and weather monitoring.

It serves as a foundation for developing more advanced intelligent traffic monitoring and prediction systems.

---
