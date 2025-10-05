# Learner Success Prediction System

A machine learning system for predicting at-risk learners using XGBoost, designed to help educational institutions identify students who may need additional support.

## Features

- **Individual Prediction**: Analyze single learner profiles for risk assessment
- **Batch Prediction**: Process multiple learners simultaneously via CSV upload
- **XGBoost Model**: High-performance gradient boosting classifier
- **Feature Engineering**: Behavioral pattern analysis and engagement metrics
- **Web Interface**: User-friendly Flask-based dashboard
- **Comprehensive Evaluation**: ROC curves, confusion matrices, and feature importance analysis

## Tech Stack

- **Backend**: Flask, Python 3.8+
- **Machine Learning**: XGBoost, scikit-learn
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Frontend**: HTML, CSS, JavaScript

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/learner-success-prediction.git
cd learner-success-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Sample Data

Generate synthetic learner data for testing:

```bash
python generate_sample_data.py
```

This creates `sample_learner_data.csv` with 1000 learner records.

### 2. Train the Model

Train the XGBoost model with hyperparameter optimization:

```bash
python train_model.py
```

For faster training without optimization:

```bash
python train_model.py sample_learner_data.csv false
```

This generates:
- `learner_predictor_model.pkl` - Trained model
- `preprocessor.pkl` - Data preprocessor
- `confusion_matrix.png` - Performance visualization
- `roc_curve.png` - ROC curve with AUC score
- `feature_importance.png` - Top features

### 3. Run the Web Application

Start the Flask server:

```bash
python app.py
```

Access the application at `http://localhost:5000`

## Data Format

Input CSV files should contain the following features:

| Feature | Description | Type |
|---------|-------------|------|
| video_watch_time | Hours spent watching videos | Float |
| quiz_scores | Average quiz score (0-100) | Float |
| login_frequency | Logins per week | Integer |
| session_duration | Average session length (hours) | Float |
| assignment_scores | Average assignment score (0-100) | Float |
| assignment_submission | Number of submissions | Integer |
| forum_posts | Forum participation count | Integer |
| completion_rate | Course completion percentage | Float |
| at_risk | Target variable (0=not at risk, 1=at risk) | Integer |

Additional features are automatically generated during preprocessing.

## API Endpoints

### Individual Prediction
```http
POST /predict
Content-Type: application/json

{
  "video_watch_time": 25.5,
  "quiz_scores": 75.0,
  "login_frequency": 12,
  "session_duration": 2.5,
  "assignment_scores": 80.0,
  "assignment_submission": 8,
  "forum_posts": 5,
  "completion_rate": 70.0
}
```

### Batch Prediction
```http
POST /batch_predict
Content-Type: multipart/form-data

file: learner_data.csv
```

### Model Information
```http
GET /model_info
```

## Project Structure

```
learner-success-prediction/
├── app.py                      # Flask application
├── data_preprocessing.py       # Data preprocessing pipeline
├── feature_engineering.py      # Feature creation
├── model_training.py           # XGBoost training
├── model_evaluation.py         # Model evaluation metrics
├── train_model.py              # Training script
├── generate_sample_data.py     # Sample data generator
├── templates/
│   └── index.html              # Web interface
├── learner_predictor_model.pkl # Trained model (generated)
├── preprocessor.pkl            # Preprocessor (generated)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Model Performance

The trained model achieves:
- **Accuracy**: ~85-90%
- **Precision**: ~80-85%
- **Recall**: ~75-85%
- **F1 Score**: ~80-85%
- **AUC**: ~0.85-0.90

Results may vary based on dataset characteristics.

## Configuration

Modify these parameters in `model_training.py`:

```python
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with XGBoost for high-performance predictions
- Flask for web framework
- scikit-learn for machine learning utilities

## Support

For issues and questions:
- Create an issue on GitHub
- Contact: loyanganba.ngathem@gmail.com

## Future Enhancements

- [ ] Real-time prediction updates
- [ ] Dashboard analytics
- [ ] Multi-class risk levels
- [ ] LSTM for temporal pattern analysis
- [ ] Intervention recommendation system
- [ ] A/B testing framework
