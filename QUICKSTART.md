# Quick Start Guide

Get up and running with the Learner Success Prediction System in 5 minutes.

## Prerequisites

- Python 3.8+ installed
- pip package manager
- 500MB free disk space

## Option 1: Automated Setup (Recommended)

### Linux/macOS
```bash
chmod +x setup.sh
./setup.sh
python app.py
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python generate_sample_data.py
python train_model.py
python app.py
```

Open `http://localhost:5000` in your browser.

## Option 2: Manual Setup

### Step 1: Clone and Setup Environment
```bash
git clone https://github.com/yourusername/learner-success-prediction.git
cd learner-success-prediction
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate Sample Data
```bash
python generate_sample_data.py
```

Output:
- `sample_learner_data.csv` (1000 learner records)

### Step 4: Train Model
```bash
python train_model.py
```

This takes 2-5 minutes and generates:
- `learner_predictor_model.pkl`
- `preprocessor.pkl`
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`

### Step 5: Run Application
```bash
python app.py
```

Visit: `http://localhost:5000`

## Option 3: Docker Setup

```bash
docker-compose up --build
```

Visit: `http://localhost:5000`

## First Prediction

### Individual Prediction

1. Click "Individual Prediction" tab
2. Enter learner metrics:
   - Video Watch Time: 25
   - Quiz Score: 75
   - Login Frequency: 12
   - Session Duration: 2.5
   - Assignment Score: 80
   - Assignment Submissions: 8
   - Forum Posts: 5
   - Completion Rate: 70
3. Click "Predict Risk Level"

### Batch Prediction

1. Click "Batch Prediction" tab
2. Upload CSV file with learner data
3. View aggregate results

## Sample CSV Format

```csv
video_watch_time,quiz_scores,login_frequency,session_duration,assignment_scores,assignment_submission,forum_posts,completion_rate
25.5,75.0,12,2.5,80.0,8,5,70.0
15.2,60.5,8,1.5,65.0,5,2,50.0
```

## Testing API Endpoints

### Using curl

```bash
# Individual prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "video_watch_time": 25.5,
    "quiz_scores": 75.0,
    "login_frequency": 12,
    "session_duration": 2.5,
    "assignment_scores": 80.0,
    "assignment_submission": 8,
    "forum_posts": 5,
    "completion_rate": 70.0
  }'

# Model info
curl http://localhost:5000/model_info
```

### Using Python

```python
import requests

url = "http://localhost:5000/predict"
data = {
    "video_watch_time": 25.5,
    "quiz_scores": 75.0,
    "login_frequency": 12,
    "session_duration": 2.5,
    "assignment_scores": 80.0,
    "assignment_submission": 8,
    "forum_posts": 5,
    "completion_rate": 70.0
}

response = requests.post(url, json=data)
print(response.json())
```

## Training with Your Own Data

1. Prepare CSV with required features (see Data Format in README)
2. Train model:
   ```bash
   python train_model.py your_data.csv
   ```
3. Restart application:
   ```bash
   python app.py
   ```

## Common Issues

### Issue: Model not found
**Solution**: Run `python train_model.py` first

### Issue: Import errors
**Solution**: Ensure virtual environment is activated and dependencies are installed
```bash
pip install -r requirements.txt
```

### Issue: Port 5000 already in use
**Solution**: Change port in `app.py`:
```python
app.run(host='0.0.0.0', port=8000, debug=True)
```

### Issue: Low model accuracy
**Solution**: 
- Increase dataset size
- Enable hyperparameter optimization
- Check data quality

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Review [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Check [API Documentation](#api-endpoints) for integration
- Explore feature engineering in `feature_engineering.py`

## Need Help?

- Create an issue on GitHub
- Check documentation
- Review code comments

## Performance Tips

1. **Faster Training**: Use `python train_model.py data.csv false`
2. **Production Mode**: Set `debug=False` in `app.py`
3. **Larger Datasets**: Consider batch processing
4. **Model Updates**: Retrain periodically with new data

Happy predicting! 🎓
