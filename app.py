from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from data_preprocessing import LearnerDataPreprocessor
from model_training import XGBoostLearnerPredictor

app = Flask(__name__)

MODEL_PATH = 'learner_predictor_model.pkl'
PREPROCESSOR_PATH = 'preprocessor.pkl'

model_predictor = XGBoostLearnerPredictor()
preprocessor = None

if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    model_predictor.load_model(MODEL_PATH)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    print("Model and preprocessor loaded successfully")
    print(f"Loaded {len(preprocessor.feature_columns)} features for prediction")
else:
    print("ERROR: Model or preprocessor not found. Please train the model first using train_model.py")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}, Preprocessor exists: {os.path.exists(PREPROCESSOR_PATH)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if preprocessor is None or model_predictor.model is None:
            return jsonify({
                'success': False,
                'error': 'Model or preprocessor not loaded. Please train the model first.'
            }), 500
        
        data = request.json
        
        df = pd.DataFrame([data])
        
        for col in preprocessor.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        df = df[preprocessor.feature_columns]
        
        X_scaled = preprocessor.scaler.transform(df)
        
        prediction = model_predictor.predict(X_scaled)[0]
        probability = model_predictor.predict_proba(X_scaled)[0]
        
        risk_level = 'At Risk' if prediction == 1 else 'Not At Risk'
        risk_probability = probability[1] * 100
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'risk_level': risk_level,
            'risk_probability': float(risk_probability),
            'confidence': float(max(probability) * 100)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        if preprocessor is None or model_predictor.model is None:
            return jsonify({
                'success': False,
                'error': 'Model or preprocessor not loaded. Please train the model first.'
            }), 500
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        df = pd.read_csv(file)
        
        for col in preprocessor.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[preprocessor.feature_columns]
        
        X_scaled = preprocessor.scaler.transform(X)
        
        predictions = model_predictor.predict(X_scaled)
        probabilities = model_predictor.predict_proba(X_scaled)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'learner_id': i + 1,
                'prediction': int(pred),
                'risk_level': 'At Risk' if pred == 1 else 'Not At Risk',
                'risk_probability': float(prob[1] * 100)
            })
        
        return jsonify({
            'success': True,
            'predictions': results,
            'total_learners': len(results),
            'at_risk_count': sum(predictions),
            'at_risk_percentage': float(sum(predictions) / len(predictions) * 100)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    if model_predictor.model is None or preprocessor is None:
        return jsonify({
            'success': False,
            'error': 'Model or preprocessor not loaded'
        }), 500
    
    info = {
        'success': True,
        'model_type': 'XGBoost Classifier',
        'features_count': len(preprocessor.feature_columns),
        'features': preprocessor.feature_columns
    }
    
    if model_predictor.best_params:
        info['best_params'] = model_predictor.best_params
    
    if model_predictor.feature_importance:
        top_features = sorted(
            model_predictor.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        info['top_features'] = [{'name': f, 'importance': float(i)} for f, i in top_features]
    
    return jsonify(info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
