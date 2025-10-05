import xgboost as xgb
from sklearn.model_selection import GridSearchCV, cross_val_score
import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class XGBoostLearnerPredictor:
    def __init__(self):
        self.model = None
        self.best_params = None
        self.feature_importance = None
        
    def train_baseline_model(self, X_train, y_train):
        """Train baseline XGBoost model"""
        print("\nTraining baseline XGBoost model...")
        
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        print("Baseline model trained successfully")
        
        return self.model
    
    def hyperparameter_optimization(self, X_train, y_train, cv=5):
        """Optimize XGBoost hyperparameters using GridSearchCV"""
        print("\n" + "="*60)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*60)
        
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [50, 100, 200],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            eval_metric='logloss'
        )
        
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("Starting grid search...")
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print("\n" + "="*60)
        print("BEST HYPERPARAMETERS")
        print("="*60)
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        
        print(f"\nBest F1 Score (CV): {grid_search.best_score_:.4f}")
        print("="*60)
        
        return self.model
    
    def train_optimized_model(self, X_train, y_train, optimize=True):
        """Train model with or without optimization"""
        if optimize:
            return self.hyperparameter_optimization(X_train, y_train)
        else:
            return self.train_baseline_model(X_train, y_train)
    
    def get_feature_importance(self, feature_names):
        """Extract feature importance from trained model"""
        if self.model is None:
            print("No model trained yet")
            return None
        
        importance = self.model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importance))
        
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        print("\n" + "="*60)
        print("TOP 10 MOST IMPORTANT FEATURES")
        print("="*60)
        for feature, score in sorted_importance[:10]:
            print(f"{feature}: {score:.4f}")
        print("="*60)
        
        return self.feature_importance
    
    def cross_validate_model(self, X, y, cv=5):
        """Perform cross-validation"""
        if self.model is None:
            print("No model trained yet")
            return None
        
        scores = cross_val_score(
            self.model, X, y,
            cv=cv,
            scoring='f1'
        )
        
        print(f"\nCross-Validation F1 Scores: {scores}")
        print(f"Mean F1 Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save_model(self, filepath='learner_predictor_model.pkl'):
        """Save trained model to disk"""
        if self.model is None:
            print("No model to save")
            return False
        
        model_data = {
            'model': self.model,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {filepath}")
        return True
    
    def load_model(self, filepath='learner_predictor_model.pkl'):
        """Load trained model from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.best_params = model_data.get('best_params')
            self.feature_importance = model_data.get('feature_importance')
            
            print(f"Model loaded from: {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, X):
        """Make predictions on new data"""
        if self.model is None:
            print("No model trained yet")
            return None
        
        predictions = self.model.predict(X)
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            print("No model trained yet")
            return None
        
        probabilities = self.model.predict_proba(X)
        return probabilities
