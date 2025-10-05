import warnings
warnings.filterwarnings('ignore')

from data_preprocessing import LearnerDataPreprocessor
from feature_engineering import BehavioralFeatureEngine
from model_training import XGBoostLearnerPredictor
from model_evaluation import ModelEvaluator
import os

def train_learner_success_model(dataset_path='sample_learner_data.csv', optimize=True):
    """Complete training pipeline for learner success prediction"""
    
    print("\n" + "="*80)
    print(" " * 20 + "LEARNER SUCCESS PREDICTION MODEL TRAINING")
    print("="*80)
    
    if not os.path.exists(dataset_path):
        print(f"\nError: Dataset not found at {dataset_path}")
        print("Please provide a valid dataset or run generate_sample_data.py first")
        return False
    
    preprocessor = LearnerDataPreprocessor()
    results = preprocessor.preprocess_pipeline(dataset_path, target_col='at_risk')
    
    if results is None:
        print("Preprocessing failed")
        return False
    
    X_train = results['X_train']
    X_test = results['X_test']
    y_train = results['y_train']
    y_test = results['y_test']
    feature_columns = results['feature_columns']
    
    trainer = XGBoostLearnerPredictor()
    model = trainer.train_optimized_model(X_train, y_train, optimize=optimize)
    
    feature_importance = trainer.get_feature_importance(feature_columns)
    
    trainer.save_model('learner_predictor_model.pkl')
    
    with open('preprocessor.pkl', 'wb') as f:
        import pickle
        pickle.dump(preprocessor, f)
    print("Preprocessor saved to: preprocessor.pkl")
    
    print("\n" + "="*80)
    print("EVALUATING MODEL ON TEST SET")
    print("="*80)
    
    y_pred = trainer.predict(X_test)
    y_pred_proba = trainer.predict_proba(X_test)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.generate_evaluation_report(
        y_test, y_pred, y_pred_proba, feature_importance
    )
    
    print("\n" + "="*80)
    print("MODEL TRAINING COMPLETE!")
    print("="*80)
    print("\nGenerated Files:")
    print("  1. learner_predictor_model.pkl - Trained XGBoost model")
    print("  2. preprocessor.pkl - Data preprocessor")
    print("  3. confusion_matrix.png - Confusion matrix visualization")
    print("  4. roc_curve.png - ROC curve with AUC score")
    print("  5. feature_importance.png - Top feature importance plot")
    
    print("\n" + "="*80)
    print("FINAL MODEL PERFORMANCE")
    print("="*80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")
    print("="*80 + "\n")
    
    return True

if __name__ == "__main__":
    import sys
    
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'sample_learner_data.csv'
    optimize = True if len(sys.argv) <= 2 or sys.argv[2].lower() != 'false' else False
    
    success = train_learner_success_model(dataset_path, optimize)
    
    if success:
        print("You can now run the Flask application with: python app.py")
    else:
        print("Training failed. Please check the error messages above.")
