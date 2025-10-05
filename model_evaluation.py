from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ModelEvaluator:
    def __init__(self):
        self.metrics = {}
        
    def evaluate_model(self, y_true, y_pred, y_pred_proba=None):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION METRICS")
        print("="*60)
        
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred, average='binary')
        self.metrics['recall'] = recall_score(y_true, y_pred, average='binary')
        self.metrics['f1_score'] = f1_score(y_true, y_pred, average='binary')
        
        if y_pred_proba is not None:
            self.metrics['auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1 Score:  {self.metrics['f1_score']:.4f}")
        
        if 'auc' in self.metrics:
            print(f"AUC:       {self.metrics['auc']:.4f}")
        
        print("="*60)
        
        return self.metrics
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification report"""
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        
        report = classification_report(
            y_true, y_pred,
            target_names=['Not At Risk', 'At Risk']
        )
        print(report)
        print("="*60)
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not At Risk', 'At Risk'],
            yticklabels=['Not At Risk', 'At Risk']
        )
        plt.title('Confusion Matrix - Learner Success Prediction', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {save_path}")
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path='roc_curve.png'):
        """Plot and save ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba[:, 1])
        auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Learner Success Prediction', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curve saved to: {save_path}")
        return fpr, tpr, auc_score
    
    def plot_feature_importance(self, feature_importance, top_n=15, save_path='feature_importance.png'):
        """Plot top N most important features"""
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, importance = zip(*sorted_features)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(features)), importance, color='steelblue')
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature importance plot saved to: {save_path}")
    
    def generate_evaluation_report(self, y_true, y_pred, y_pred_proba=None, feature_importance=None):
        """Generate complete evaluation report with visualizations"""
        print("\n" + "="*60)
        print("GENERATING EVALUATION REPORT")
        print("="*60)
        
        metrics = self.evaluate_model(y_true, y_pred, y_pred_proba)
        
        self.print_classification_report(y_true, y_pred)
        
        self.plot_confusion_matrix(y_true, y_pred)
        
        if y_pred_proba is not None:
            self.plot_roc_curve(y_true, y_pred_proba)
        
        if feature_importance:
            self.plot_feature_importance(feature_importance)
        
        print("\n" + "="*60)
        print("EVALUATION REPORT COMPLETE")
        print("="*60)
        
        return metrics
    
    def get_metrics_summary(self):
        """Return metrics as dictionary"""
        return self.metrics
