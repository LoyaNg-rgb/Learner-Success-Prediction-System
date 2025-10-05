import pandas as pd
import numpy as np

class BehavioralFeatureEngine:
    def __init__(self):
        self.feature_names = []
    
    def create_engagement_features(self, df):
        """Create engagement-based features"""
        features = df.copy()
        
        if 'video_watch_time' in df.columns and 'total_videos' in df.columns:
            features['avg_video_watch_time'] = df['video_watch_time'] / (df['total_videos'] + 1)
        
        if 'login_frequency' in df.columns and 'total_days' in df.columns:
            features['login_rate'] = df['login_frequency'] / (df['total_days'] + 1)
        
        if 'quiz_scores' in df.columns and 'total_quizzes' in df.columns:
            features['quiz_completion_rate'] = df['quiz_scores'].notna().sum() / (df['total_quizzes'] + 1)
        
        return features
    
    def create_performance_features(self, df):
        """Create performance-based features"""
        features = df.copy()
        
        if 'quiz_scores' in df.columns:
            features['avg_quiz_score'] = df['quiz_scores']
            features['quiz_score_std'] = df.groupby(level=0)['quiz_scores'].transform('std').fillna(0)
        
        if 'assignment_scores' in df.columns:
            features['avg_assignment_score'] = df['assignment_scores']
        
        if 'pre_test_score' in df.columns and 'post_test_score' in df.columns:
            features['knowledge_gain'] = df['post_test_score'] - df['pre_test_score']
            features['knowledge_gain_ratio'] = features['knowledge_gain'] / (df['pre_test_score'] + 1)
        
        return features
    
    def create_behavioral_patterns(self, df):
        """Create behavioral pattern features"""
        features = df.copy()
        
        if 'session_duration' in df.columns:
            features['avg_session_duration'] = df['session_duration']
            features['total_study_time'] = df['session_duration'] * df.get('login_frequency', 1)
        
        if 'assignment_submission' in df.columns and 'total_assignments' in df.columns:
            features['submission_rate'] = df['assignment_submission'] / (df['total_assignments'] + 1)
            features['late_submission_rate'] = df.get('late_submissions', 0) / (df['total_assignments'] + 1)
        
        if 'forum_posts' in df.columns:
            features['social_engagement'] = df['forum_posts'] + df.get('peer_interactions', 0)
        
        return features
    
    def create_trend_features(self, df):
        """Create trend-based features"""
        features = df.copy()
        
        if 'session_duration' in df.columns and 'week' in df.columns:
            features['session_trend'] = df.groupby('learner_id')['session_duration'].transform(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        
        if 'login_frequency' in df.columns and 'week' in df.columns:
            features['engagement_trend'] = df.groupby('learner_id')['login_frequency'].transform(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
            )
        
        return features
    
    def create_consistency_features(self, df):
        """Create consistency and regularity features"""
        features = df.copy()
        
        if 'login_frequency' in df.columns:
            features['login_consistency'] = df.groupby('learner_id')['login_frequency'].transform('std').fillna(0)
        
        if 'session_duration' in df.columns:
            features['session_consistency'] = df.groupby('learner_id')['session_duration'].transform('std').fillna(0)
        
        return features
    
    def engineer_features(self, df):
        """Complete feature engineering pipeline"""
        print("\n" + "="*60)
        print("BEHAVIORAL FEATURE ENGINEERING")
        print("="*60)
        
        original_features = df.shape[1]
        
        df = self.create_engagement_features(df)
        df = self.create_performance_features(df)
        df = self.create_behavioral_patterns(df)
        
        new_features = df.shape[1] - original_features
        print(f"Created {new_features} new behavioral features")
        
        self.feature_names = df.columns.tolist()
        
        print("="*60)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*60)
        
        return df
    
    def get_feature_importance_categories(self):
        """Return feature categories for interpretation"""
        categories = {
            'engagement': ['video_watch_time', 'login_frequency', 'avg_video_watch_time', 'login_rate'],
            'performance': ['quiz_scores', 'assignment_scores', 'knowledge_gain', 'avg_quiz_score'],
            'behavioral': ['session_duration', 'submission_rate', 'social_engagement'],
            'trends': ['session_trend', 'engagement_trend'],
            'consistency': ['login_consistency', 'session_consistency']
        }
        return categories
