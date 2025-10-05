import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class LearnerDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_data(self, filepath):
        """Load learner dataset from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown', inplace=True)
        
        print(f"Missing values handled: {df.isnull().sum().sum()} remaining")
        return df
    
    def encode_categorical_features(self, df, categorical_columns=None):
        """Encode categorical features using Label Encoding"""
        if categorical_columns is None:
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_columns:
            if col in df.columns and col != 'learner_id':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"Encoded {len(categorical_columns)} categorical features")
        return df
    
    def create_target_variable(self, df, target_column='dropout', threshold=None):
        """Create binary target variable for at-risk learner prediction"""
        if target_column in df.columns:
            if df[target_column].dtype == 'object':
                df[target_column] = (df[target_column].str.lower() == 'yes').astype(int)
            return df
        
        if 'final_grade' in df.columns and threshold:
            df['at_risk'] = (df['final_grade'] < threshold).astype(int)
            return df
        
        if 'completion_rate' in df.columns:
            df['at_risk'] = (df['completion_rate'] < 0.5).astype(int)
            return df
        
        print("Warning: No suitable target variable found")
        return df
    
    def prepare_features(self, df, target_col='at_risk', exclude_cols=None):
        """Prepare feature matrix and target variable"""
        if exclude_cols is None:
            exclude_cols = ['learner_id', 'student_id', 'user_id', 'name']
        
        exclude_cols.append(target_col)
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target_col] if target_col in df.columns else None
        
        self.feature_columns = feature_cols
        print(f"Prepared {len(feature_cols)} features for modeling")
        
        return X, y
    
    def scale_features(self, X_train, X_test=None):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Data split: {len(X_train)} training, {len(X_test)} testing samples")
        return X_train, X_test, y_train, y_test
    
    def preprocess_pipeline(self, filepath, target_col='at_risk', test_size=0.2):
        """Complete preprocessing pipeline"""
        print("\n" + "="*60)
        print("LEARNER DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        df = self.load_data(filepath)
        if df is None:
            return None
        
        df = self.handle_missing_values(df)
        
        df = self.encode_categorical_features(df)
        
        df = self.create_target_variable(df, target_column=target_col)
        
        X, y = self.prepare_features(df, target_col=target_col)
        
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'original_data': df
        }
