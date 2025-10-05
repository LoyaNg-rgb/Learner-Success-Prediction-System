import pandas as pd
import numpy as np

np.random.seed(42)

def generate_sample_learner_data(n_samples=1000, save_path='sample_learner_data.csv'):
    """Generate sample learner engagement dataset"""
    
    print(f"Generating {n_samples} sample learner records...")
    
    data = {
        'learner_id': range(1, n_samples + 1),
        'video_watch_time': np.random.gamma(3, 2, n_samples) * 10,
        'quiz_scores': np.random.beta(8, 2, n_samples) * 100,
        'login_frequency': np.random.poisson(15, n_samples),
        'session_duration': np.random.gamma(2, 1.5, n_samples),
        'assignment_scores': np.random.beta(7, 2, n_samples) * 100,
        'assignment_submission': np.random.binomial(10, 0.7, n_samples),
        'forum_posts': np.random.poisson(5, n_samples),
        'completion_rate': np.random.beta(5, 2, n_samples) * 100,
        'total_videos': np.random.randint(20, 50, n_samples),
        'total_quizzes': np.random.randint(5, 15, n_samples),
        'total_assignments': np.full(n_samples, 10),
        'total_days': np.full(n_samples, 60),
        'pre_test_score': np.random.beta(4, 4, n_samples) * 100,
        'post_test_score': np.random.beta(6, 3, n_samples) * 100,
        'age': np.random.randint(18, 45, n_samples),
        'prior_knowledge': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'learning_style': np.random.choice(['Visual', 'Auditory', 'Kinesthetic'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    risk_score = (
        (100 - df['quiz_scores']) * 0.25 +
        (100 - df['completion_rate']) * 0.3 +
        (100 - df['assignment_scores']) * 0.2 +
        (15 - df['login_frequency']).clip(0) * 3 +
        (5 - df['forum_posts']).clip(0) * 2
    )
    
    threshold = risk_score.quantile(0.6)
    df['at_risk'] = (risk_score > threshold).astype(int)
    
    df.to_csv(save_path, index=False)
    
    print(f"\nSample dataset generated successfully!")
    print(f"Saved to: {save_path}")
    print(f"Total learners: {n_samples}")
    print(f"At-risk learners: {df['at_risk'].sum()} ({df['at_risk'].mean()*100:.1f}%)")
    print(f"Not at-risk learners: {(1-df['at_risk']).sum()} ({(1-df['at_risk']).mean()*100:.1f}%)")
    
    print("\nDataset Features:")
    for col in df.columns:
        print(f"  - {col}")
    
    return df

if __name__ == "__main__":
    df = generate_sample_learner_data(1000, 'sample_learner_data.csv')
    print("\nSample data preview:")
    print(df.head())
