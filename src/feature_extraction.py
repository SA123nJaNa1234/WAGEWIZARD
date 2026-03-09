from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import numpy as np

class FeatureExtractor:
    def __init__(self, df):
        self.df = df
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.scaler = StandardScaler()
        self.le = LabelEncoder()
    
    def extract_tfidf_features(self):
        """Convert text descriptions to TF-IDF vectors"""
        tfidf_matrix = self.vectorizer.fit_transform(self.df['description_cleaned'])
        feature_names = self.vectorizer.get_feature_names_out()
        return tfidf_matrix, feature_names
    
    def encode_categorical(self):
        """Encode categorical variables"""
        self.df['experience_level_encoded'] = self.le.fit_transform(self.df['experience_level'])
        self.df['is_remote_int'] = self.df['is_remote'].astype(int)
        return self.df
    
    def combine_features(self, tfidf_matrix):
        """Combine TF-IDF with numerical features"""
        numerical_features = self.df[['skills_count', 'is_remote_int', 'experience_level_encoded']].values
        numerical_features = self.scaler.fit_transform(numerical_features)
        
        # Combine TF-IDF and numerical features
        combined_features = np.hstack([tfidf_matrix.toarray(), numerical_features])
        return combined_features
    
    def prepare_data(self):
        """Prepare complete feature set"""
        self.encode_categorical()
        tfidf_matrix, feature_names = self.extract_tfidf_features()
        X = self.combine_features(tfidf_matrix)
        y = self.df['salary_numeric'].values
        
        return X, y, feature_names

# Usage
if __name__ == "__main__":
    df = pd.read_csv('jobs_cleaned.csv')
    extractor = FeatureExtractor(df)
    X, y, feature_names = extractor.prepare_data()
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target shape: {y.shape}")