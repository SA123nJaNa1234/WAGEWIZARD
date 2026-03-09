import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import nltk

nltk.download('punkt')
nltk.download('stopwords')

class DataPreprocessor:
    def __init__(self, df):
        self.df = df.copy()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_salary(self):
        """Extract numeric salary range"""
        def extract_salary(salary_str):
            if salary_str == "N/A":
                return np.nan
            # Extract numbers from salary string
            numbers = re.findall(r'\$?(\d+(?:,\d+)*)', str(salary_str))
            if numbers:
                return int(numbers[0].replace(',', ''))
            return np.nan
        
        self.df['salary_numeric'] = self.df['salary'].apply(extract_salary)
        return self.df
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize_and_remove_stopwords(self, text):
        """Tokenize and remove stopwords"""
        tokens = word_tokenize(text)
        return [word for word in tokens if word not in self.stop_words]
    
    def feature_engineering(self):
        """Extract features from job descriptions"""
        # Experience level
        def get_experience_level(desc):
            if pd.isna(desc):
                return "Unknown"
            desc_lower = desc.lower()
            if 'senior' in desc_lower or 'lead' in desc_lower:
                return "Senior"
            elif 'junior' in desc_lower or 'entry' in desc_lower:
                return "Junior"
            else:
                return "Mid"
        
        # Check for remote work
        def is_remote(location):
            if pd.isna(location):
                return False
            return 'remote' in str(location).lower()
        
        self.df['experience_level'] = self.df['description'].apply(get_experience_level)
        self.df['is_remote'] = self.df['location'].apply(is_remote)
        
        # Count skills mentioned
        skills_keywords = ['python', 'sql', 'machine learning', 'deep learning', 'tensorflow', 'pytorch']
        self.df['skills_count'] = self.df['description'].apply(
            lambda x: sum(1 for skill in skills_keywords if skill in str(x).lower())
        )
        
        return self.df
    
    def preprocess(self):
        """Full preprocessing pipeline"""
        print("Cleaning salary data...")
        self.clean_salary()
        
        print("Cleaning text...")
        self.df['description_cleaned'] = self.df['description'].apply(self.clean_text)
        
        print("Feature engineering...")
        self.feature_engineering()
        
        # Remove rows with missing target (salary)
        self.df = self.df.dropna(subset=['salary_numeric'])
        
        return self.df

# Usage
if __name__ == "__main__":
    df = pd.read_csv('data/raw/jobs.csv')
    preprocessor = DataPreprocessor(df)
    cleaned_df = preprocessor.preprocess()
    cleaned_df.to_csv('data/processed/jobs_cleaned.csv', index=False)
    print(cleaned_df.head())