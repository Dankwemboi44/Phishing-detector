import pandas as pd
import re
import json
from pathlib import Path
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, slang_path='data/slang/sheng_slang.json'):
        """Initialize preprocessor with Sheng slang dictionary"""
        self.slang_path = Path(slang_path)
        
        # Load slang dictionary if it exists
        if self.slang_path.exists():
            try:
                with open(self.slang_path, 'r', encoding='utf-8') as f:
                    self.slang_map = json.load(f)
                print(f"✅ Loaded {len(self.slang_map)} slang terms")
            except json.JSONDecodeError:
                print(f"⚠️ Slang file is corrupted. Using empty slang map.")
                self.slang_map = {}
        else:
            print(f"⚠️ Slang file not found at {slang_path}")
            print("   Creating empty slang map...")
            self.slang_map = {}
    
    def clean_text(self, text):
        """Basic text cleaning"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace URLs
        text = re.sub(r'http\S+|www\.\S+', ' URL_HERE ', text)
        
        # Replace phone numbers (10+ digits)
        text = re.sub(r'\d{10,}', ' PHONE_HERE ', text)
        
        # Replace other numbers
        text = re.sub(r'\d+', ' NUM_HERE ', text)
        
        # Remove punctuation (keep spaces)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_slang(self, text):
        """Replace Sheng slang with standard forms"""
        if not self.slang_map:
            return text
        
        words = text.split()
        normalized = [self.slang_map.get(word, word) for word in words]
        return ' '.join(normalized)
    
    def tokenize(self, text):
        """Tokenize text into words"""
        try:
            return word_tokenize(text)
        except:
            return text.split()
    
    def preprocess(self, text, do_tokenize=False):
        """Full preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.normalize_slang(text)
        
        if do_tokenize:
            return self.tokenize(text)
        return text
    
    def preprocess_dataframe(self, df, text_column='message', do_tokenize=False):
        """Preprocess entire dataframe column"""
        df_copy = df.copy()
        df_copy['processed_text'] = df_copy[text_column].apply(
            lambda x: self.preprocess(x, do_tokenize=do_tokenize)
        )
        return df_copy
    
    def add_features(self, df):
        """Add useful features for phishing detection"""
        df_copy = df.copy()
        
        # Convert to string first to avoid issues
        messages = df_copy['message'].fillna('').astype(str)
        
        # Count URLs (using apply with re.findall instead of str.count with case parameter)
        def count_urls(text):
            return len(re.findall(r'http|bit\.ly|tinyurl|shorturl|link', text.lower()))
        
        df_copy['url_count'] = messages.apply(count_urls)
        
        # Count phone numbers
        def count_phones(text):
            return len(re.findall(r'\d{10,}', text))
        
        df_copy['phone_count'] = messages.apply(count_phones)
        
        # Message length
        df_copy['msg_length'] = messages.str.len()
        
        # Count exclamation marks
        df_copy['exclamation_count'] = messages.str.count(r'!')
        
        # Count money-related words
        def count_money_keywords(text):
            money_keywords = r'pesa|money|free|win|won|bob|kes|reward|prize'
            return len(re.findall(money_keywords, text.lower()))
        
        df_copy['money_keyword_count'] = messages.apply(count_money_keywords)
        
        return df_copy
    
    def validate_data(self, df):
        """Validate input data"""
        required_columns = ['message', 'label']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        valid_labels = ['phishing', 'safe']
        invalid_labels = df[~df['label'].isin(valid_labels)]
        
        if len(invalid_labels) > 0:
            raise ValueError(f"Invalid labels found: {invalid_labels['label'].unique()}")
        
        return True

def load_and_preprocess_data(raw_path='data/raw/messages.csv', 
                             output_path='data/processed/cleaned_messages.csv'):
    """Main function to load raw data and save preprocessed version"""
    
    print("📂 Loading raw data...")
    
    # Check if raw data exists
    raw_path = Path(raw_path)
    if not raw_path.exists():
        print(f"❌ Raw data file not found at {raw_path}")
        print("   Please create data/raw/messages.csv first")
        return None
    
    df = pd.read_csv(raw_path)
    print(f"   Loaded {len(df)} messages")
    print(f"   Labels: {df['label'].value_counts().to_dict()}")
    
    print("\n🔄 Preprocessing...")
    preprocessor = TextPreprocessor()
    
    # Validate data
    preprocessor.validate_data(df)
    
    # Clean and normalize text
    df_processed = preprocessor.preprocess_dataframe(df, text_column='message')
    
    # Add additional features
    df_processed = preprocessor.add_features(df_processed)
    
    print("\n💾 Saving preprocessed data...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    
    print(f"✅ Done! Saved to {output_path}")
    
    # Show sample
    print("\n📊 Sample after preprocessing:")
    pd.set_option('display.max_colwidth', 50)
    print(df_processed[['message', 'processed_text', 'label']].head(5))
    
    return df_processed

# Create a global preprocessor instance for use by other modules
preprocessor = TextPreprocessor()

if __name__ == "__main__":
    # Run if script executed directly
    load_and_preprocess_data()