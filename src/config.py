import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelConfig:
    """Model configuration parameters"""
    def __init__(self):
        self.max_features = 1000  # Reasonable default
        self.ngram_range = (1, 2)
        self.test_size = 0.2
        self.random_state = 42
        self.class_weight = 'balanced'
        self.lstm_units = 64
        self.embedding_dim = 50
        self.dropout_rate = 0.3
        self.epochs = 10
        self.batch_size = 16

class DataConfig:
    """Data configuration"""
    def __init__(self):
        self.raw_data_path = Path('data/raw/messages.csv')
        self.processed_data_path = Path('data/processed/cleaned_messages.csv')
        self.slang_path = Path('data/slang/sheng_slang.json')
        self.phishing_patterns_path = Path('data/external/common_phishing_patterns.json')

class AppConfig:
    """Application configuration"""
    def __init__(self):
        self.debug = os.getenv('DEBUG', 'True').lower() == 'true'
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', 5000))
        self.secret_key = os.getenv('SECRET_KEY', 'phishing-detector-secret-2024')
        self.rate_limit = "100 per minute"

class ProductionConfig:
    """Production configuration"""
    def __init__(self):
        self.model_path = Path('models/production/phishing_model.pkl')
        self.vectorizer_path = Path('models/production/vectorizer.pkl')
        self.use_cache = True
        self.cache_ttl = 3600

class Config:
    """Main configuration class"""
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.app = AppConfig()
        self.production = ProductionConfig()
        
    def validate(self):
        """Validate configuration and create directories"""
        directories = [
            Path('data/raw'),
            Path('data/processed'),
            Path('data/slang'),
            Path('data/external'),
            Path('models/production'),
            Path('models/experiments'),
            Path('reports/metrics'),
            Path('reports/confusion_matrices'),
            Path('logs')
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        return True

# Global config instance
config = Config()
config.validate()