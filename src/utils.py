import json
import joblib
from pathlib import Path
from typing import Any

def save_json(data: Any, filepath: str):
    """Save data as JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath: str) -> Any:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_model(model: Any, filepath: str):
    """Save model using joblib"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"✅ Model saved to {filepath}")

def load_model(filepath: str) -> Any:
    """Load model using joblib"""
    return joblib.load(filepath)

def save_vectorizer(vectorizer: Any, filepath: str):
    """Save vectorizer using joblib"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, filepath)
    print(f"✅ Vectorizer saved to {filepath}")

def load_vectorizer(filepath: str) -> Any:
    """Load vectorizer using joblib"""
    return joblib.load(filepath)

def create_directories():
    """Create all necessary directories"""
    dirs = [
        'data/raw',
        'data/processed', 
        'data/slang',
        'models/production',
        'models/experiments',
        'reports/metrics',
        'reports/confusion_matrices',
        'logs',
        'app/static/css',
        'app/static/js',
        'app/templates'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ All directories created")