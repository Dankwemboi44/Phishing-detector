"""Train the phishing detection model"""

import pickle
import re
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data
phishing_messages = [
    "Bofya link hii upate 10000 KES free airtime",
    "Account yako imefreeze thibitisha hapa",
    "Umeshinda 50000 bob! Tuma 500 kwa 07xxxxxxx upokee",
    "Tuma password yako turestore account yako",
    "Bonyeza link hii upate free bundle ya 20GB",
    "Lipia 200 bob upate 5000 KES",
    "Account yako itafungwa ndani ya masaa 24 tuma details",
    "Free iPhone 15! Bonyeza hapa upate",
    "Umeteuliwa kushinda 100,000 KES. Thibitisha sasa",
    "M-PESA: Account yako imesimamishwa. Bofya link",
    "Urgent: Your account will be suspended",
    "Free 1000 KES! Click here to claim",
    "Verify your account now or it will be closed",
    "You won a free vacation! Claim now",
    "Security alert: Login required",
]

safe_messages = [
    "Niaje bro, tukutane stage 7 nikuje pesa yako",
    "Kazi iko leo Rongai, wakenya wa ngori",
    "Lipisha nimelipa 500. Check M-PESA",
    "Viatu freshi Rongai leo asubuhi",
    "Sherehe kesho CBD tumeet",
    "Tupatane nikujie mzigo",
    "Nk mkuu umelala aje?",
    "Bruda kesho nasema tumeet",
    "Kazi ya mjengo imeanza",
    "Watu wa Rongai mko fiti?",
    "Niko stage tupatane",
    "Hey bro, how are you?",
    "Meeting at 3pm tomorrow",
    "Your delivery has been dispatched",
    "Call me when you're free",
]

# Slang dictionary
slang_dict = {
    'nk': 'sawa', 'bana': 'friend', 'bruda': 'brother',
    'ganji': 'money', 'chapaa': 'money', 'bob': 'money',
    'bofya': 'click', 'bonyeza': 'click', 'tuma': 'send',
    'pokea': 'receive', 'lipa': 'pay', 'thibitisha': 'verify',
    'imefreeze': 'frozen', 'umeshinda': 'you_won'
}

def preprocess(text):
    """Preprocess text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' link ', text)
    text = re.sub(r'\d+', ' number ', text)
    words = text.split()
    normalized = [slang_dict.get(word, word) for word in words]
    text = ' '.join(normalized)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("=" * 50)
print("TRAINING PHISHING DETECTION MODEL")
print("=" * 50)

# Prepare data
all_messages = phishing_messages + safe_messages
labels = [1] * len(phishing_messages) + [0] * len(safe_messages)

print(f"\n📊 Training data:")
print(f"   Phishing: {len(phishing_messages)} messages")
print(f"   Safe: {len(safe_messages)} messages")
print(f"   Total: {len(all_messages)} messages")

# Preprocess
print("\n🔧 Preprocessing messages...")
processed = [preprocess(msg) for msg in all_messages]

# Create TF-IDF features
print("\n📊 Creating TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=1,
    max_df=0.95
)
X = vectorizer.fit_transform(processed)
X_dense = X.toarray()
print(f"   Feature matrix: {X_dense.shape}")

# Train model
print("\n🚀 Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
model.fit(X_dense, labels)

# Evaluate
train_pred = model.predict(X_dense)
train_acc = (train_pred == labels).mean()
print(f"\n📈 Training accuracy: {train_acc:.3f}")

# Save model
print("\n💾 Saving model...")
model_path = Path('models/production')
model_path.mkdir(parents=True, exist_ok=True)

with open(model_path / 'vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open(model_path / 'model.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"   ✅ Model saved to {model_path}")

# Test
print("\n🔍 Testing with examples:")
test_messages = [
    "Bofya link hii upate free money",
    "Niaje bro tukutane",
    "Account yako imefreeze"
]

for msg in test_messages:
    proc = preprocess(msg)
    X_test = vectorizer.transform([proc]).toarray()
    pred = model.predict(X_test)[0]
    prob = model.predict_proba(X_test)[0]
    result = "PHISHING" if pred == 1 else "SAFE"
    print(f"   {result} ({prob[pred]:.2f}): {msg[:50]}...")

print("\n" + "=" * 50)
print("✅ TRAINING COMPLETE! Run 'python app.py' to start the web app")
print("=" * 50)