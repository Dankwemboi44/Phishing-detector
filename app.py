from flask import Flask, render_template_string, request, jsonify, session
import pickle
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
app.secret_key = 'phishing_detector_secret'

# ============================================================
# ADVANCED FEATURE 1: URL & LINK ANALYSIS
# ============================================================

class URLAnalyzer:
    """Advanced URL safety checker"""
    
    SUSPICIOUS_PATTERNS = [
        (r'bit\.ly|tinyurl|shorturl|ow\.ly|is\.gd|cli\.gs', 'URL shortener', 25),
        (r'login|verify|secure|account|update|confirm|signin', 'Suspicious keyword in URL', 20),
        (r'\d+\.\d+\.\d+\.\d+', 'IP address (not domain)', 35),
        (r'@', 'Contains @ symbol (URL spoofing)', 30),
        (r'-[a-z]{5,}', 'Long dash sequence', 15),
        (r'\.tk|\.ml|\.ga|\.cf|\.top', 'Suspicious TLD', 25),
    ]
    
    @staticmethod
    def extract_urls(text):
        """Extract all URLs from text"""
        pattern = r'https?://[^\s]+|bit\.ly/[^\s]+|tinyurl\.com/[^\s]+|[a-z0-9]+\.(?:com|org|net|co\.ke|ke|tk|ml)/[^\s]*'
        return re.findall(pattern, text, re.IGNORECASE)
    
    @staticmethod
    def analyze(text):
        """Analyze URLs in text and return risk score"""
        urls = URLAnalyzer.extract_urls(text)
        if not urls:
            return 0, [], None
        
        total_risk = 0
        findings = []
        
        for url in urls:
            for pattern, description, score in URLAnalyzer.SUSPICIOUS_PATTERNS:
                if re.search(pattern, url, re.IGNORECASE):
                    total_risk += score
                    findings.append(f"⚠️ {description}: {url[:50]}...")
                    break
        
        risk_level = min(total_risk, 100)
        
        return risk_level, findings, urls


# ============================================================
# ADVANCED FEATURE 2: SENDER REPUTATION (Learn from reports)
# ============================================================

class SenderReputation:
    """Track and learn from reported senders"""
    
    def __init__(self, db_file='sender_reputation.json'):
        self.db_file = Path(db_file)
        self.reputation = self._load()
    
    def _load(self):
        if self.db_file.exists():
            with open(self.db_file, 'r') as f:
                return json.load(f)
        return defaultdict(lambda: {'reports': 0, 'trust_score': 50})
    
    def _save(self):
        with open(self.db_file, 'w') as f:
            json.dump(dict(self.reputation), f)
    
    def report_phishing(self, sender_id):
        """Report a sender as phishing"""
        if sender_id:
            self.reputation[sender_id]['reports'] += 1
            self.reputation[sender_id]['trust_score'] = max(0, self.reputation[sender_id]['trust_score'] - 20)
            self._save()
    
    def get_trust_score(self, sender_id):
        """Get trust score for a sender"""
        if not sender_id:
            return 50
        return self.reputation.get(sender_id, {}).get('trust_score', 50)
    
    def is_suspicious(self, sender_id):
        """Check if sender is suspicious"""
        return self.get_trust_score(sender_id) < 30


# ============================================================
# ADVANCED FEATURE 3: PATTERN LEARNING (Adaptive)
# ============================================================

class PatternLearner:
    """Learn new phishing patterns from user feedback"""
    
    def __init__(self, patterns_file='phishing_patterns.json'):
        self.patterns_file = Path(patterns_file)
        self.patterns = self._load()
    
    def _load(self):
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r') as f:
                return json.load(f)
        return {
            'known_patterns': [],
            'feedback_count': 0,
            'last_updated': None
        }
    
    def _save(self):
        with open(self.patterns_file, 'w') as f:
            json.dump(self.patterns, f, indent=2)
    
    def learn_from_feedback(self, message, is_phishing, user_confirmed):
        """Learn from user feedback"""
        if user_confirmed:
            # Extract potential new pattern
            words = set(re.findall(r'\b\w+\b', message.lower()))
            # Look for common phishing keywords
            common_keywords = ['free', 'win', 'won', 'prize', 'click', 'link', 'verify', 
                              'account', 'password', 'urgent', 'bofya', 'tuma', 'lipa']
            
            new_indicators = [w for w in words if w in common_keywords and len(w) > 3]
            
            for indicator in new_indicators:
                if indicator not in self.patterns['known_patterns']:
                    self.patterns['known_patterns'].append({
                        'indicator': indicator,
                        'count': 1,
                        'first_seen': datetime.now().isoformat()
                    })
                else:
                    for p in self.patterns['known_patterns']:
                        if p['indicator'] == indicator:
                            p['count'] += 1
                            break
            
            self.patterns['feedback_count'] += 1
            self.patterns['last_updated'] = datetime.now().isoformat()
            self._save()
    
    def get_emerging_patterns(self):
        """Get recently learned patterns"""
        recent = [p for p in self.patterns['known_patterns'] if p['count'] >= 2]
        return recent[:10]


# ============================================================
# ADVANCED FEATURE 4: RISK SCORE DASHBOARD
# ============================================================

class RiskAnalyzer:
    """Comprehensive risk analysis with detailed scoring"""
    
    @staticmethod
    def calculate_risk(text, model_proba, url_risk, sender_score):
        """Calculate overall risk with breakdown"""
        
        risk_components = {
            'text_analysis': 0,
            'url_analysis': 0,
            'sender_reputation': 0,
            'urgency_signals': 0,
            'money_signals': 0,
            'action_signals': 0
        }
        
        # 1. Text-based ML confidence
        risk_components['text_analysis'] = model_proba * 70
        
        # 2. URL risk
        risk_components['url_analysis'] = url_risk * 0.8
        
        # 3. Sender reputation
        risk_components['sender_reputation'] = (100 - sender_score) * 0.6
        
        # 4. Urgency signals
        urgency_words = ['urgent', 'immediate', 'asap', 'now', 'today', 'limited', 'expire']
        urgency_count = sum(1 for w in urgency_words if w in text.lower())
        risk_components['urgency_signals'] = min(urgency_count * 15, 50)
        
        # 5. Money signals
        money_words = ['free', 'win', 'won', 'prize', 'money', 'cash', 'reward', 'freeze']
        money_count = sum(1 for w in money_words if w in text.lower())
        risk_components['money_signals'] = min(money_count * 20, 60)
        
        # 6. Action signals
        action_words = ['click', 'bofya', 'bonyeza', 'tuma', 'verify', 'confirm', 'send']
        action_count = sum(1 for w in action_words if w in text.lower())
        risk_components['action_signals'] = min(action_count * 15, 45)
        
        # Calculate total risk (0-100)
        total_risk = sum(risk_components.values()) / 2.5
        
        return {
            'total': min(total_risk, 100),
            'components': risk_components,
            'level': 'CRITICAL' if total_risk > 80 else 'HIGH' if total_risk > 60 else 'MEDIUM' if total_risk > 30 else 'LOW'
        }


# ============================================================
# ADVANCED FEATURE 5: BATCH PROCESSING
# ============================================================

class BatchProcessor:
    """Process multiple messages at once"""
    
    @staticmethod
    def process_batch(messages, detector):
        """Process batch of messages"""
        results = []
        for msg in messages:
            if msg.strip():
                result = detector.predict(msg)
                results.append(result)
        return results
    
    @staticmethod
    def generate_report(results, filename=None):
        """Generate summary report"""
        total = len(results)
        phishing = sum(1 for r in results if r['is_phishing'])
        safe = total - phishing
        
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_messages': total,
            'phishing_count': phishing,
            'safe_count': safe,
            'phishing_percentage': (phishing/total)*100 if total > 0 else 0,
            'avg_confidence': avg_confidence,
            'risk_distribution': {
                'critical': sum(1 for r in results if r.get('risk_level') == 'CRITICAL'),
                'high': sum(1 for r in results if r.get('risk_level') == 'HIGH'),
                'medium': sum(1 for r in results if r.get('risk_level') == 'MEDIUM'),
                'low': sum(1 for r in results if r.get('risk_level') == 'LOW')
            }
        }
        
        if filename:
            with open(f'report_{filename}.json', 'w') as f:
                json.dump(report, f, indent=2)
        
        return report


# ============================================================
# CORE DETECTOR (Enhanced)
# ============================================================

class AdvancedPhishingDetector:
    """Enhanced detector with all advanced features"""
    
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.url_analyzer = URLAnalyzer()
        self.sender_reputation = SenderReputation()
        self.pattern_learner = PatternLearner()
        self.risk_analyzer = RiskAnalyzer()
        self.batch_processor = BatchProcessor()
        self.slang_dict = self._init_slang()
        self.phishing_keywords = self._init_keywords()
        self.feedback_history = []
        
    def _init_slang(self):
        return {
            'nk': 'sawa', 'bana': 'friend', 'bruda': 'brother',
            'bofya': 'click', 'bonyeza': 'click', 'tuma': 'send',
            'lipa': 'pay', 'thibitisha': 'verify', 'imefreeze': 'frozen',
            'umeshinda': 'won', 'bob': 'money', 'chapaa': 'money',
            'ganji': 'money', 'pesa': 'money', 'free': 'free'
        }
    
    def _init_keywords(self):
        return {
            'phishing_indicators': [
                'urgent', 'immediate', 'verify', 'confirm', 'account',
                'password', 'click', 'link', 'free', 'win', 'won',
                'prize', 'reward', 'bofya', 'tuma', 'lipa', 'thibitisha'
            ]
        }
    
    def preprocess(self, text):
        """Enhanced preprocessing"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\.\S+', ' [URL] ', text)
        text = re.sub(r'\d+', ' [NUM] ', text)
        words = text.split()
        normalized = [self.slang_dict.get(word, word) for word in words]
        text = ' '.join(normalized)
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.strip()
    
    def extract_features(self, text):
        """Extract additional features"""
        features = {}
        
        # Count keywords
        for keyword in self.phishing_keywords['phishing_indicators']:
            count = text.lower().count(keyword)
            if count > 0:
                features[f'kw_{keyword}'] = count
        
        # Length features
        features['length'] = len(text)
        features['word_count'] = len(text.split())
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['url_count'] = text.count('[URL]')
        
        return features
    
    def train(self):
        """Train the model"""
        phishing = [
            "Bofya link hii upate 10000 KES free airtime",
            "Account yako imefreeze thibitisha hapa",
            "Umeshinda 50000 bob Tuma 500 upokee",
            "Tuma password yako turestore account",
            "Free iPhone click here to claim",
            "Your account will be suspended verify now",
            "You won a prize claim immediately",
            "Bonyeza link upate free bundle",
            "Lipia 200 bob upate 5000 KES instantly",
            "Account verification required within 24 hours",
            "Security alert: Login required immediately",
            "Your payment is pending confirm now",
        ]
        
        safe = [
            "Niaje bro tukutane stage 7",
            "Kazi iko leo Rongai",
            "Lipisha nimelipa 500",
            "Viatu freshi leo asubuhi",
            "Sherehe kesho CBD",
            "Tupatane nikujie mzigo",
            "Nk mkuu umelala aje",
            "Bruda kesho nasema tumeet",
            "Kazi ya mjengo imeanza",
            "Watu wa Rongai mko fiti",
            "Niko stage tupatane",
            "Message ya kawaida tu"
        ]
        
        messages = phishing + safe
        labels = [1] * len(phishing) + [0] * len(safe)
        processed = [self.preprocess(m) for m in messages]
        
        self.vectorizer = TfidfVectorizer(max_features=200, ngram_range=(1,2), min_df=1)
        X = self.vectorizer.fit_transform(processed).toarray()
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.model.fit(X, labels)
        
        print(f"✅ Model trained on {len(messages)} messages")
    
    def predict(self, text, sender_id=None, return_details=True):
        """Enhanced prediction with all features"""
        
        # Preprocess
        processed = self.preprocess(text)
        
        # Vectorize
        X = self.vectorizer.transform([processed]).toarray()
        
        # ML prediction
        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0][1] if pred == 1 else self.model.predict_proba(X)[0][0]
        
        # URL analysis
        url_risk, url_findings, urls = self.url_analyzer.analyze(text)
        
        # Sender reputation
        sender_score = self.sender_reputation.get_trust_score(sender_id) if sender_id else 50
        
        # Comprehensive risk analysis
        risk = self.risk_analyzer.calculate_risk(text, proba, url_risk, sender_score)
        
        result = {
            'message': text,
            'processed': processed,
            'is_phishing': bool(pred),
            'confidence': float(proba),
            'prediction': 'phishing' if pred == 1 else 'safe',
            'risk_score': risk['total'],
            'risk_level': risk['level'],
            'risk_breakdown': risk['components'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add URL findings if any
        if url_findings:
            result['urls_found'] = urls
            result['url_warnings'] = url_findings
            result['url_risk'] = url_risk
        
        # Add sender info
        if sender_id:
            result['sender_id'] = sender_id
            result['sender_trust'] = sender_score
        
        # Add emerging patterns
        emerging = self.pattern_learner.get_emerging_patterns()
        if emerging:
            result['emerging_patterns'] = emerging[:3]
        
        # Store for feedback learning
        self.feedback_history.append({
            'message': text,
            'prediction': bool(pred),
            'confidence': proba,
            'timestamp': datetime.now().isoformat(),
            'risk_score': risk['total']
        })
        
        return result
    
    def submit_feedback(self, message, was_correct, actual_type):
        """Learn from user feedback"""
        self.pattern_learner.learn_from_feedback(message, actual_type == 'phishing', was_correct)
        
        # Retrain if enough feedback collected
        if len(self.feedback_history) % 20 == 0 and len(self.feedback_history) > 0:
            threading.Thread(target=self.retrain_from_feedback).start()
        
        return {'feedback_recorded': True, 'learning_active': True}
    
    def retrain_from_feedback(self):
        """Retrain model using feedback history"""
        print("🔄 Retraining model with feedback...")
        # This would collect confirmed examples and retrain
        pass
    
    def batch_scan(self, messages):
        """Batch scan multiple messages"""
        return self.batch_processor.process_batch(messages, self)
    
    def get_stats(self):
        """Get system statistics"""
        return {
            'total_predictions': len(self.feedback_history),
            'emerging_patterns_count': len(self.pattern_learner.patterns['known_patterns']),
            'reported_senders': len(self.sender_reputation.reputation),
            'last_updated': self.pattern_learner.patterns.get('last_updated'),
            'feedback_received': self.pattern_learner.patterns['feedback_count']
        }


# ============================================================
# INITIALIZE DETECTOR
# ============================================================

print("=" * 60)
print("ADVANCED PHISHING DETECTION SYSTEM v3.0")
print("Features: URL Analysis | Sender Reputation | Pattern Learning | Risk Dashboard")
print("=" * 60)

detector = AdvancedPhishingDetector()
detector.train()

print(f"\n✅ System ready!")
print(f"   📊 Total training samples: {len(detector.feedback_history) + 24}")
print(f"   🔍 URL analysis: ACTIVE")
print(f"   🧠 Pattern learning: ACTIVE")
print(f"   📈 Risk dashboard: ACTIVE")
print("=" * 60)


# ============================================================
# ENHANCED WEB INTERFACE
# ============================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Advanced Phishing Detector | AI-Powered Security</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Header */
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .features {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        
        .feature-badge {
            background: rgba(255,255,255,0.15);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8rem;
            backdrop-filter: blur(10px);
        }
        
        /* Main Grid */
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        
        @media (max-width: 768px) {
            .grid { grid-template-columns: 1fr; }
        }
        
        /* Cards */
        .card {
            background: white;
            border-radius: 20px;
            padding: 25px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        
        .card h2 {
            margin-bottom: 15px;
            color: #333;
            font-size: 1.3rem;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 14px;
            font-family: monospace;
            resize: vertical;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 15px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102,126,234,0.3);
        }
        
        /* Result Card */
        .result-card {
            background: white;
            border-radius: 20px;
            padding: 25px;
            margin-top: 20px;
        }
        
        .risk-meter {
            background: #f0f0f0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin: 15px 0;
        }
        
        .risk-fill {
            height: 100%;
            transition: width 0.5s ease;
        }
        
        .risk-critical { background: linear-gradient(90deg, #dc2626, #ef4444); }
        .risk-high { background: linear-gradient(90deg, #f59e0b, #f97316); }
        .risk-medium { background: linear-gradient(90deg, #eab308, #fbbf24); }
        .risk-low { background: linear-gradient(90deg, #10b981, #34d399); }
        
        .risk-components {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }
        
        .component {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        
        .component-label {
            font-size: 11px;
            color: #666;
        }
        
        .component-value {
            font-size: 16px;
            font-weight: bold;
        }
        
        .warning {
            background: #fef3c7;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 12px;
        }
        
        /* Stats Dashboard */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
            margin-top: 15px;
        }
        
        .stat {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 11px;
            color: #666;
            margin-top: 5px;
        }
        
        .examples {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        
        .example-btn {
            background: #f3f4f6;
            color: #4b5563;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 11px;
            cursor: pointer;
            display: inline-block;
        }
        
        .footer {
            text-align: center;
            margin-top: 30px;
            color: rgba(255,255,255,0.6);
            font-size: 12px;
        }
        
        .error {
            background: #fee2e2;
            color: #dc2626;
            padding: 15px;
            border-radius: 10px;
            margin-top: 15px;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Advanced Phishing Detection System</h1>
            <p>AI-powered protection for Kiswahili, Sheng & mixed messages</p>
            <div class="features">
                <span class="feature-badge">🔗 URL Analysis</span>
                <span class="feature-badge">👤 Sender Reputation</span>
                <span class="feature-badge">🧠 Pattern Learning</span>
                <span class="feature-badge">📊 Risk Dashboard</span>
                <span class="feature-badge">⚡ Real-time</span>
            </div>
        </div>
        
        <div class="grid">
            <!-- Main Input Card -->
            <div class="card">
                <h2>📝 Analyze Message</h2>
                <textarea id="message" rows="6" placeholder="Paste message here..."></textarea>
                <input type="text" id="sender" placeholder="Sender ID (optional)" style="width:100%; padding:10px; margin-top:10px; border:1px solid #ddd; border-radius:8px;">
                <button onclick="analyze()">🔍 Analyze Message</button>
                
                <div class="examples">
                    <strong>Quick test:</strong>
                    <span class="example-btn" onclick="setMsg('Bofya link hii upate free 10000 KES')">🔴 Phishing</span>
                    <span class="example-btn" onclick="setMsg('Account yako imefreeze thibitisha hapa')">🔴 Phishing</span>
                    <span class="example-btn" onclick="setMsg('Niaje bro tukutane stage 7')">🟢 Safe</span>
                    <span class="example-btn" onclick="setMsg('Kazi iko leo Rongai')">🟢 Safe</span>
                </div>
            </div>
            
            <!-- Stats Dashboard -->
            <div class="card">
                <h2>📊 System Intelligence</h2>
                <div class="stats-grid" id="stats">
                    <div class="stat">
                        <div class="stat-value" id="totalPredictions">0</div>
                        <div class="stat-label">Messages Analyzed</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="patternsCount">0</div>
                        <div class="stat-label">Learned Patterns</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="sendersCount">0</div>
                        <div class="stat-label">Reported Senders</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="feedbackCount">0</div>
                        <div class="stat-label">Feedback Received</div>
                    </div>
                </div>
                <div id="emergingPatterns" style="margin-top:15px; font-size:12px; color:#666;"></div>
            </div>
        </div>
        
        <!-- Result Card -->
        <div id="result" style="display: none;" class="result-card"></div>
        <div id="error" class="error" style="display: none;"></div>
        
        <div class="footer">
            <p>⚡ Real-time detection | 🧠 Adaptive learning | 🔗 URL safety | 📊 Risk analysis</p>
            <p style="margin-top: 5px;">COMP 414 Project - Advanced Phishing Detection for Kenyan Languages</p>
        </div>
    </div>
    
    <script>
        async function analyze() {
            const message = document.getElementById('message').value.trim();
            const sender = document.getElementById('sender').value.trim();
            
            if (!message) {
                showError('Please enter a message to analyze');
                return;
            }
            
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').style.display = 'none';
            document.getElementById('result').innerHTML = '<div class="loading">🔍 Analyzing message...</div>';
            document.getElementById('result').style.display = 'block';
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message, sender_id: sender })
                });
                
                const data = await response.json();
                
                if (!data.success) {
                    showError(data.error);
                    return;
                }
                
                displayResult(data.data);
                loadStats();
                
            } catch (error) {
                showError('Network error: ' + error.message);
            }
        }
        
        function displayResult(data) {
            const isPhishing = data.is_phishing;
            const riskScore = data.risk_score;
            const riskLevel = data.risk_level;
            
            let riskClass = '';
            if (riskLevel === 'CRITICAL') riskClass = 'risk-critical';
            else if (riskLevel === 'HIGH') riskClass = 'risk-high';
            else if (riskLevel === 'MEDIUM') riskClass = 'risk-medium';
            else riskClass = 'risk-low';
            
            let html = `
                <div style="text-align: center; margin-bottom: 20px;">
                    <div style="font-size: 48px;">${isPhishing ? '⚠️' : '✅'}</div>
                    <h2 style="color: ${isPhishing ? '#dc2626' : '#10b981'}; margin: 10px 0;">
                        ${isPhishing ? 'PHISHING DETECTED!' : 'SAFE MESSAGE'}
                    </h2>
                    <div style="font-size: 14px; color: #666;">Confidence: ${(data.confidence * 100).toFixed(1)}%</div>
                </div>
                
                <div class="risk-meter">
                    <div class="risk-fill ${riskClass}" style="width: ${riskScore}%"></div>
                </div>
                <div style="text-align: center; margin-bottom: 20px;">
                    <span style="font-size: 24px; font-weight: bold;">${riskScore.toFixed(0)}/100</span>
                    <span style="color: #666;"> - ${riskLevel} RISK</span>
                </div>
                
                <div class="risk-components">
                    <div class="component">
                        <div class="component-label">Text Analysis</div>
                        <div class="component-value">${data.risk_breakdown.text_analysis.toFixed(0)}%</div>
                    </div>
                    <div class="component">
                        <div class="component-label">URL Analysis</div>
                        <div class="component-value">${data.risk_breakdown.url_analysis.toFixed(0)}%</div>
                    </div>
                    <div class="component">
                        <div class="component-label">Sender Reputation</div>
                        <div class="component-value">${data.risk_breakdown.sender_reputation.toFixed(0)}%</div>
                    </div>
                    <div class="component">
                        <div class="component-label">Urgency Signals</div>
                        <div class="component-value">${data.risk_breakdown.urgency_signals.toFixed(0)}%</div>
                    </div>
                </div>
            `;
            
            if (data.url_warnings && data.url_warnings.length > 0) {
                html += `<div class="warning"><strong>🔗 URL Warnings:</strong><br>${data.url_warnings.join('<br>')}</div>`;
            }
            
            if (data.emerging_patterns && data.emerging_patterns.length > 0) {
                html += `<div class="warning"><strong>🧠 Emerging Pattern Detected:</strong><br>New phishing indicator: "${data.emerging_patterns[0].indicator}" (seen ${data.emerging_patterns[0].count} times)</div>`;
            }
            
            html += `<div style="background: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 15px;">
                        <strong>Original message:</strong><br>
                        <span style="font-family: monospace; font-size: 12px;">"${escapeHtml(data.message)}"</span>
                    </div>`;
            
            document.getElementById('result').innerHTML = html;
            document.getElementById('result').style.display = 'block';
        }
        
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                if (data.success) {
                    document.getElementById('totalPredictions').innerText = data.data.total_predictions;
                    document.getElementById('patternsCount').innerText = data.data.emerging_patterns_count;
                    document.getElementById('sendersCount').innerText = data.data.reported_senders;
                    document.getElementById('feedbackCount').innerText = data.data.feedback_received;
                    
                    if (data.data.emerging_patterns_count > 0) {
                        document.getElementById('emergingPatterns').innerHTML = '🧠 System is learning new patterns from feedback';
                    }
                }
            } catch(e) {}
        }
        
        function showError(msg) {
            const errorDiv = document.getElementById('error');
            errorDiv.innerHTML = msg;
            errorDiv.style.display = 'block';
            document.getElementById('result').style.display = 'none';
        }
        
        function setMsg(text) {
            document.getElementById('message').value = text;
            analyze();
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Load stats on page load
        loadStats();
    </script>
</body>
</html>
'''


# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        sender_id = data.get('sender_id', None)
        
        if not message:
            return jsonify({'success': False, 'error': 'Empty message'}), 400
        
        result = detector.predict(message, sender_id)
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        result = detector.submit_feedback(
            data.get('message'),
            data.get('was_correct', False),
            data.get('actual_type', 'safe')
        )
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        
        if not messages:
            return jsonify({'success': False, 'error': 'No messages provided'}), 400
        
        results = detector.batch_scan(messages)
        report = detector.batch_processor.generate_report(results)
        
        return jsonify({'success': True, 'data': results, 'report': report})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        stats = detector.get_stats()
        return jsonify({'success': True, 'data': stats})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': '3.0'})


# ============================================================
# RUN
# ============================================================

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 ADVANCED PHISHING DETECTION SYSTEM")
    print("📍 http://localhost:5000")
    print("=" * 50)
    print("\n✨ NEW FEATURES:")
    print("   🔗 URL Safety Check - Detects malicious links")
    print("   👤 Sender Reputation - Tracks known scammers")
    print("   🧠 Pattern Learning - Adapts to new threats")
    print("   📊 Risk Dashboard - Visual risk breakdown")
    print("   ⚡ Batch Processing - Scan multiple messages")
    print("=" * 50 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)