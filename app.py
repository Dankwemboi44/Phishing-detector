from flask import Flask, render_template_string, request, jsonify
import re
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from dataset import get_training_data, get_dataset_stats

app = Flask(__name__)
app.secret_key = 'phishing_detector_secret_2024'


# ============================================================
# URL ANALYZER
# ============================================================

class URLAnalyzer:
    SUSPICIOUS_PATTERNS = [
        (r'bit\.ly|tinyurl|shorturl|ow\.ly|is\.gd|cli\.gs', 'URL shortener detected', 25),
        (r'login|verify|secure|account|update|confirm|signin', 'Suspicious keyword in URL', 20),
        (r'\d+\.\d+\.\d+\.\d+', 'IP address instead of domain', 35),
        (r'@', 'Contains @ symbol (URL spoofing)', 30),
        (r'\.tk|\.ml|\.ga|\.cf|\.top', 'Suspicious top-level domain', 25),
    ]

    @staticmethod
    def extract_urls(text):
        pattern = r'https?://[^\s]+|bit\.ly/[^\s]+|tinyurl\.com/[^\s]+|[a-z0-9]+\.(?:com|org|net|co\.ke|ke|tk|ml)/[^\s]*'
        return re.findall(pattern, text, re.IGNORECASE)

    @staticmethod
    def analyze(text):
        urls = URLAnalyzer.extract_urls(text)
        if not urls:
            return 0, [], []

        total_risk = 0
        findings = []

        for url in urls:
            for pattern, description, score in URLAnalyzer.SUSPICIOUS_PATTERNS:
                if re.search(pattern, url, re.IGNORECASE):
                    total_risk += score
                    findings.append(f"{description}: {url[:50]}")
                    break

        return min(total_risk, 100), findings, urls


# ============================================================
# SENDER REPUTATION
# ============================================================

class SenderReputation:
    def __init__(self, db_file='sender_reputation.json'):
        self.db_file = Path(db_file)
        self.reputation = self._load()

    def _load(self):
        if self.db_file.exists():
            with open(self.db_file, 'r') as f:
                return json.load(f)
        return {}

    def _save(self):
        with open(self.db_file, 'w') as f:
            json.dump(self.reputation, f)

    def report_phishing(self, sender_id):
        if sender_id:
            if sender_id not in self.reputation:
                self.reputation[sender_id] = {'reports': 0, 'trust_score': 50}
            self.reputation[sender_id]['reports'] += 1
            self.reputation[sender_id]['trust_score'] = max(
                0, self.reputation[sender_id]['trust_score'] - 20
            )
            self._save()

    def get_trust_score(self, sender_id):
        if not sender_id or sender_id not in self.reputation:
            return 50
        return self.reputation[sender_id].get('trust_score', 50)


# ============================================================
# RISK ANALYZER
# ============================================================

class RiskAnalyzer:
    @staticmethod
    def calculate_risk(text, model_proba, url_risk, sender_score):
        components = {
            'text_analysis': model_proba * 70,
            'url_analysis': url_risk * 0.8,
            'sender_reputation': (100 - sender_score) * 0.6,
            'urgency_signals': 0,
            'money_signals': 0,
            'action_signals': 0,
        }

        urgency_words = ['urgent', 'immediate', 'asap', 'now', 'today', 'limited', 'expire']
        urgency_count = sum(1 for w in urgency_words if w in text.lower())
        components['urgency_signals'] = min(urgency_count * 15, 50)

        money_words = ['free', 'win', 'won', 'prize', 'money', 'cash', 'reward', 'freeze']
        money_count = sum(1 for w in money_words if w in text.lower())
        components['money_signals'] = min(money_count * 20, 60)

        action_words = ['click', 'bofya', 'bonyeza', 'tuma', 'verify', 'confirm', 'send']
        action_count = sum(1 for w in action_words if w in text.lower())
        components['action_signals'] = min(action_count * 15, 45)

        total = sum(components.values()) / 2.5

        if total > 80:
            level = 'CRITICAL'
        elif total > 60:
            level = 'HIGH'
        elif total > 30:
            level = 'MEDIUM'
        else:
            level = 'LOW'

        return {'total': min(total, 100), 'components': components, 'level': level}


# ============================================================
# DETECTOR
# ============================================================

class PhishingDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.url_analyzer = URLAnalyzer()
        self.sender_reputation = SenderReputation()
        self.risk_analyzer = RiskAnalyzer()
        self.slang_dict = self._init_slang()
        self.metrics = {}
        self.feedback_history = []

    def _init_slang(self):
        return {
            'nk': 'sawa', 'bana': 'friend', 'bruda': 'brother',
            'bofya': 'click', 'bonyeza': 'click', 'tuma': 'send',
            'lipa': 'pay', 'thibitisha': 'verify', 'imefreeze': 'frozen',
            'umeshinda': 'won', 'bob': 'money', 'chapaa': 'money',
            'ganji': 'money', 'pesa': 'money', 'free': 'free'
        }

    def preprocess(self, text):
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

    def train(self):
        """Train model with proper train/test split and cross-validation"""
        messages, labels = get_training_data()
        stats = get_dataset_stats()

        print("=" * 60)
        print("TRAINING PHISHING DETECTION MODEL")
        print("=" * 60)
        print(f"Dataset: {stats['total']} messages "
              f"({stats['phishing']} phishing, {stats['safe']} safe)")

        # Preprocess
        processed = [self.preprocess(m) for m in messages]
        y = np.array(labels)

        # Train/test split (80/20, stratified)
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed, y, test_size=0.2, random_state=42, stratify=y
        )

        # Vectorize (fit on training only - no data leakage)
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        X_train = self.vectorizer.fit_transform(X_train_text).toarray()
        X_test = self.vectorizer.transform(X_test_text).toarray()

        print(f"Training set: {len(X_train_text)} samples")
        print(f"Test set: {len(X_test_text)} samples")
        print(f"Features: {X_train.shape[1]}")

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=10,
        )
        self.model.fit(X_train, y_train)

        # Cross-validation on training set
        print("\nRunning 5-fold cross-validation...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1 = cross_val_score(self.model, X_train, y_train, cv=skf, scoring='f1')
        cv_acc = cross_val_score(self.model, X_train, y_train, cv=skf, scoring='accuracy')

        # Test set evaluation (held-out - honest metric)
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        self.metrics = {
            'dataset': stats,
            'train_size': len(X_train_text),
            'test_size': len(X_test_text),
            'feature_count': X_train.shape[1],
            'cv_f1_mean': float(cv_f1.mean()),
            'cv_f1_std': float(cv_f1.std()),
            'cv_accuracy_mean': float(cv_acc.mean()),
            'cv_accuracy_std': float(cv_acc.std()),
            'test_accuracy': float(test_accuracy),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_f1': float(test_f1),
            'confusion_matrix': cm.tolist(),
            'confusion_matrix_labels': ['True Negative', 'False Positive',
                                         'False Negative', 'True Positive'],
            'classification_report': classification_report(
                y_test, y_pred, target_names=['safe', 'phishing'], zero_division=0
            ),
            'trained_at': datetime.now().isoformat(),
        }

        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (on held-out test set)")
        print("=" * 60)
        print(f"Accuracy:  {test_accuracy:.3f}")
        print(f"Precision: {test_precision:.3f}")
        print(f"Recall:    {test_recall:.3f}")
        print(f"F1-Score:  {test_f1:.3f}")
        print(f"\nCross-validation F1: {cv_f1.mean():.3f} (+/- {cv_f1.std():.3f})")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
        print("=" * 60)

        # Save metrics
        Path('reports').mkdir(exist_ok=True)
        with open('reports/metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print("\nModel trained successfully. Metrics saved to reports/metrics.json")

    def predict(self, text, sender_id=None):
        processed = self.preprocess(text)
        X = self.vectorizer.transform([processed]).toarray()

        pred = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0][1] if pred == 1 else self.model.predict_proba(X)[0][0]

        url_risk, url_findings, urls = self.url_analyzer.analyze(text)
        sender_score = self.sender_reputation.get_trust_score(sender_id) if sender_id else 50
        risk = self.risk_analyzer.calculate_risk(text, proba, url_risk, sender_score)

        result = {
            'message': text,
            'is_phishing': bool(pred),
            'confidence': float(proba),
            'prediction': 'phishing' if pred == 1 else 'safe',
            'risk_score': float(risk['total']),
            'risk_level': risk['level'],
            'risk_breakdown': {k: float(v) for k, v in risk['components'].items()},
            'timestamp': datetime.now().isoformat(),
        }

        if url_findings:
            result['urls_found'] = urls
            result['url_warnings'] = url_findings
            result['url_risk'] = float(url_risk)

        if sender_id:
            result['sender_id'] = sender_id
            result['sender_trust'] = float(sender_score)

        self.feedback_history.append({
            'message': text,
            'prediction': bool(pred),
            'confidence': float(proba),
            'timestamp': datetime.now().isoformat(),
        })

        return result

    def get_stats(self):
        return {
            'total_predictions': len(self.feedback_history),
            'reported_senders': len(self.sender_reputation.reputation),
            'metrics': self.metrics,
        }


# ============================================================
# INITIALIZE
# ============================================================

print("\n" + "=" * 60)
print("ADVANCED PHISHING DETECTION SYSTEM")
print("=" * 60)

detector = PhishingDetector()
detector.train()


# ============================================================
# WEB INTERFACE (NO EMOJIS)
# ============================================================

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Phishing Detector | Kiswahili & Sheng</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; color: white; margin-bottom: 30px; }
        .header h1 { font-size: 2.2rem; margin-bottom: 10px; font-weight: 600; }
        .header p { opacity: 0.85; }
        .features {
            display: flex; justify-content: center; gap: 15px;
            flex-wrap: wrap; margin-top: 15px;
        }
        .feature-badge {
            background: rgba(255,255,255,0.12);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 0.8rem;
            backdrop-filter: blur(10px);
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
        .card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        .card h2 {
            margin-bottom: 15px;
            color: #333;
            font-size: 1.2rem;
            font-weight: 600;
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
        textarea:focus { outline: none; border-color: #667eea; }
        input[type="text"] {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 13px;
        }
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            margin-top: 15px;
            transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        .result-card {
            background: white;
            border-radius: 16px;
            padding: 24px;
            margin-top: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        .risk-meter {
            background: #f0f0f0;
            border-radius: 10px;
            height: 16px;
            overflow: hidden;
            margin: 12px 0;
        }
        .risk-fill { height: 100%; transition: width 0.5s; }
        .risk-critical { background: linear-gradient(90deg, #dc2626, #ef4444); }
        .risk-high { background: linear-gradient(90deg, #f59e0b, #f97316); }
        .risk-medium { background: linear-gradient(90deg, #eab308, #fbbf24); }
        .risk-low { background: linear-gradient(90deg, #10b981, #34d399); }
        .risk-components {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 10px;
            margin: 15px 0;
        }
        .component {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .component-label { font-size: 10px; color: #666; text-transform: uppercase; }
        .component-value { font-size: 18px; font-weight: bold; margin-top: 4px; }
        .warning {
            background: #fef3c7;
            border-left: 3px solid #f59e0b;
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
            font-size: 12px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            margin-top: 15px;
        }
        .stat {
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        .stat-value {
            font-size: 22px;
            font-weight: bold;
            color: #667eea;
        }
        .stat-label {
            font-size: 10px;
            color: #666;
            margin-top: 4px;
            text-transform: uppercase;
        }
        .examples {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 15px;
        }
        .example-btn {
            background: #f3f4f6;
            color: #4b5563;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 11px;
            cursor: pointer;
            border: none;
            transition: all 0.2s;
        }
        .example-btn:hover { background: #667eea; color: white; }
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
        .verdict {
            text-align: center;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-size: 20px;
            font-weight: bold;
        }
        .verdict-phishing { background: #fee2e2; color: #dc2626; }
        .verdict-safe { background: #dcfce7; color: #16a34a; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #eee; }
        th { background: #f8f9fa; font-weight: 600; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Phishing Detection System</h1>
            <p>AI-powered protection for Kiswahili, Sheng and code-switched messages</p>
            <div class="features">
                <span class="feature-badge">URL Analysis</span>
                <span class="feature-badge">Sender Reputation</span>
                <span class="feature-badge">Risk Scoring</span>
                <span class="feature-badge">Cross-Validated</span>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h2>Analyze Message</h2>
                <textarea id="message" rows="6" placeholder="Paste message here..."></textarea>
                <input type="text" id="sender" placeholder="Sender ID (optional)">
                <button onclick="analyze()">Analyze Message</button>

                <div class="examples">
                    <strong>Quick test:</strong>
                    <button class="example-btn" onclick="setMsg('Bofya link hii upate free 10000 KES')">Phishing example</button>
                    <button class="example-btn" onclick="setMsg('Niaje bro tukutane stage 7')">Safe example</button>
                </div>
            </div>

            <div class="card">
                <h2>System Metrics</h2>
                <div class="stats-grid">
                    <div class="stat">
                        <div class="stat-value" id="testAcc">-</div>
                        <div class="stat-label">Test Accuracy</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="testF1">-</div>
                        <div class="stat-label">Test F1-Score</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="cvF1">-</div>
                        <div class="stat-label">CV F1 (5-fold)</div>
                    </div>
                    <div class="stat">
                        <div class="stat-value" id="datasetSize">-</div>
                        <div class="stat-label">Dataset Size</div>
                    </div>
                </div>
                <div id="metricsDetail" style="margin-top:15px; font-size:12px; color:#666;"></div>
            </div>
        </div>

        <div id="result"></div>
        <div id="error" class="error" style="display:none;"></div>

        <div class="footer">
            <p>Real-time detection | Adaptive learning | URL safety | Risk analysis</p>
            <p style="margin-top: 5px;">COMP 414 Project - Phishing Detection for Kenyan Languages</p>
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

            document.getElementById('result').innerHTML = '<div class="result-card" style="text-align:center;">Analyzing...</div>';
            document.getElementById('error').style.display = 'none';

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
            } catch (error) {
                showError('Network error: ' + error.message);
            }
        }

        function displayResult(data) {
            const isPhishing = data.is_phishing;
            const riskScore = data.risk_score;
            const riskLevel = data.risk_level;

            let riskClass = 'risk-low';
            if (riskLevel === 'CRITICAL') riskClass = 'risk-critical';
            else if (riskLevel === 'HIGH') riskClass = 'risk-high';
            else if (riskLevel === 'MEDIUM') riskClass = 'risk-medium';

            let html = '<div class="result-card">';
            html += '<div class="verdict ' + (isPhishing ? 'verdict-phishing' : 'verdict-safe') + '">';
            html += isPhishing ? 'PHISHING DETECTED' : 'SAFE MESSAGE';
            html += '</div>';
            html += '<div style="text-align:center; color:#666; margin-bottom:15px;">Confidence: ' + (data.confidence * 100).toFixed(1) + '%</div>';

            html += '<div class="risk-meter"><div class="risk-fill ' + riskClass + '" style="width:' + riskScore + '%"></div></div>';
            html += '<div style="text-align:center; margin-bottom:15px;">';
            html += '<span style="font-size:22px; font-weight:bold;">' + riskScore.toFixed(0) + '/100</span>';
            html += '<span style="color:#666;"> - ' + riskLevel + ' RISK</span>';
            html += '</div>';

            html += '<div class="risk-components">';
            html += '<div class="component"><div class="component-label">Text Analysis</div><div class="component-value">' + data.risk_breakdown.text_analysis.toFixed(0) + '%</div></div>';
            html += '<div class="component"><div class="component-label">URL Analysis</div><div class="component-value">' + data.risk_breakdown.url_analysis.toFixed(0) + '%</div></div>';
            html += '<div class="component"><div class="component-label">Sender Trust</div><div class="component-value">' + data.risk_breakdown.sender_reputation.toFixed(0) + '%</div></div>';
            html += '<div class="component"><div class="component-label">Urgency</div><div class="component-value">' + data.risk_breakdown.urgency_signals.toFixed(0) + '%</div></div>';
            html += '</div>';

            if (data.url_warnings && data.url_warnings.length > 0) {
                html += '<div class="warning"><strong>URL Warnings:</strong><br>' + data.url_warnings.join('<br>') + '</div>';
            }

            html += '<div style="background:#f8f9fa; padding:12px; border-radius:8px; margin-top:15px;">';
            html += '<strong>Message:</strong><br>';
            html += '<span style="font-family:monospace; font-size:12px;">"' + escapeHtml(data.message) + '"</span>';
            html += '</div></div>';

            document.getElementById('result').innerHTML = html;
        }

        async function loadMetrics() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                if (data.success && data.data.metrics) {
                    const m = data.data.metrics;
                    document.getElementById('testAcc').innerText = (m.test_accuracy * 100).toFixed(1) + '%';
                    document.getElementById('testF1').innerText = m.test_f1.toFixed(3);
                    document.getElementById('cvF1').innerText = m.cv_f1_mean.toFixed(3);
                    document.getElementById('datasetSize').innerText = m.dataset.total;
                    document.getElementById('metricsDetail').innerHTML =
                        'Test set size: ' + m.test_size + ' | ' +
                        'Train size: ' + m.train_size + ' | ' +
                        'Features: ' + m.feature_count;
                }
            } catch(e) {}
        }

        function showError(msg) {
            const errorDiv = document.getElementById('error');
            errorDiv.innerHTML = msg;
            errorDiv.style.display = 'block';
            document.getElementById('result').innerHTML = '';
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

        loadMetrics();
    </script>
</body>
</html>
'''


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


@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        return jsonify({'success': True, 'data': detector.get_stats()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'version': '3.1'})


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("SERVER RUNNING AT http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)