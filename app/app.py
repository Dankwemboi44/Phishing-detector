from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.phishing_detector import ShengPhishingDetector

app = Flask(__name__)
CORS(app)

# Initialize detector
print("Loading phishing detector...")
detector = ShengPhishingDetector()
detector.load()
print("Detector ready!")

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Sheng Phishing Detector - Advanced Protection</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .header .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8rem;
        }
        
        .card {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 12px;
            font-size: 1rem;
            font-family: monospace;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #4f46e5;
        }
        
        button {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 15px;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(79,70,229,0.3);
        }
        
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 12px;
            animation: slideIn 0.3s ease;
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .phishing {
            background: #fee2e2;
            border-left: 4px solid #dc2626;
        }
        
        .safe {
            background: #dcfce7;
            border-left: 4px solid #16a34a;
        }
        
        .result-title {
            font-size: 1.25rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .phishing .result-title { color: #dc2626; }
        .safe .result-title { color: #16a34a; }
        
        .confidence {
            font-size: 0.875rem;
            color: #666;
            margin-bottom: 15px;
        }
        
        .explanations {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        
        .explanations h4 {
            margin-bottom: 10px;
            color: #333;
        }
        
        .explanation-item {
            padding: 8px 0;
            border-bottom: 1px solid #e0e0e0;
            font-size: 0.875rem;
        }
        
        .explanation-item:last-child {
            border-bottom: none;
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
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
            width: auto;
            margin: 0;
        }
        
        .example-btn:hover {
            background: #4f46e5;
            color: white;
        }
        
        .footer {
            text-align: center;
            color: rgba(255,255,255,0.6);
            font-size: 0.75rem;
            margin-top: 30px;
        }
        
        .error {
            background: #fef3c7;
            color: #d97706;
            padding: 15px;
            border-radius: 12px;
            margin-top: 15px;
        }
        
        @media (max-width: 768px) {
            .header h1 { font-size: 1.5rem; }
            .card { padding: 20px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔐 Sheng Phishing Detector</h1>
            <p>AI-powered protection for Kiswahili, Sheng & mixed messages</p>
            <div class="badge">✨ Intelligent | 🎯 Accurate | 📊 Explainable</div>
        </div>
        
        <div class="card">
            <textarea id="messageInput" rows="5" placeholder="Paste your message here..."></textarea>
            <button onclick="checkMessage()">🔍 Analyze Message</button>
            
            <div style="margin-top: 15px;">
                <div class="examples">
                    <button class="example-btn" onclick="setExample('Bofya link hii upate 10000 KES free airtime')">🔴 Phishing: Prize Scam</button>
                    <button class="example-btn" onclick="setExample('Account yako imefreeze thibitisha hapa')">🔴 Phishing: Account Freeze</button>
                    <button class="example-btn" onclick="setExample('Niaje bro tukutane stage 7')">🟢 Safe: Meeting</button>
                    <button class="example-btn" onclick="setExample('Kazi iko leo Rongai')">🟢 Safe: Job</button>
                </div>
            </div>
            
            <div id="result"></div>
            <div id="error" class="error" style="display: none;"></div>
        </div>
        
        <div class="footer">
            <p>⚡ Real-time detection | 🧠 Explainable AI | 🇰🇪 Designed for Kenya</p>
            <p style="margin-top: 10px;">This system detects phishing in Kiswahili, Sheng, and code-switched text</p>
        </div>
    </div>
    
    <script>
        async function checkMessage() {
            const message = document.getElementById('messageInput').value.trim();
            
            if (!message) {
                showError('Please enter a message to analyze');
                return;
            }
            
            document.getElementById('result').innerHTML = '<div class="card" style="text-align: center;">⏳ Analyzing...</div>';
            document.getElementById('error').style.display = 'none';
            
            try {
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
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
        
        function displayResult(result) {
            const isPhishing = result.is_phishing;
            const confidence = result.confidence * 100;
            
            let html = `
                <div class="result ${isPhishing ? 'phishing' : 'safe'}">
                    <div class="result-title">
                        ${isPhishing ? '⚠️ PHISHING MESSAGE DETECTED!' : '✅ SAFE MESSAGE'}
                    </div>
                    <div class="confidence">Confidence: ${confidence.toFixed(1)}%</div>
                    <div style="background: #f5f5f5; padding: 10px; border-radius: 8px; font-family: monospace; font-size: 0.875rem;">
                        "${escapeHtml(result.message)}"
                    </div>
            `;
            
            if (isPhishing && result.explanations && result.explanations.length > 0) {
                html += `
                    <div class="explanations">
                        <h4>🔍 Why this is suspicious:</h4>
                `;
                for (const exp of result.explanations) {
                    html += `<div class="explanation-item">⚠️ ${escapeHtml(exp.explanation)}</div>`;
                }
                html += `</div>`;
            }
            
            html += `</div>`;
            document.getElementById('result').innerHTML = html;
        }
        
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.innerHTML = message;
            errorDiv.style.display = 'block';
            document.getElementById('result').innerHTML = '';
        }
        
        function setExample(text) {
            document.getElementById('messageInput').value = text;
            checkMessage();
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
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
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'Missing message field'}), 400
        
        result = detector.predict(data['message'])
        return jsonify({'success': True, 'data': result})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': detector.is_trained})

if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("🚀 Superior Phishing Detection System")
    print("📍 http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)