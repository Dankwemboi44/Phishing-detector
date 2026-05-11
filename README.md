# 🔐 Advanced Phishing Detection System for Kiswahili & Sheng

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Flask](https://img.shields.io/badge/Flask-2.3+-red.svg)](https://flask.palletsprojects.com)
[![ML](https://img.shields.io/badge/ML-RandomForest-orange.svg)](https://scikit-learn.org)

## 🎯 Overview

An **AI-powered phishing detection system** specifically designed for **Kiswahili, Sheng, and code-switched messages** commonly used in Kenya. This system protects users from SMS, WhatsApp, and social media scams that traditional English-only detectors miss.

### Why This Matters

- Kenyan users receive 10+ scam messages weekly in local languages
- Existing detectors are trained only on English text
- Sheng evolves rapidly, requiring adaptive detection
- Code-switching confuses traditional NLP models

## ✨ Key Features

| Feature | Description | Status |
|---------|-------------|--------|
| 🔗 **URL Analysis** | Detects malicious links, shorteners, and suspicious domains | ✅ |
| 👤 **Sender Reputation** | Tracks reported scam numbers and builds trust scores | ✅ |
| 🧠 **Pattern Learning** | Adapts to new phishing patterns from user feedback | ✅ |
| 📊 **Risk Dashboard** | Visual breakdown of why a message is flagged | ✅ |
| ⚡ **Batch Processing** | Scan multiple messages at once | ✅ |
| 🌐 **Multi-language** | English, Kiswahili, Sheng, code-switching | ✅ |
| 🎯 **Real-time** | <100ms response time | ✅ |

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/phishing-detector-kenya.git
cd phishing-detector-kenya

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py