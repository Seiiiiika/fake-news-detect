from flask import Flask, request, jsonify, render_template_string
import joblib
import os

app = Flask(__name__)

try:
    classifier = joblib.load('D:\DS 423\prj_gr\models\model.pkl')
    vectorizer = joblib.load('D:\DS 423\prj_gr\models\\vectorization.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None
    vectorizer = None

def predict_fake_news(text):
    if classifier is None or vectorizer is None:
        return {'error': 'Model not loaded'}
    
    try:
        # Vectorize text
        text_vectorized = vectorizer.transform([text])
        
        # Predict
        prediction = classifier.predict(text_vectorized)[0]
        probability = classifier.predict_proba(text_vectorized)[0]
        
        # fake: 0, real: 1
        # probability[0] = xs fake
        # probability[1] = xs real
        fake_prob = float(probability[0])
        real_prob = float(probability[1])
        
        # prediction = 0: fake, prediction = 1: real
        is_fake = (prediction == 0)
        
        return {
            'prediction': 'fake' if is_fake else 'real',
            'confidence': fake_prob if is_fake else real_prob,
            'fake_probability': fake_prob,
            'real_probability': real_prob
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fake News Detector</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            textarea {
                width: 100%;
                height: 150px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                font-size: 14px;
                resize: vertical;
            }
            button {
                background-color: #007bff;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
            }
            button:hover {
                background-color: #0056b3;
            }
            #result {
                margin-top: 20px;
                padding: 15px;
                border-radius: 5px;
                display: none;
            }
            .fake {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
            }
            .real {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
            }
            .error {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
            }
            .loading {
                text-align: center;
                color: #666;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fake News Detection System</h1>
            <p>Enter the information you want to check in the box below:</p>
            
            <textarea id="newsText" placeholder="Enter the news content to check..."></textarea>
            <br>
            <button onclick="checkNews()">Check the news</button>
            
            <div id="result"></div>
        </div>

        <script>
            async function checkNews() {
                const text = document.getElementById('newsText').value.trim();
                const resultDiv = document.getElementById('result');
                
                if (!text) {
                    resultDiv.innerHTML = '<div class="error">Please enter news content!</div>';
                    resultDiv.style.display = 'block';
                    return;
                }
                
                // Show loading
                resultDiv.innerHTML = '<div class="loading">Analyzing...</div>';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({text: text})
                    });
                    
                    const result = await response.json();
                    
                    if (result.error) {
                        resultDiv.innerHTML = `<div class="error">Lỗi: ${result.error}</div>`;
                    } else {
                        const resultClass = result.prediction === 'fake' ? 'fake' : 'real';
                        
                        resultDiv.innerHTML = `
                            <div class="${resultClass}">
                                <h3>Result: ${result.prediction === 'fake' ? 'FAKE NEWS' : 'REAL NEWS'}</h3>
                                <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                                <p><strong>Fake news probability:</strong> ${(result.fake_probability * 100).toFixed(1)}%</p>
                                <p><strong>Real news probability:</strong> ${(result.real_probability * 100).toFixed(1)}%</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    resultDiv.innerHTML = `<div class="error">Lỗi kết nối: ${error.message}</div>`;
                }
            }
            
            // Allow Enter key to submit
            document.getElementById('newsText').addEventListener('keydown', function(e) {
                if (e.ctrlKey && e.key === 'Enter') {
                    checkNews();
                }
            });
        </script>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'Không có dữ liệu text'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Text không được để trống'}), 400
        
        result = predict_fake_news(text)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'Lỗi server: {str(e)}'}), 500

if __name__ == '__main__':
    if not os.path.exists('models'):
        os.makedirs('models')
    
    app.run(debug=True, host='0.0.0.0', port=5000)