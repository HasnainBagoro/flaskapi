from flask import Flask, request, jsonify
import joblib
import traceback

app = Flask(__name__)

# Load trained model
try:
    bundle = joblib.load("url_model_tldfreq.pkl")
    model = bundle["model"]
    encoder = bundle["encoder"]
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    encoder = None

@app.route('/predict', methods=['POST'])
def predict_url():
    if model is None or encoder is None:
        return jsonify({
            'error': 'Model not loaded properly'
        }), 500

    try:
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                'error': 'Please provide a URL in the request body'
            }), 400

        url = data['url']
        if not isinstance(url, str):
            return jsonify({
                'error': 'URL must be a string'
            }), 400

        # Make prediction
        features = encoder.transform([url])
        prediction = model.predict(features)[0]
        label = encoder.inverse_transform([prediction])[0]
        malicious = bool(prediction)

        return jsonify({
            'url': url,
            'malicious': malicious,
            'prediction': label
        })

    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=8000)



