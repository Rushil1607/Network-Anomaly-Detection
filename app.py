from flask import Flask, request, jsonify, render_template
import time
import random

# Initialize the Flask application
app = Flask(__name__)

# --- Placeholder Prediction Logic ---
# This function simulates your ML model's behavior.
# We will replace this with your actual model later.
def mock_predict(data):
    """
    Simulates a prediction based on simple rules.
    - Port numbers below 1024 are often used for system services,
      so we'll flag them as potentially malicious for this demo.
    - TCP protocol is slightly more likely to be flagged.
    """
    print(f"Simulating prediction for: {data}")
    
    # Simulate processing time
    time.sleep(0.5) 
    
    port = int(data.get('id_orig_p', 0))
    protocol = data.get('proto', 'tcp')

    # Simple rule-based logic for demonstration
    if port < 1024 and port != 80 and port != 443:
        prediction = 'Malicious'
        confidence = random.uniform(85, 98)
    elif protocol == 'tcp' and port > 40000:
        if random.random() > 0.7: # 30% chance of being malicious
             prediction = 'Malicious'
             confidence = random.uniform(60, 80)
        else:
             prediction = 'Benign'
             confidence = random.uniform(90, 99)
    else:
        prediction = 'Benign'
        confidence = random.uniform(95, 99.9)
        
    return {
        'prediction': prediction,
        'confidence': f'{confidence:.2f}'
    }

# --- Flask Routes ---
@app.route('/')
def landing_page():
    """Render the landing page."""
    return render_template('landingpage.html')

@app.route('/dashboard')
def dashboard():
    """Render the main security dashboard page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive data from the form, run the mock prediction,
    and return the result.
    """
    try:
        # Get data from the POST request's JSON body
        data = request.json
        
        # Get the mock prediction
        result = mock_predict(data)
        
        # Return the result as JSON
        return jsonify(result)

    except Exception as e:
        print(f"🔴 An error occurred: {e}")
        return jsonify({'error': str(e)}), 400

# --- Run the application ---
if __name__ == '__main__':
    # Runs the app on http://127.0.0.1:5000
    app.run(debug=True)

