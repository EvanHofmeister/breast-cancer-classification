from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('../model/rf_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = data['features']
        y_validation = data.get('y_validation')

        # Check if input is for multiple records
        if isinstance(features[0], list):
            predictions = model.predict(features)
            if y_validation is not None:
                correct = [int(pred == y_val) for pred, y_val in zip(predictions, y_validation)]
                accuracy = sum(correct) / len(correct)
                result = {'predictions': predictions.tolist(), 'correct': correct, 'accuracy': accuracy}
            else:
                result = {'predictions': predictions.tolist()}
        else:
            # Single record
            features = np.array(features).reshape(1, -1)
            prediction = model.predict(features)
            result = {'prediction': int(prediction[0])}
            if y_validation is not None:
                result['correct'] = int(prediction[0] == y_validation)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)


