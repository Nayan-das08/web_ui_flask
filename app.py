from flask import Flask, request, jsonify, render_template
from keras.models import load_model
import numpy as np
import tensorflow as tf

model = load_model('./model_v5_0.keras', compile=False)
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='sparse_categorical_crossentropy')
# model = load_model('./model_v5_0.keras')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)

    # Make sure the data is in the correct shape.
    assert np.array(data).shape == (5, 14), "Data is not in the correct shape"

    # Convert the data to a numpy array and reshape it for the model
    predict_request = np.array(data).reshape(1, 5, 14)

    # Use the model to predict.
    prediction = model.predict(predict_request)

    # Take the first (and only) prediction from the output.
    output = prediction[0].tolist()

    # Return the prediction in the response.
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)