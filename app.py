from flask import Flask, request, jsonify
from flask_cors import CORS
from predict_model import predict_from_dict

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # izinkan semua origin

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        result = predict_from_dict(payload)
        return jsonify({"status": "ok", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
       