from flask import Blueprint, render_template, jsonify, request
from .services import fertilizer_service

fertilizer_bp = Blueprint('fertilizer', __name__)

@fertilizer_bp.route('/fertilizer')
def fertilizer_page():
    return render_template('fertilizer.html')

@fertilizer_bp.route('/api/fertilizer/config', methods=['GET'])
def get_config():
    config = fertilizer_service.get_config()
    if "error" in config:
        return jsonify(config), 503
    return jsonify(config)

@fertilizer_bp.route('/api/fertilizer/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = fertilizer_service.predict(data)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Prediction failed: {str(e)}")
        return jsonify({"error": str(e)}), 500
