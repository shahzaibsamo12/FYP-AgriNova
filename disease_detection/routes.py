from flask import Blueprint, render_template, jsonify, request
from werkzeug.utils import secure_filename
import os
from .services import disease_service
from config import Config

disease_bp = Blueprint('disease', __name__)

@disease_bp.route('/disease')
def disease_page():
    return render_template('disease.html')

@disease_bp.route('/api/disease/detect', methods=['POST'])
def detect_disease():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file:
            filename = secure_filename(file.filename)
            os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
            file_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Content validation using PIL
            try:
                from PIL import Image
                with Image.open(file_path) as img:
                    img.verify()  # Quick check of image format/headers
            except Exception:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return jsonify({"error": "Uploaded file is not a valid image or is corrupted."}), 400

            language = request.form.get('language', 'en')
            result = disease_service.predict(file_path, language=language)
            # Safe cleanup after prediction
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify(result)
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        return jsonify({"error": str(e), "traceback": tb_str}), 500
