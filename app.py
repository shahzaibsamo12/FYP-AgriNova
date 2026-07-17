from flask import Flask, render_template, jsonify, request, send_file
from flask_cors import CORS
from config import Config
from fertilizer_recommendation.routes import fertilizer_bp
from disease_detection.routes import disease_bp
from chatbot.routes import chatbot_bp

app = Flask(__name__)
app.config.from_object(Config)
CORS(app)

# Register Blueprints
app.register_blueprint(fertilizer_bp)
app.register_blueprint(disease_bp)
app.register_blueprint(chatbot_bp)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify({"status": "AgriNova Running", "modules": ["Fertilizer", "Disease", "Chatbot"]})

@app.route('/api/tts')
def text_to_speech():
    text = request.args.get('text', '')
    lang = request.args.get('lang', 'ur')
    if not text:
        return "Missing text parameter", 400
    
    import re
    import os
    import hashlib

    # Clean markdown formatting
    clean_text = re.sub(r'[#*`_-]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    if not clean_text:
        return "Empty text parameter after cleaning", 400

    # Ensure cache directory exists
    cache_dir = os.path.join(app.config.get('BASE_DIR', os.path.dirname(os.path.abspath(__file__))), 'tts_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Generate a unique hash for the lang and clean_text
    text_hash = hashlib.md5(f"{lang}_{clean_text}".encode('utf-8')).hexdigest()
    cache_file_path = os.path.join(cache_dir, f"{text_hash}.mp3")

    if os.path.exists(cache_file_path):
        try:
            return send_file(cache_file_path, mimetype="audio/mp3", as_attachment=False, conditional=True)
        except Exception as e:
            print(f"[ERROR] Failed to serve cached file {cache_file_path}: {str(e)}")
            # Try to regenerate if cached file is corrupted or failed to serve

    try:
        # Split text into sentences for parallel TTS generation
        sentences = [s.strip() for s in re.split(r'[۔\n.!?]', clean_text) if s.strip()]
        
        if len(sentences) <= 1:
            from gtts import gTTS
            tts = gTTS(text=clean_text, lang=lang)
            tts.save(cache_file_path)
        else:
            import concurrent.futures
            import io
            from gtts import gTTS

            def fetch_chunk(sentence):
                tts = gTTS(text=sentence, lang=lang)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                return fp.getvalue()

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(sentences), 8)) as executor:
                # executor.map preserves input order
                audio_chunks = list(executor.map(fetch_chunk, sentences))

            # Concatenate chunks and save
            with open(cache_file_path, 'wb') as f:
                for chunk in audio_chunks:
                    f.write(chunk)

        return send_file(cache_file_path, mimetype="audio/mp3", as_attachment=False, conditional=True)
    except Exception as e:
        print(f"[ERROR] gTTS failed: {str(e)}")
        # Delete incomplete or corrupted cache file if it was created
        if os.path.exists(cache_file_path):
            try:
                os.remove(cache_file_path)
            except Exception:
                pass
        return f"TTS Error: {str(e)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)

