# AgriNova

AgriNova is a Flask-based AI web application developed as a Final Year Project. It provides Pakistani farmers with intelligent, data-driven support across three domains: crop disease identification, fertilizer recommendation, and an agricultural advisory chatbot. All modules are integrated into a single web interface with bilingual support in English and Urdu.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Modules](#modules)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Machine Learning Models](#machine-learning-models)
- [Dataset Information](#dataset-information)
- [API Reference](#api-reference)
- [Installation and Setup](#installation-and-setup)
- [Training the Models](#training-the-models)
- [Configuration](#configuration)

---

## Project Overview

AgriNova addresses the gap between agricultural expertise and field-level decision-making. By combining computer vision, classical machine learning, and large language models, the system enables farmers without specialist knowledge to identify plant diseases from photographs, receive tailored fertilizer guidance, and ask open-ended agricultural questions in their preferred language.

The application is deployed as a web server accessible through any browser. It serves three independent Flask blueprints under a unified interface, backed by pre-trained models that are loaded into memory at startup to eliminate per-request loading latency.

---

## Modules

### Disease Detection

The disease detection pipeline operates in two sequential stages.

Stage 1 uses a custom DenseNet-121 implementation in PyTorch to classify the uploaded leaf image as either Cotton or Rice. The model was trained specifically on this binary classification task and outputs softmax probabilities over the two crop classes.

Stage 2 routes the image to the corresponding crop-specific disease model, implemented using the EfficientNet family of architectures in TensorFlow and Keras. The Cotton model identifies seven disease categories and the Rice model identifies eight. Both models accept 224x224 RGB images normalised to the range zero to one.

Before inference, every uploaded image is assessed for blur using the Laplacian variance method. Images falling below the configured threshold are enhanced using OpenCV's detail enhancement filter. The pipeline returns the top-three predicted classes with confidence percentages, a confidence gate that warns the user if the highest probability is below forty percent, and AI-generated treatment advice from the Mistral Large language model structured as disease description, immediate treatment steps, and prevention tips.

### Fertilizer Recommendation

This module accepts soil sensor readings and crop metadata submitted through a web form and predicts the most appropriate fertilizer using an XGBoost classifier. The model is bundled with its associated label encoder, standard scaler, and categorical feature encoders in a single pickle file, allowing the complete preprocessing and prediction pipeline to be executed in one call.

The prediction result is passed to the Mistral Large language model, which returns a structured JSON response containing the localised fertilizer name, a one-sentence agronomist remark, and formatted advisory sections covering key benefits, usage instructions, and safety precautions. The service includes JSON parsing with a fallback to delimiter-based text extraction if the LLM response cannot be parsed.

Supported crops: Rice, Wheat, Cotton.  
Supported soil types: Loamy, Peaty, Neutral, Acidic.  
Input parameters: Nitrogen, Phosphorous, Potassium, Temperature, Humidity, Moisture, Soil Type, Crop Type.

### Agricultural Chatbot

The chatbot exposes a direct interface to the Mistral Large language model with a system prompt configured for agricultural advisory responses. It enforces structured Markdown output with bold headers and flat bullet points for readability. The language parameter switches the system prompt between English and Urdu modes, with Urdu responses specifically constrained to use simple, conversational vocabulary accessible to non-literate users.

### Text-to-Speech

A text-to-speech endpoint converts any text string to audio using the Google Text-to-Speech library. Long texts are split on sentence boundaries and processed in parallel using a thread pool executor, then concatenated into a single MP3 file. Responses are cached on disk using an MD5 hash of the language code and cleaned text, preventing repeated API calls for identical content.

---

## System Architecture

The application follows a blueprint-based modular structure with singleton service instances. All models are loaded once at application startup inside their respective service constructors, and a warm-up pass on a dummy tensor is executed immediately after loading to eliminate JIT-compilation overhead on the first real request.

```
Browser
    |
    v
Flask Application (app.py)
    |
    |--- GET  /                          Home page
    |--- GET  /disease                   Disease detection page
    |--- GET  /fertilizer                Fertilizer recommendation page
    |
    |--- POST /api/disease/detect
    |         |
    |         |-- Image quality check (Laplacian blur detection)
    |         |-- Optional: detail enhancement (OpenCV)
    |         |-- Stage 1: DenseNet-121 (PyTorch) -> crop type
    |         |-- Stage 2: EfficientNet (Keras/TF) -> disease class
    |         |-- Confidence gate (threshold: 40%)
    |         |-- Mistral LLM -> treatment advice
    |         |-- Returns JSON: crop, disease, confidence, top3, advice, note
    |
    |--- POST /api/fertilizer/predict
    |         |
    |         |-- XGBoost inference (pkl bundle)
    |         |-- Mistral LLM -> structured JSON advice
    |         |-- Returns JSON: fertilizer, remarks, advice
    |
    |--- POST /api/chat/ask
    |         |
    |         |-- Mistral LLM (agricultural system prompt)
    |         |-- Returns JSON: response
    |
    |--- GET  /api/tts
              |
              |-- MD5 cache lookup
              |-- Parallel sentence TTS (gTTS + ThreadPoolExecutor)
              |-- Returns: MP3 audio stream
```

---

## Technology Stack

| Layer | Technology |
| --- | --- |
| Web Framework | Flask, Flask-CORS |
| Stage 1 Disease Model | PyTorch (DenseNet-121, custom implementation) |
| Stage 2 Disease Models | TensorFlow 2.16+, Keras (EfficientNet family) |
| Fertilizer Model | XGBoost, scikit-learn |
| Image Processing | OpenCV, Pillow, Albumentations |
| Language Model | Mistral AI (mistral-large-latest) |
| Text-to-Speech | gTTS |
| Serialisation | joblib, pickle |
| Data Processing | NumPy, Pandas |
| Reporting | Matplotlib, Seaborn, openpyxl |
| Frontend | Jinja2 templates, HTML, Vanilla CSS and JavaScript |

---

## Project Structure

```
fyp_final/
|
|-- app.py                              Entry point. Registers blueprints, exposes TTS endpoint.
|-- config.py                           Centralised configuration: model paths, API keys, limits.
|-- requirements.txt                    Python package dependencies.
|-- .gitignore                          Excludes datasets, uploads, cache, virtual environments.
|
|-- core/
|   |-- utils.py                        get_llm_response() wrapper for Mistral API calls.
|
|-- chatbot/
|   |-- __init__.py
|   |-- routes.py                       POST /api/chat/ask blueprint.
|
|-- disease_detection/
|   |-- __init__.py
|   |-- routes.py                       POST /api/disease/detect blueprint.
|   |-- services.py                     DiseaseService class. Full two-stage inference pipeline.
|   |-- augmenter_offline.py            Albumentations offline augmentation script.
|   |-- train_crop.py                   Stage 1 training script (DenseNet binary classifier).
|   |-- train_disease.py                Stage 2 training script (multi-backbone, multi-crop).
|   |-- test_targeting.py               Unit tests for crop targeting logic.
|   |-- dataset_description.txt        Dataset class counts and augmentation statistics.
|   |
|   |-- models/
|       |-- crop_densenet.py            DenseNet-121 architecture definition and load_densenet().
|       |-- densenet_cotton_rice_classifier.pth    Stage 1 weights (28 MB, PyTorch).
|       |-- cotton_disease_model.keras             Stage 2 Cotton model (42 MB, Keras).
|       |-- cotton_model_labels.pkl               Cotton class label list.
|       |-- rice_disease_model.keras               Stage 2 Rice model (42 MB, Keras).
|       |-- rice_model_labels.pkl                 Rice class label list.
|
|-- fertilizer_recommendation/
|   |-- __init__.py
|   |-- routes.py                       GET /api/fertilizer/config, POST /api/fertilizer/predict.
|   |-- services.py                     FertilizerService class. XGBoost inference and LLM advice.
|   |-- fertilizer_recommendation_dataset.csv     Training data (786 KB).
|   |-- Fertilizer_recommendation_preprocessing_and_model_train.ipynb    Training notebook.
|   |
|   |-- models/
|       |-- fertilizer_model_bundle.pkl            XGBoost model + encoders + scaler.
|
|-- scripts/
|   |-- train_disease.py                CLI training script (mirrors disease_detection/train_disease.py).
|   |-- evaluate_disease.py             Evaluate saved models on a test directory. Outputs metrics and plots.
|   |-- infer_disease.py                Standalone CLI inference engine. Supports single image and batch mode.
|
|-- notebooks/
|   |-- agrinova_disease_training.ipynb    Jupyter notebook for disease model training.
|
|-- reports/
|   |-- agrinova_model_comparison.csv     Accuracy, F1, AUC-ROC across all backbones and crops.
|   |-- agrinova_model_comparison.xlsx    Same data in Excel format.
|   |-- Cotton_*/Rice_* .png              Confusion matrices and training curves per backbone.
|
|-- templates/
|   |-- base.html                       Base Jinja2 layout (navigation, shared styles).
|   |-- index.html                      Home page template.
|   |-- disease.html                    Disease detection UI with image upload.
|   |-- fertilizer.html                 Fertilizer recommendation form and result display.
|
|-- static/
    |-- style.css                       Global stylesheet.
    |-- logo.png, hero.png              Branding images.
    |-- cotton_*.png, rice_*.png        Condition-specific imagery used in the fertilizer UI.
    |-- soil_*.png, ph_*.png, npk_*.png Contextual visual assets for fertilizer UI states.
```

---

## Machine Learning Models

### Stage 1 — Crop Classifier (DenseNet-121, PyTorch)

The architecture is a hand-implemented DenseNet-121 consisting of four dense blocks with configurations of 6, 12, 24, and 16 layers respectively, separated by transition blocks that halve channel count and spatial resolution. The classifier head is a single linear layer mapping 1024 features to 2 output logits.

| Property | Value |
| --- | --- |
| Architecture | DenseNet-121 |
| Framework | PyTorch |
| Task | Binary classification: Cotton vs Rice |
| Input size | 224 x 224, RGB, ImageNet-normalised |
| Output | Softmax probabilities over 2 classes |
| File | densenet_cotton_rice_classifier.pth (28 MB) |

### Stage 2 — Disease Models (EfficientNet, TensorFlow/Keras)

Both models were trained using a two-phase fine-tuning strategy: the backbone is frozen in the first phase to train only the classification head, then the top 40 layers are unfrozen for joint fine-tuning with a reduced learning rate.

**Cotton Disease Model**

| Class | Original Images | Augmented Images |
| --- | --- | --- |
| Bacterial Blight | 250 | 1750 |
| Curl Virus | 431 | 3017 |
| Healthy Leaf | 257 | 1799 |
| Herbicide Growth Damage | 280 | 1960 |
| Leaf Hopper Jassids | 225 | 1575 |
| Leaf Redding | 578 | 4046 |
| Leaf Variegation | 116 | 812 |
| Total | 2137 | 14959 |

**Rice Disease Model**

| Class | Original Images | Augmented Images |
| --- | --- | --- |
| Bacterial Leaf Blight | 180 | 1260 |
| Brown Spot | 267 | 1869 |
| Healthy Rice Leaf | 157 | 1099 |
| Leaf Blast | 305 | 2135 |
| Leaf Scald | 189 | 1323 |
| Narrow Brown Leaf Spot | 117 | 819 |
| Rice Hispa | 215 | 1505 |
| Sheath Blight | 271 | 1897 |
| Total | 1701 | 11907 |

**Training hyperparameters**

| Parameter | Value |
| --- | --- |
| Image size | 224 x 224 |
| Batch size | 32 |
| Phase 1 epochs | 20 |
| Phase 2 epochs | 15 |
| Unfrozen layers (phase 2) | 40 |
| Random seed | 42 |

**Backbones evaluated**: EfficientNetV2B0, EfficientNetV2B3, EfficientNetV2M, EfficientNetV2L, MobileNetV3Large, ResNet50, DenseNet121.

### Fertilizer Model (XGBoost)

| Property | Value |
| --- | --- |
| Algorithm | XGBoost Classifier |
| Features | Nitrogen, Phosphorous, Potassium, Temperature, Humidity, Moisture, Soil Type, Crop Type |
| Bundle contains | Trained model, LabelEncoder, StandardScaler, per-feature categorical encoders |
| File | fertilizer_model_bundle.pkl |

---

## Dataset Information

### Disease Detection Datasets

The disease detection models were trained on two publicly available datasets from Mendeley Data.

| Dataset | Source | Classes | Original Images |
| --- | --- | --- | --- |
| Rice Leaf Disease | Mendeley Data | 8 | 1701 |
| Cotton Leaf Disease | Mendeley Data | 7 | 2137 |

Offline augmentation was applied using four deterministic transforms per image: horizontal flip, random rotation (up to 30 degrees), Gaussian blur, and random brightness and contrast adjustment. This expanded the total dataset to 30,704 images. Original images were kept strictly isolated for test-set evaluation.

Expected directory layout after downloading:

```
disease_detection/Rice_Cotton_disease_Data/
|-- Rice_Disease_Data/Rice/
|   |-- Bacterial_Leaf_Blight/
|   |-- Brown_Spot/
|   |-- Healthy_Rice_Leaf/
|   |-- Leaf_Blast/
|   |-- Leaf_scald/
|   |-- Narrow_Brown_Leaf_Spot/
|   |-- Rice_Hispa/
|   |-- Sheath_Blight/
|
|-- Cotton_Disease_Data/Cotton/
    |-- Bacterial_Blight/
    |-- Curl_Virus/
    |-- Healthy_Leaf/
    |-- Herbicide_Growth_Damage/
    |-- Leaf_Hopper_Jassids/
    |-- Leaf_Redding/
    |-- Leaf_Variegation/
```

### Fertilizer Dataset

The fertilizer training dataset is included in the repository at `fertilizer_recommendation/fertilizer_recommendation_dataset.csv` (786 KB). It contains labeled records with NPK values, environmental conditions, soil type, crop type, and the corresponding fertilizer label.

---

## API Reference

### POST /api/disease/detect

Accepts a multipart form upload containing a leaf image and an optional language code. Returns crop type, predicted disease, confidence score, top-three alternatives, AI-generated treatment advice, and optional processing or warning notes.

Request fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| file | image | Yes | Leaf photograph (JPG or PNG, maximum 16 MB) |
| language | string | No | Response language: en (default) or ur |

Success response:

```json
{
  "crop": "Rice",
  "disease": "Leaf Blast",
  "confidence": "97.34%",
  "top3": [
    { "class": "Leaf Blast",    "confidence": "97.3%" },
    { "class": "Brown Spot",    "confidence": "1.9%"  },
    { "class": "Sheath Blight", "confidence": "0.5%"  }
  ],
  "advice": "### 1. Disease Description\n...",
  "note": "Image quality enhanced (blur score: 45.2)"
}
```

Low-confidence response (confidence below 40%):

```json
{
  "crop": "Cotton",
  "disease": "Leaf Variegation",
  "confidence": "38.10%",
  "warning": "Low confidence (38.1%). Please upload a clearer, closer image of the leaf.",
  "advice": "..."
}
```

---

### GET /api/fertilizer/config

Returns the list of accepted features and valid categorical options.

```json
{
  "features": ["Nitrogen", "Phosphorous", "Potassium", "Temperature", "Humidity", "Moisture", "Soil Type", "Crop Type"],
  "num_cols": ["Nitrogen", "Phosphorous", "Potassium", "Temperature", "Humidity", "Moisture"],
  "cat_cols": ["Soil Type", "Crop Type"],
  "options": {
    "Crop Type": ["Rice", "Wheat", "Cotton"],
    "Soil Type": ["Loamy Soil", "Peaty Soil", "Neutral Soil", "Acidic Soil"]
  }
}
```

---

### POST /api/fertilizer/predict

Accepts a JSON body with soil and crop parameters. Returns fertilizer name, agronomist remark, and structured advice.

Request body:

```json
{
  "Nitrogen": 50,
  "Phosphorous": 30,
  "Potassium": 40,
  "Temperature": 28,
  "Humidity": 65,
  "Moisture": 20,
  "Soil Type": "Loamy Soil",
  "Crop Type": "Rice",
  "language": "en"
}
```

Response:

```json
{
  "fertilizer": "Urea",
  "remarks": "Urea supplies nitrogen which is critical for vegetative growth.",
  "advice": "## Key Benefits\n- ...\n## Usage Instructions\n- ...\n## Safety Precautions\n- ..."
}
```

---

### POST /api/chat/ask

Accepts a JSON body with a user query and optional language code. Returns a Markdown-formatted agricultural advisory response.

Request body:

```json
{
  "query": "What is the best time to sow wheat in Pakistan?",
  "language": "en"
}
```

Response:

```json
{
  "response": "**Best Sowing Time for Wheat**\n- ..."
}
```

---

### GET /api/tts

Converts text to speech and returns an MP3 audio stream. Responses are cached on disk.

Query parameters:

| Parameter | Description |
| --- | --- |
| text | The text to synthesise |
| lang | Language code, for example ur or en |

---

### GET /api/health

Returns application status and registered module names.

```json
{
  "status": "AgriNova Running",
  "modules": ["Fertilizer", "Disease", "Chatbot"]
}
```

---

## Installation and Setup

### Prerequisites

- Python 3.9 or later
- pip
- A Mistral AI API key (obtain from console.mistral.ai)

### Step 1 — Clone the Repository

```bash
git clone https://github.com/shahzaibsamo12/FYP-AgriNova.git
cd FYP-AgriNova
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_key_here
SECRET_KEY=your_flask_secret_key
```

The default API key embedded in `config.py` is for development only and should be replaced for production use.

### Step 5 — Run the Application

```bash
python app.py
```

The application starts on `http://localhost:5000`. The three pre-trained models load into memory during startup. Startup time is approximately 10 to 20 seconds depending on hardware.

---

## Training the Models

The pre-trained models are included in the repository. These steps are only required if you wish to retrain or experiment with different architectures.

### Step 1 — Download the Datasets

Download the Rice and Cotton disease datasets from Mendeley Data and place them under the path described in the Dataset Information section above.

### Step 2 — Generate Augmented Data

```bash
cd disease_detection
python augmenter_offline.py
```

This applies four deterministic transforms to every original image and writes the results to `disease_detection/AGRINOVA_DATASET/`. Total output is approximately 30,000 images.

### Step 3 — Train Stage 2 Disease Models

```bash
# Train all backbones for both crops
python disease_detection/train_disease.py

# Train a specific crop only
python disease_detection/train_disease.py --crop rice
python disease_detection/train_disease.py --crop cotton

# Train a specific backbone only
python disease_detection/train_disease.py --backbone EfficientNetV2B0

# Use original images only (no augmentation)
python disease_detection/train_disease.py --no-aug
```

Output models are saved to `disease_detection/models/`. Training reports including confusion matrices and accuracy curves are saved to `reports/`.

### Step 4 — Evaluate Saved Models

```bash
python scripts/evaluate_disease.py --crop rice
python scripts/evaluate_disease.py --crop cotton
python scripts/evaluate_disease.py --all
```

### Step 5 — Run CLI Inference

```bash
# Single image
python scripts/infer_disease.py --image path/to/leaf.jpg

# Force crop class (skip Stage 1)
python scripts/infer_disease.py --image path/to/leaf.jpg --crop rice

# Batch inference on a folder
python scripts/infer_disease.py --batch path/to/folder/
```

---

## Configuration

All configurable values are centralised in `config.py`.

| Setting | Default | Description |
| --- | --- | --- |
| MISTRAL_API_KEY | Environment variable | API key for Mistral AI |
| MISTRAL_ENDPOINT | https://api.mistral.ai/v1/chat/completions | LLM inference endpoint |
| FERTILIZER_MODEL_PATH | fertilizer_recommendation/models/fertilizer_model_bundle.pkl | XGBoost bundle |
| CROP_MODEL_PATH | disease_detection/models/densenet_cotton_rice_classifier.pth | Stage 1 weights |
| RICE_MODEL_PATH | disease_detection/models/rice_disease_model.keras | Stage 2 Rice model |
| COTTON_MODEL_PATH | disease_detection/models/cotton_disease_model.keras | Stage 2 Cotton model |
| UPLOAD_FOLDER | uploads/ | Temporary image storage (cleared after each request) |
| MAX_CONTENT_LENGTH | 16 MB | Maximum upload file size |
| SECRET_KEY | Environment variable | Flask session secret |

---

## Contributors

| Name | Role |
| --- | --- |
| Shahzaib Samo | Machine learning engineering, backend development, full-stack integration |

---

AgriNova — Final Year Project. Developed for the farming community of Pakistan.
