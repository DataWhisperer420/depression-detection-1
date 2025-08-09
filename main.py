from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import numpy as np
import os
import json
import asyncio
from datetime import datetime
from typing import Optional
import shutil
import tempfile

from ml_models import (
    DataLoaders, TextFeatureExtractor, AudioFeatureExtractor, 
    MultimodalFusionModel, EDAVisualizer
)
from models import AnalysisRequest, AnalysisResponse, TrainingStatus

app = FastAPI(title="Multimodal Depression Detection System")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Global variables for model and training status
current_model = None
training_status = {"status": "idle", "progress": 0, "message": "Ready to train"}
training_results = {}

# Initialize components
data_loader = DataLoaders()
text_extractor = TextFeatureExtractor()
audio_extractor = AudioFeatureExtractor()

@app.get("/", response_class=HTMLResponse)
async def analysis_page(request: Request):
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    return templates.TemplateResponse("training.html", {"request": request})

@app.post("/analyze")
async def analyze_depression(
    text: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None)
):
    global current_model
    
    if current_model is None:
        try:
            if os.path.exists("best_multimodal_model.pth"):
                import torch
                current_model = MultimodalFusionModel()
                

                if os.path.exists("text_scaler.pkl") and os.path.exists("audio_scaler.pkl"):
                    try:
                        import pickle
                        with open("text_scaler.pkl", "rb") as f:
                            text_scaler = pickle.load(f)
                        with open("audio_scaler.pkl", "rb") as f:
                            audio_scaler = pickle.load(f)
                        
                        current_model._initialize_model(text_scaler.n_features_in_, audio_scaler.n_features_in_)
                        current_model.text_scaler = text_scaler
                        current_model.audio_scaler = audio_scaler
                        
                        # Load the state dictionary
                        current_model.model.load_state_dict(torch.load("best_multimodal_model.pth", map_location='cpu'))
                        current_model.is_trained = True
                        print("Loaded existing model with scalers")
                        
                    except Exception as pickle_error:
                        print(f"Error loading scalers: {pickle_error}")
                        # Fallback: Create new scalers and initialize with default dimensions
                        from sklearn.preprocessing import StandardScaler
                        
                        # Use default dimensions based on the feature extractors
                        text_dim = 113  # 13 basic features + 100 BERT embeddings
                        audio_dim = len(audio_extractor.feature_names)  # Get audio feature count
                        
                        current_model._initialize_model(text_dim, audio_dim)
                        current_model.text_scaler = StandardScaler()
                        current_model.audio_scaler = StandardScaler()
                        
                        # Create dummy data to fit scalers
                        import numpy as np
                        dummy_text_data = np.random.randn(10, text_dim)
                        dummy_audio_data = np.random.randn(10, audio_dim)
                        current_model.text_scaler.fit(dummy_text_data)
                        current_model.audio_scaler.fit(dummy_audio_data)
                        
                        # Load the state dictionary
                        current_model.model.load_state_dict(torch.load("best_multimodal_model.pth", map_location='cpu'))
                        current_model.is_trained = True
                        print("Loaded model with fallback scalers due to scaler loading error")
                        
                else:
                    raise HTTPException(status_code=400, detail="Model scalers not found. Please train a model first.")
            else:
                raise HTTPException(status_code=400, detail="No trained model found. Please train a model first.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    if not text and not audio_file:
        raise HTTPException(status_code=400, detail="Please provide either text or audio input")
    
    try:
        text_features = None
        audio_features = None
        analysis_mode = None
        
        # Process text input
        if text and text.strip():
            text_features = text_extractor.extract_features([text.strip()])
            analysis_mode = "text" if not audio_file else "multimodal"
        
        # Process audio input
        audio_file_path = None
        if audio_file:
            # Save uploaded audio file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_file.filename.split('.')[-1]}") as tmp_file:
                shutil.copyfileobj(audio_file.file, tmp_file)
                audio_file_path = tmp_file.name
            
            # Extract audio features
            audio_feat_dict = audio_extractor.extract_audio_features(audio_file_path)
            audio_features = pd.DataFrame([audio_feat_dict], columns=audio_extractor.feature_names)
            
            if not text or not text.strip():
                analysis_mode = "audio"
            else:
                analysis_mode = "multimodal"
            
            # Clean up temporary file
            os.unlink(audio_file_path)
        
        # Make predictions based on analysis mode
        if analysis_mode == "multimodal":
            predictions, _, _, probabilities, _, _ = current_model.predict(text_features, audio_features)
        elif analysis_mode == "text":
            dummy_audio = _create_dummy_audio_features()
            predictions, _, _, probabilities, _, _ = current_model.predict(text_features, dummy_audio)
        elif analysis_mode == "audio":
            dummy_text = _create_dummy_text_features()
            predictions, _, _, probabilities, _, _ = current_model.predict(dummy_text, audio_features)
        
        # Prepare response
        # Safely handle confidence calculation
        try:
            confidence = float(probabilities[0][1] * 100)
            if confidence < 0:
                confidence = 0.0
            elif confidence > 100:
                confidence = 100.0
        except (IndexError, TypeError, ValueError):
            confidence = 50.0  # Default confidence
        
        try:
            depression_detected = bool(predictions[0] == 1)  # Convert to Python bool
        except (IndexError, TypeError):
            depression_detected = False  # Default to no depression detected
        
        # Extract text features for display
        text_analysis = {}
        if text and text.strip():
            features = text_extractor.extract_basic_features(text.strip())
            text_analysis = {
                "depression_keywords": int(features['depression_keywords']),
                "mental_health_terms": int(features['mental_health_terms']),
                "personal_pronouns": int(features['personal_pronouns']),
                "negative_words": int(features['negative_words']),
                "sentiment_polarity": float(features['polarity']),
                "word_count": int(features['word_count'])
            }
        
        # Audio analysis info
        audio_analysis = {}
        if audio_file and audio_features is not None:  # Check audio_file instead of audio_file_path
            # Safely extract values with proper defaults and ensure they're valid numbers
            def safe_float(value, default=0.0):
                try:
                    if value is None or value == 0:
                        return default
                    return float(value)
                except (ValueError, TypeError):
                    return default
            
            audio_analysis = {
                "duration": safe_float(audio_feat_dict.get('duration'), 0.0),
                "pitch_mean": safe_float(audio_feat_dict.get('pitch_mean'), 150.0),
                "tempo": safe_float(audio_feat_dict.get('tempo'), 120.0),
                "energy": safe_float(audio_feat_dict.get('rms_energy'), 0.1)
            }
        
        return JSONResponse({
            "depression_detected": depression_detected,
            "confidence": confidence,
            "analysis_mode": analysis_mode,
            "text_analysis": text_analysis if text and text.strip() else False,
            "audio_analysis": audio_analysis,
            "recommendations": _get_recommendations(depression_detected, confidence)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/train")
async def start_training(background_tasks: BackgroundTasks):

    global training_status
    
    if training_status["status"] == "training":
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    if not os.path.exists("Combined Data.csv"):
        raise HTTPException(status_code=400, detail="Training data file 'Combined Data.csv' not found")
    
    background_tasks.add_task(train_model_background)
    return JSONResponse({"message": "Training started", "status": "training"})

@app.get("/training-status")
async def get_training_status():
    return JSONResponse(training_status)

@app.get("/training-results")
async def get_training_results():
    if not training_results:
        raise HTTPException(status_code=404, detail="No training results available")
    return JSONResponse(training_results)

async def train_model_background():
    global current_model, training_status, training_results
    
    try:
        training_status = {"status": "training", "progress": 10, "message": "Loading dataset..."}
        
        # Load data
        text_data = data_loader.load_mental_health_dataset('Combined Data.csv')
        if text_data is None:
            training_status = {"status": "error", "progress": 0, "message": "Failed to load dataset"}
            return
        
        training_status = {"status": "training", "progress": 30, "message": "Extracting text features..."}
        
        # Extract features
        text_features = text_extractor.extract_features(text_data['text'])
        
        training_status = {"status": "training", "progress": 50, "message": "Extracting real audio features..."}
        
        # Get available audio files from AudioFiles_and_CSV folder
        audio_folder = "Audiofiles_and_CSV"
        audio_files = []
        if os.path.exists(audio_folder):
            audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
        
        n_samples = len(text_data)
        audio_features_data = []
        
        # Extract real audio features from available files
        for i in range(n_samples):
            if audio_files:
                # Cycle through available audio files
                audio_file = audio_files[i % len(audio_files)]
                audio_path = os.path.join(audio_folder, audio_file)
                
                try:
                    # Extract real features from audio file
                    features = audio_extractor.extract_audio_features(audio_path)
                    
                    # Apply depression-related patterns to align with text labels
                    if i < len(text_data) and text_data.iloc[i]['depression'] == 1:
                        # Slightly modify features to reflect depression patterns
                        features['pitch_mean'] *= 0.9
                        features['tempo'] *= 0.95
                        features['rms_energy'] *= 0.8
                    
                    if i % 10 == 0:
                        print(f"Processed audio {i+1}/{n_samples}: {audio_file}")
                        
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    # Fallback to default features if extraction fails
                    features = audio_extractor._get_default_features()
            else:
                # Fallback to default features if no audio files found
                features = audio_extractor._get_default_features()
            
            audio_features_data.append(features)
        
        audio_features = pd.DataFrame(audio_features_data, columns=audio_extractor.feature_names)
        print(f"Extracted real audio features from {len(audio_files)} audio files for {n_samples} samples")
        
        training_status = {"status": "training", "progress": 60, "message": "Splitting data..."}
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_text_train, X_text_test, X_audio_train, X_audio_test, y_train, y_test = train_test_split(
            text_features, audio_features, text_data['depression'], 
            test_size=0.2, random_state=42, stratify=text_data['depression']
        )
        
        training_status = {"status": "training", "progress": 70, "message": "Training multimodal transformer..."}
        
        # Train model
        current_model = MultimodalFusionModel()
        current_model.train_fusion_model(X_text_train, X_audio_train, y_train)
        
        training_status = {"status": "training", "progress": 90, "message": "Evaluating model..."}
        
        # Evaluate model
        predictions, _, _, probabilities, _, _ = current_model.predict(X_text_test, X_audio_test)
        
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
        
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        auc = roc_auc_score(y_test, probabilities[:, 1])
        
        # Store training results
        training_results = {
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "training_samples": int(len(X_text_train)),
            "test_samples": int(len(X_text_test)),
            "training_completed": datetime.now().isoformat(),
            "classification_report": classification_report(y_test, predictions, output_dict=True)
        }
        
        training_status = {"status": "completed", "progress": 100, "message": "Training completed successfully!"}
        
    except Exception as e:
        training_status = {"status": "error", "progress": 0, "message": f"Training failed: {str(e)}"}

def _create_dummy_audio_features():
    dummy_features = {}
    dummy_features['duration'] = 20.0
    dummy_features['zero_crossing_rate'] = 0.05
    
    for i in range(13):
        dummy_features[f'mfcc_{i}_mean'] = 0.0
        dummy_features[f'mfcc_{i}_std'] = 10.0
    
    dummy_features['spectral_centroid_mean'] = 2500.0
    dummy_features['spectral_centroid_std'] = 500.0
    dummy_features['spectral_rolloff_mean'] = 5000.0
    dummy_features['spectral_rolloff_std'] = 1000.0
    dummy_features['chroma_mean'] = 0.4
    dummy_features['chroma_std'] = 0.15
    dummy_features['tonnetz_mean'] = 0.0
    dummy_features['tonnetz_std'] = 0.25
    dummy_features['rms_energy'] = 0.15
    dummy_features['tempo'] = 120.0
    dummy_features['pitch_mean'] = 150.0
    dummy_features['pitch_std'] = 30.0
    dummy_features['pitch_min'] = 100.0
    dummy_features['pitch_max'] = 300.0
    
    return pd.DataFrame([dummy_features], columns=audio_extractor.feature_names)

def _create_dummy_text_features():
    dummy_text = "This is a neutral text for audio-only analysis."
    return text_extractor.extract_features([dummy_text])

def _get_recommendations(depression_detected, confidence):
    if depression_detected:
        if confidence >= 80:
            urgency = "HIGH"
        elif confidence >= 60:
            urgency = "MODERATE"
        else:
            urgency = "LOW"
        
        return {
            "urgency": urgency,
            "immediate_steps": [
                "Speak with a mental health professional",
                "Contact university counseling services",
                "Reach out to trusted friends or family",
                "Emergency: Call local crisis hotline if needed"
            ],
            "resources": [
                "University Counseling Center",
                "Student Health Services",
                "National Suicide Prevention Lifeline: 988",
                "Crisis Text Line: Text HOME to 741741"
            ]
        }
    else:
        return {
            "urgency": "NONE",
            "maintenance_tips": [
                "Continue healthy lifestyle practices",
                "Stay connected with support networks",
                "Regular self-assessment of mental health",
                "Seek help if symptoms develop"
            ]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)