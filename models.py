from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    audio_file: Optional[str] = None  # File path or base64 encoded audio

class TextAnalysisFeatures(BaseModel):
    depression_keywords: int
    mental_health_terms: int
    personal_pronouns: int
    negative_words: int
    sentiment_polarity: float
    word_count: int

class AudioAnalysisFeatures(BaseModel):
    duration: float
    pitch_mean: float
    tempo: float
    energy: float

class Recommendations(BaseModel):
    urgency: str
    immediate_steps: Optional[List[str]] = None
    resources: Optional[List[str]] = None
    maintenance_tips: Optional[List[str]] = None

class AnalysisResponse(BaseModel):
    depression_detected: bool
    confidence: float
    analysis_mode: str
    text_analysis: Optional[TextAnalysisFeatures] = None
    audio_analysis: Optional[AudioAnalysisFeatures] = None
    recommendations: Recommendations

class TrainingStatus(BaseModel):
    status: str  # "idle", "training", "completed", "error"
    progress: int  # 0-100
    message: str

class TrainingResults(BaseModel):
    accuracy: float
    f1_score: float
    auc_roc: float
    training_samples: int
    test_samples: int
    training_completed: str
    classification_report: Dict[str, Any]

class ModelInfo(BaseModel):
    model_type: str
    architecture: str
    total_parameters: int
    training_status: str
    last_trained: Optional[str] = None