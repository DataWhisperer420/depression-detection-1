import pandas as pd
import numpy as np
import librosa
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                            roc_curve, roc_auc_score, precision_recall_curve, f1_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from textblob import TextBlob
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from collections import Counter
from wordcloud import WordCloud
import os

WORDCLOUD_AVAILABLE = True
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class EDAVisualizer:
    
    def __init__(self):
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        
    def analyze_dataset(self, df):
        print("EXPLORATORY DATA ANALYSIS")
        
        print(f"\nDataset Overview:")
        print(f"Total samples: {len(df):,}")
        print(f"Features: {df.columns.tolist()}")
        
        # Class distribution
        print(f"\nClass Distribution:")
        status_counts = df['status'].value_counts()
        depression_counts = df['depression'].value_counts()
        
        for status, count in status_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  - {status.title()}: {count:,} samples ({percentage:.1f}%)")
        
        print(f"\nBinary Classification:")
        print(f"  - Depression: {depression_counts[1]:,} samples ({(depression_counts[1]/len(df)*100):.1f}%)")
        print(f"  - Non-Depression: {depression_counts[0]:,} samples ({(depression_counts[0]/len(df)*100):.1f}%)")
        
        self._analyze_text_features(df)
        self._create_visualizations(df)
        
    def _analyze_text_features(self, df):
        print(f"\nText Analysis:")
        
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['sentence_count'] = df['text'].str.count(r'[.!?]+')
        
        print(f"Average text length: {df['text_length'].mean():.1f} characters")
        print(f"Average word count: {df['word_count'].mean():.1f} words")
        print(f"Average sentences: {df['sentence_count'].mean():.1f} sentences")
        
        # Depression vs Non-depression comparison
        dep_stats = df[df['depression'] == 1]
        non_dep_stats = df[df['depression'] == 0]
        
        print(f"\nDepression vs Non-Depression Text Patterns:")
        print(f"Depression texts - Avg length: {dep_stats['text_length'].mean():.1f} chars, Avg words: {dep_stats['word_count'].mean():.1f}")
        print(f"Non-Depression texts - Avg length: {non_dep_stats['text_length'].mean():.1f} chars, Avg words: {non_dep_stats['word_count'].mean():.1f}")
        
    def _create_visualizations(self, df):
        try:
            print("Creating EDA visualizations")
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Mental Health Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')
            
            # 1. Class Distribution (Status)
            status_counts = df['status'].value_counts()
            colors = plt.cm.Set3(np.linspace(0, 1, len(status_counts)))
            axes[0, 0].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
                          colors=colors, startangle=90)
            axes[0, 0].set_title('Distribution by Mental Health Status', fontweight='bold')
            
            # 2. Binary Classification
            binary_counts = df['depression'].value_counts()
            colors_binary = ['#7fbf7f', '#ff7f7f'] 
            axes[0, 1].bar(['Non-Depression', 'Depression'], 
                          [binary_counts[0], binary_counts[1]], 
                          color=colors_binary, alpha=0.7)
            axes[0, 1].set_title('Binary Classification Distribution', fontweight='bold')
            axes[0, 1].set_ylabel('Number of Samples')
            for i, v in enumerate([binary_counts[0], binary_counts[1]]):
                axes[0, 1].text(i, v + 10, f'{v:,}', ha='center', fontweight='bold')
            
            # 3. Text Length Distribution
            if 'text_length' not in df.columns:
                df['text_length'] = df['text'].str.len()
            axes[0, 2].hist(df[df['depression'] == 0]['text_length'], alpha=0.7, 
                           label='Non-Depression', bins=30, color='green')
            axes[0, 2].hist(df[df['depression'] == 1]['text_length'], alpha=0.7, 
                           label='Depression', bins=30, color='red')
            axes[0, 2].set_title('Text Length Distribution', fontweight='bold')
            axes[0, 2].set_xlabel('Text Length (characters)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
            
            # 4. Word Count Distribution
            if 'word_count' not in df.columns:
                df['word_count'] = df['text'].str.split().str.len()
            axes[1, 0].boxplot([df[df['depression'] == 0]['word_count'].dropna(), 
                               df[df['depression'] == 1]['word_count'].dropna()], 
                              labels=['Non-Depression', 'Depression'])
            axes[1, 0].set_title('Word Count Distribution', fontweight='bold')
            axes[1, 0].set_ylabel('Number of Words')
            
            # 5. Status vs Depression Heatmap
            cross_tab = pd.crosstab(df['status'], df['depression'])
            sns.heatmap(cross_tab, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
            axes[1, 1].set_title('Status vs Depression Cross-tabulation', fontweight='bold')
            axes[1, 1].set_xlabel('Depression (0=No, 1=Yes)')
            
            # 6. Text Length by Category
            unique_statuses = df['status'].unique()
            positions = range(len(unique_statuses))
            violin_data = [df[df['status'] == status]['text_length'].values for status in unique_statuses]
            
            if len(violin_data) > 0 and all(len(data) > 0 for data in violin_data):
                axes[1, 2].violinplot(violin_data, positions=positions)
                axes[1, 2].set_xticks(positions)
                axes[1, 2].set_xticklabels(unique_statuses, rotation=45)
                axes[1, 2].set_title('Text Length Distribution by Status', fontweight='bold')
                axes[1, 2].set_ylabel('Text Length (characters)')
            else:
                axes[1, 2].text(0.5, 0.5, 'Insufficient data\nfor violin plot', 
                               ha='center', va='center', transform=axes[1, 2].transAxes)
                axes[1, 2].set_title('Text Length Distribution by Status', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('static/eda_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("EDA visualizations saved as 'static/eda_analysis.png'")
            
            # Create word clouds
            self._create_wordclouds(df)
            
        except Exception as e:
            print(f"Error creating EDA visualizations: {e}")
            print("Continuing without EDA visualizations...")
        
    def _create_wordclouds(self, df):
        if not WORDCLOUD_AVAILABLE:
            print("WordCloud library not available. Skipping word cloud generation.")
            return
            
        try:
            print("Creating word clouds")
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Depression word cloud
            depression_text = ' '.join(df[df['depression'] == 1]['text'].astype(str))
            if depression_text.strip():
                wordcloud_dep = WordCloud(width=800, height=400, 
                                         background_color='white',
                                         colormap='Reds').generate(depression_text)
                axes[0].imshow(wordcloud_dep, interpolation='bilinear')
                axes[0].set_title('Depression-Related Texts Word Cloud', fontweight='bold', fontsize=14)
                axes[0].axis('off')
            
            # Non-depression word cloud
            non_depression_text = ' '.join(df[df['depression'] == 0]['text'].astype(str))
            if non_depression_text.strip():
                wordcloud_non_dep = WordCloud(width=800, height=400, 
                                             background_color='white',
                                             colormap='Greens').generate(non_depression_text)
                axes[1].imshow(wordcloud_non_dep, interpolation='bilinear')
                axes[1].set_title('Non-Depression Texts Word Cloud', fontweight='bold', fontsize=14)
                axes[1].axis('off')
            
            plt.tight_layout()
            plt.savefig('static/wordclouds.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Word clouds saved as 'static/wordclouds.png'")
            
        except Exception as e:
            print(f"Word cloud generation failed: {e}")

class DataLoaders:
    
    def __init__(self):
        self.text_data = None
        self.audio_features = None
        self.eda_visualizer = EDAVisualizer()
    
    def load_mental_health_dataset(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df = df.dropna(subset=['statement', 'status'])
            df['status'] = df['status'].str.lower().str.strip()
            
            print("Unique mental health statuses in dataset:")
            unique_statuses = df['status'].unique()
            for status in sorted(unique_statuses):
                count = (df['status'] == status).sum()
                print(f"  - {status}: {count} samples")
            
            print("\nSampling data from each category for faster training...")
            sampled_dfs = []
            
            for status in unique_statuses:
                status_df = df[df['status'] == status]
                sample_size = max(1, len(status_df) // 32)
                sampled_status_df = status_df.sample(n=sample_size, random_state=42)
                sampled_dfs.append(sampled_status_df)
                print(f"  - {status}: {len(status_df)} → {len(sampled_status_df)} samples")
            
            df_sampled = pd.concat(sampled_dfs, ignore_index=True)
            df_sampled['depression'] = (df_sampled['status'] == 'depression').astype(int)
            df_sampled['text'] = df_sampled['statement']
            
            final_df = df_sampled[['text', 'depression', 'status']].copy()
            final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"\nFinal sampled dataset for depression detection:")
            print(f"Total samples: {len(final_df)} (reduced from {len(df)})")
            print(f"Depression samples: {final_df['depression'].sum()}")
            print(f"Non-depression samples: {len(final_df) - final_df['depression'].sum()}")
            
            print("\nPerforming Exploratory Data Analysis...")
            self.eda_visualizer.analyze_dataset(final_df)
            
            return final_df
            
        except Exception as e:
            print(f"Error loading mental health dataset: {e}")
            return None

class TextFeatureExtractor:
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.load_bert_model()
        
    def load_bert_model(self):
        try:
            model_name = 'bert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except:
            print("BERT model not available, using basic features only")
    
    def extract_basic_features(self, text):
        features = {}
        
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(text.split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        blob = TextBlob(text)
        features['polarity'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity
        
        depression_keywords = [
            'sad', 'depressed', 'hopeless', 'worthless', 'tired', 'empty',
            'lonely', 'anxious', 'worried', 'stressed', 'overwhelmed',
            'suicide', 'death', 'pain', 'hurt', 'cry', 'sleep', 'insomnia',
            'restless', 'nervous', 'scared', 'afraid', 'panic', 'anxiety',
            'fear', 'worry', 'stress', 'pressure', 'burden', 'failure',
            'broken', 'lost', 'confused', 'helpless', 'struggling', 'difficult'
        ]
        
        text_lower = text.lower()
        features['depression_keywords'] = sum(1 for keyword in depression_keywords if keyword in text_lower)
        
        mental_health_terms = [
            'mental', 'therapy', 'counseling', 'psychiatrist', 'psychologist',
            'medication', 'treatment', 'disorder', 'illness', 'condition'
        ]
        features['mental_health_terms'] = sum(1 for term in mental_health_terms if term in text_lower)
        
        personal_pronouns = ['i', 'me', 'my', 'myself', 'mine']
        features['personal_pronouns'] = sum(1 for pronoun in personal_pronouns if pronoun in text_lower.split())
        
        negative_words = [
            'no', 'not', 'never', 'nothing', 'nobody', 'none', 'cannot', 
            "can't", "won't", "don't", "didn't", "isn't", "aren't", "wasn't", "weren't"
        ]
        features['negative_words'] = sum(1 for neg in negative_words if neg in text_lower.split())
        
        intensity_words = ['very', 'extremely', 'really', 'so', 'too', 'quite', 'rather']
        features['intensity_words'] = sum(1 for word in intensity_words if word in text_lower.split())
        
        return features
    
    def get_bert_embeddings(self, text):
        if self.tokenizer and self.model:
            try:
                inputs = self.tokenizer(text[:512], return_tensors='pt', truncation=True, 
                                      padding=True, max_length=256)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()[:100]
                    if len(embeddings) < 100:
                        embeddings = np.pad(embeddings, (0, 100 - len(embeddings)))
                return embeddings
            except:
                return np.zeros(100)
        else:
            return np.zeros(100)
    
    def extract_features(self, texts):
        all_features = []
        
        print(f"Extracting features from {len(texts)} texts...")
        
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"  Processing text {i+1}/{len(texts)}")
                
            features = self.extract_basic_features(text)
            
            bert_embeddings = self.get_bert_embeddings(text)
            for j, emb in enumerate(bert_embeddings):
                features[f'bert_{j}'] = emb
            
            all_features.append(features)
        
        print("Feature extraction completed!")
        return pd.DataFrame(all_features)

class AudioFeatureExtractor:
    
    def __init__(self):
        self.sample_rate = 22050
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self):
        features = []
        features.extend(['duration', 'zero_crossing_rate'])
        
        for i in range(13):
            features.append(f'mfcc_{i}_mean')
            features.append(f'mfcc_{i}_std')
        
        features.extend([
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_rolloff_mean', 'spectral_rolloff_std'
        ])
        
        features.extend(['chroma_mean', 'chroma_std'])
        features.extend(['tonnetz_mean', 'tonnetz_std'])
        features.extend(['rms_energy', 'tempo'])
        features.extend(['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max'])
        
        return features
    
    def _get_default_features(self):
        return {feature: 0.0 for feature in self.feature_names}
        
    def extract_audio_features(self, audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, duration=30)
            features = self._get_default_features()
            
            features['duration'] = len(y) / sr
            features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
            
            try:
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                for i in range(13):
                    features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                    features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
            except Exception as e:
                print(f"MFCC extraction failed: {e}")
            
            try:
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
                features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
                features['spectral_centroid_std'] = float(np.std(spectral_centroids))
                
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
                features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
            except Exception as e:
                print(f"Spectral feature extraction failed: {e}")
            
            try:
                chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                features['chroma_mean'] = float(np.mean(chroma))
                features['chroma_std'] = float(np.std(chroma))
            except Exception as e:
                print(f"Chroma extraction failed: {e}")
            
            try:
                tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                features['tonnetz_mean'] = float(np.mean(tonnetz))
                features['tonnetz_std'] = float(np.std(tonnetz))
            except Exception as e:
                print(f"Tonnetz extraction failed: {e}")
            
            try:
                features['rms_energy'] = float(np.mean(librosa.feature.rms(y=y)))
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features['tempo'] = float(tempo)
            except Exception as e:
                print(f"Energy/tempo extraction failed: {e}")
            
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitch_values = []
                for t in range(pitches.shape[1]):
                    index = magnitudes[:, t].argmax()
                    pitch = pitches[index, t]
                    if pitch > 0:
                        pitch_values.append(pitch)
                
                if pitch_values:
                    features['pitch_mean'] = float(np.mean(pitch_values))
                    features['pitch_std'] = float(np.std(pitch_values))
                    features['pitch_min'] = float(np.min(pitch_values))
                    features['pitch_max'] = float(np.max(pitch_values))
            except Exception as e:
                print(f"Pitch extraction failed: {e}")
            
            ordered_features = {name: features[name] for name in self.feature_names}
            return ordered_features
            
        except Exception as e:
            print(f"Error extracting audio features: {e}")
            return self._get_default_features()

class MultimodalTransformer(nn.Module):

    
    def __init__(self, text_feature_dim, audio_feature_dim, hidden_dim=256, 
                 num_heads=8, num_layers=4, dropout=0.1):
        super(MultimodalTransformer, self).__init__()
        
        self.text_feature_dim = text_feature_dim
        self.audio_feature_dim = audio_feature_dim
        self.hidden_dim = hidden_dim
        
        # Text Feature Encoder
        self.text_encoder = nn.Sequential(
            nn.Linear(text_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Audio Feature Encoder  
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Positional encoding for sequence modeling
        self.text_pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.audio_pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Multi-head Self-Attention layers
        self.text_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        self.audio_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-modal attention layers
        self.cross_attention_text_to_audio = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.cross_attention_audio_to_text = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.text_norm1 = nn.LayerNorm(hidden_dim)
        self.text_norm2 = nn.LayerNorm(hidden_dim)
        self.audio_norm1 = nn.LayerNorm(hidden_dim)
        self.audio_norm2 = nn.LayerNorm(hidden_dim)
        
        # Fusion mechanism
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenated features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, text_features, audio_features):
        batch_size = text_features.size(0)
        
        # Encode text and audio features
        text_encoded = self.text_encoder(text_features)  # [batch_size, hidden_dim]
        audio_encoded = self.audio_encoder(audio_features)  # [batch_size, hidden_dim]
        
        # Add positional encoding and expand dimensions for attention
        text_encoded = text_encoded.unsqueeze(1) + self.text_pos_encoding  # [batch_size, 1, hidden_dim]
        audio_encoded = audio_encoded.unsqueeze(1) + self.audio_pos_encoding  # [batch_size, 1, hidden_dim]
        
        # Self-attention for text features
        text_attended, _ = self.text_attention(text_encoded, text_encoded, text_encoded)
        text_encoded = self.text_norm1(text_encoded + text_attended)
        
        # Self-attention for audio features  
        audio_attended, _ = self.audio_attention(audio_encoded, audio_encoded, audio_encoded)
        audio_encoded = self.audio_norm1(audio_encoded + audio_attended)
        
        # Cross-modal attention
        # Text attending to audio
        text_cross_attended, text_attention_weights = self.cross_attention_text_to_audio(
            text_encoded, audio_encoded, audio_encoded
        )
        text_encoded = self.text_norm2(text_encoded + text_cross_attended)
        
        # Audio attending to text
        audio_cross_attended, audio_attention_weights = self.cross_attention_audio_to_text(
            audio_encoded, text_encoded, text_encoded
        )
        audio_encoded = self.audio_norm2(audio_encoded + audio_cross_attended)
        
        # Concatenate modalities for fusion
        fused_features = torch.cat([text_encoded, audio_encoded], dim=1)  # [batch_size, 2, hidden_dim]
        
        # Apply transformer encoder to fused features
        transformer_output = self.transformer_encoder(fused_features)  # [batch_size, 2, hidden_dim]
        
        # Global average pooling and concatenation for final classification
        text_final = transformer_output[:, 0, :]  # [batch_size, hidden_dim]
        audio_final = transformer_output[:, 1, :]  # [batch_size, hidden_dim]
        
        # Final fusion through concatenation
        final_features = torch.cat([text_final, audio_final], dim=1)  # [batch_size, hidden_dim * 2]
        
        # Classification
        logits = self.classifier(final_features)  # [batch_size, 2]
        
        return logits, {
            'text_attention_weights': text_attention_weights,
            'audio_attention_weights': audio_attention_weights,
            'text_features': text_final,
            'audio_features': audio_final
        }

class MultimodalFusionModel:
    """
    Enhanced Multimodal Fusion Model using Transformer Architecture
    """
    
    def __init__(self, device=None):
        # Set device
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model will be initialized after seeing data dimensions
        self.model = None
        
        # Preprocessing components
        self.text_scaler = StandardScaler()
        self.audio_scaler = StandardScaler()
        
        # Training parameters
        self.learning_rate = 1e-4
        self.batch_size = 32
        self.num_epochs = 100
        self.early_stopping_patience = 10
        
        # Training state
        self.is_trained = False
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    def _initialize_model(self, text_dim, audio_dim):
        """Initialize the multimodal transformer model"""
        self.model = MultimodalTransformer(
            text_feature_dim=text_dim,
            audio_feature_dim=audio_dim,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        ).to(self.device)
        
        print(f"Model initialized with text_dim={text_dim}, audio_dim={audio_dim}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_fusion_model(self, X_text, X_audio, y):
        """Train the multimodal transformer model"""
        print("Training Multimodal Transformer Model...")
        
        # Preprocessing
        X_text_scaled = self.text_scaler.fit_transform(X_text)
        X_audio_scaled = self.audio_scaler.fit_transform(X_audio)
        
        # Initialize model if not already done
        if self.model is None:
            self._initialize_model(X_text_scaled.shape[1], X_audio_scaled.shape[1])
        
        # Convert to tensors
        X_text_tensor = torch.FloatTensor(X_text_scaled).to(self.device)
        X_audio_tensor = torch.FloatTensor(X_audio_scaled).to(self.device)
        y_tensor = torch.LongTensor(y.values).to(self.device)
        
        # Create data loaders
        dataset = TensorDataset(X_text_tensor, X_audio_tensor, y_tensor)
        
        # Split into train and validation
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        
        # Training loop
        for epoch in range(self.num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_text, batch_audio, batch_labels in train_loader:
                optimizer.zero_grad()
                
                logits, attention_info = self.model(batch_text, batch_audio)
                loss = criterion(logits, batch_labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_text, batch_audio, batch_labels in val_loader:
                    logits, _ = self.model(batch_text, batch_audio)
                    loss = criterion(logits, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
            
            # Calculate metrics
            train_loss_avg = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            val_loss_avg = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_loss_avg)
            
            # Store history
            self.training_history['loss'].append(train_loss_avg)
            self.training_history['accuracy'].append(train_accuracy)
            self.training_history['val_loss'].append(val_loss_avg)
            self.training_history['val_accuracy'].append(val_accuracy)
            
            # Print progress
            if epoch % 10 == 0 or epoch == self.num_epochs - 1:
                print(f'Epoch [{epoch+1}/{self.num_epochs}]')
                print(f'  Train Loss: {train_loss_avg:.4f}, Train Acc: {train_accuracy:.2f}%')
                print(f'  Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.2f}%')
                print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Early stopping
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_multimodal_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_multimodal_model.pth'))
        self.is_trained = True
        
        # Save scalers for later use in prediction
        import pickle
        with open('text_scaler.pkl', 'wb') as f:
            pickle.dump(self.text_scaler, f)
        with open('audio_scaler.pkl', 'wb') as f:
            pickle.dump(self.audio_scaler, f)
        
        print("Multimodal Transformer training completed!")
    
    def predict(self, X_text, X_audio=None):
        """Make predictions using the trained multimodal transformer"""
        if not self.is_trained or self.model is None:
            raise ValueError("Model is not trained yet. Please complete training first.")
        
        self.model.eval()
        
        try:
            # Preprocess features
            X_text_scaled = self.text_scaler.transform(X_text)
            X_audio_scaled = self.audio_scaler.transform(X_audio) if X_audio is not None else np.zeros((X_text_scaled.shape[0], self.audio_scaler.n_features_in_))
            
            # Convert to tensors
            X_text_tensor = torch.FloatTensor(X_text_scaled).to(self.device)
            X_audio_tensor = torch.FloatTensor(X_audio_scaled).to(self.device)
            
            with torch.no_grad():
                logits, attention_info = self.model(X_text_tensor, X_audio_tensor)
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
            
            # Convert back to numpy
            predictions_np = predictions.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()
            
            # For compatibility with existing code, return in expected format
            return predictions_np, predictions_np, predictions_np, probabilities_np, probabilities_np, probabilities_np
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            raise
    
    def get_attention_weights(self, X_text, X_audio):
        """Get attention weights for interpretability"""
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")
        
        self.model.eval()
        X_text_scaled = self.text_scaler.transform(X_text)
        X_audio_scaled = self.audio_scaler.transform(X_audio)
        
        X_text_tensor = torch.FloatTensor(X_text_scaled).to(self.device)
        X_audio_tensor = torch.FloatTensor(X_audio_scaled).to(self.device)
        
        with torch.no_grad():
            _, attention_info = self.model(X_text_tensor, X_audio_tensor)
        
        return attention_info