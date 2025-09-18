"""
Pattern Detection Engine for ICT Concepts
Real-time pattern recognition with AI/ML models
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import pickle
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class Pattern:
    """Detected ICT pattern"""
    concept_name: str
    concept_number: int
    timestamp: datetime
    confidence: float
    pattern_type: str  # 'bullish', 'bearish', 'neutral'
    strength: str  # 'A+', 'A', 'B', 'C'
    timeframe: str
    price_level: float
    supporting_evidence: List[str]
    pattern_data: Dict[str, Any]

@dataclass 
class PatternDetectionResult:
    """Result from pattern detection"""
    symbol: str
    timestamp: datetime
    patterns: List[Pattern]
    total_patterns: int
    high_confidence_patterns: int
    execution_time_ms: float

class LSTMPatternNet(nn.Module):
    """LSTM network for time series pattern recognition"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, num_classes: int = 65):
        super(LSTMPatternNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_size, 8, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size//2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last time step for classification
        out = self.classifier(attn_out[:, -1, :])
        return out

class TransformerPatternNet(nn.Module):
    """Transformer network for pattern recognition"""
    
    def __init__(self, input_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 4, num_classes: int = 65):
        super(TransformerPatternNet, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model//2, num_classes),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        x += self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        x = self.transformer(x)
        out = self.classifier(x[:, -1, :])  # Use last token
        return out

class ICTPatternRecognitionAI:
    """Main AI engine for ICT pattern recognition"""
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # ICT Concept mapping
        self.ict_concepts = {
            1: "Market Structure (HH, HL, LH, LL)",
            2: "Liquidity (Buy-side & Sell-side)",
            3: "Liquidity Pools",
            4: "Order Blocks (Bullish & Bearish)",
            5: "Breaker Blocks",
            6: "Fair Value Gaps (FVG)",
            7: "Rejection Blocks",
            8: "Mitigation Blocks",
            9: "Supply & Demand Zones",
            10: "Premium & Discount (OTE)",
            11: "Dealing Ranges",
            12: "Swing Highs & Swing Lows",
            13: "Market Maker Buy & Sell Models",
            14: "Market Maker Programs",
            15: "Judas Swing",
            16: "Turtle Soup",
            17: "Power of 3",
            18: "Optimal Trade Entry",
            19: "SMT Divergence",
            20: "Liquidity Voids",
            21: "Killzones",
            22: "Session Opens",
            23: "Fibonacci Ratios",
            24: "Daily & Weekly Range Expectations",
            25: "Session Liquidity Raids",
            26: "Weekly Profiles",
            27: "Daily Bias",
            28: "Weekly Bias",
            29: "Monthly Bias",
            30: "Time of Day Highs & Lows",
            31: "Trade Journaling & Backtesting",
            32: "Entry Models (FVG, OB, Breaker)",
            33: "Exit Models",
            34: "Risk-to-Reward Optimization",
            35: "Position Sizing",
            36: "Drawdown Control",
            37: "Compounding Models",
            38: "Daily Loss Limits",
            39: "Probability Profiles",
            40: "High Probability Scenarios",
            41: "Liquidity Runs",
            42: "Reversals vs Continuations",
            43: "Accumulation & Distribution",
            44: "Order Flow",
            45: "High/Low Day Identification",
            46: "Range Expansion",
            47: "Inside/Outside Days",
            48: "Weekly Profile Analysis",
            49: "IPDA Theory",
            50: "Algo Price Delivery",
            51: "Silver Bullet Strategy",
            52: "Pre-Market Breakout",
            53: "Market Open Reversal",
            54: "Power Hour Strategy",
            55: "FVG Sniper Entry",
            56: "Order Block Strategy",
            57: "Breaker Block Strategy",
            58: "Rejection Block Strategy",
            59: "SMT Divergence Strategy",
            60: "Turtle Soup Strategy",
            61: "Power of 3 Strategy",
            62: "Daily Bias + Liquidity Strategy",
            63: "Morning Session Strategy",
            64: "Afternoon Reversal Strategy",
            65: "Optimal Trade Entry Strategy"
        }
        
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Performance tracking
        self.detection_metrics = {
            'total_detections': 0,
            'high_confidence_detections': 0,
            'avg_confidence': 0.0,
            'avg_execution_time': 0.0
        }
        
    def load_models(self):
        """Load pre-trained models"""
        try:
            # Load Random Forest models
            rf_path = self.model_dir / "random_forest_ensemble.pkl"
            if rf_path.exists():
                with open(rf_path, 'rb') as f:
                    self.models['random_forest'] = pickle.load(f)
                    
            # Load LSTM model
            lstm_path = self.model_dir / "lstm_pattern_net.pth"
            if lstm_path.exists():
                self.models['lstm'] = LSTMPatternNet(input_size=200)  # Assuming 200 features
                self.models['lstm'].load_state_dict(torch.load(lstm_path))
                self.models['lstm'].eval()
                
            # Load Transformer model
            transformer_path = self.model_dir / "transformer_pattern_net.pth"
            if transformer_path.exists():
                self.models['transformer'] = TransformerPatternNet(input_size=200)
                self.models['transformer'].load_state_dict(torch.load(transformer_path))
                self.models['transformer'].eval()
                
            # Load scalers
            scaler_path = self.model_dir / "feature_scalers.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scalers = pickle.load(f)
                    
            self.is_trained = len(self.models) > 0
            logger.info(f"Loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_trained = False
            
    def save_models(self):
        """Save trained models"""
        try:
            # Save Random Forest
            if 'random_forest' in self.models:
                with open(self.model_dir / "random_forest_ensemble.pkl", 'wb') as f:
                    pickle.dump(self.models['random_forest'], f)
                    
            # Save LSTM
            if 'lstm' in self.models:
                torch.save(self.models['lstm'].state_dict(), self.model_dir / "lstm_pattern_net.pth")
                
            # Save Transformer
            if 'transformer' in self.models:
                torch.save(self.models['transformer'].state_dict(), self.model_dir / "transformer_pattern_net.pth")
                
            # Save scalers
            if self.scalers:
                with open(self.model_dir / "feature_scalers.pkl", 'wb') as f:
                    pickle.dump(self.scalers, f)
                    
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            
    def prepare_features_for_inference(self, features: pd.DataFrame, sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for model inference"""
        
        # Scale features
        if 'main' in self.scalers:
            scaled_features = self.scalers['main'].transform(features.fillna(0))
        else:
            scaled_features = features.fillna(0).values
            
        # Create sequences for LSTM/Transformer
        sequences = []
        if len(scaled_features) >= sequence_length:
            for i in range(sequence_length, len(scaled_features)):
                sequences.append(scaled_features[i-sequence_length:i])
                
            sequences = np.array(sequences)
        else:
            # If not enough data, pad with zeros
            padded = np.zeros((sequence_length, scaled_features.shape[1]))
            padded[:len(scaled_features)] = scaled_features
            sequences = np.array([padded])
            
        return scaled_features, sequences
        
    def detect_patterns_with_ensemble(self, features: pd.DataFrame, symbol: str, timeframe: str) -> List[Pattern]:
        """Detect patterns using ensemble of models"""
        
        if not self.is_trained:
            logger.warning("Models not trained, using rule-based detection")
            return self._rule_based_pattern_detection(features, symbol, timeframe)
            
        patterns = []
        
        try:
            # Prepare features
            scaled_features, sequences = self.prepare_features_for_inference(features)
            
            # Get predictions from each model
            predictions = {}
            
            # Random Forest predictions
            if 'random_forest' in self.models and len(scaled_features) > 0:
                rf_pred = self.models['random_forest'].predict_proba(scaled_features[-1:])
                predictions['random_forest'] = rf_pred[0]
                
            # LSTM predictions
            if 'lstm' in self.models and len(sequences) > 0:
                with torch.no_grad():
                    lstm_input = torch.FloatTensor(sequences[-1:])
                    lstm_pred = self.models['lstm'](lstm_input)
                    predictions['lstm'] = lstm_pred.numpy()[0]
                    
            # Transformer predictions
            if 'transformer' in self.models and len(sequences) > 0:
                with torch.no_grad():
                    transformer_input = torch.FloatTensor(sequences[-1:])
                    transformer_pred = self.models['transformer'](transformer_input)
                    predictions['transformer'] = transformer_pred.numpy()[0]
                    
            # Ensemble predictions (average)
            if predictions:
                ensemble_pred = np.mean([pred for pred in predictions.values()], axis=0)
                
                # Convert predictions to patterns
                current_time = datetime.now()
                current_price = features.index[-1] if len(features) > 0 else 0
                
                for concept_idx, confidence in enumerate(ensemble_pred):
                    if confidence > 0.5:  # Confidence threshold
                        concept_num = concept_idx + 1
                        
                        # Determine pattern type and strength
                        pattern_type = 'bullish' if confidence > 0.7 else 'bearish' if confidence < 0.3 else 'neutral'
                        strength = self._classify_pattern_strength(confidence)
                        
                        pattern = Pattern(
                            concept_name=self.ict_concepts.get(concept_num, f"Concept {concept_num}"),
                            concept_number=concept_num,
                            timestamp=current_time,
                            confidence=float(confidence),
                            pattern_type=pattern_type,
                            strength=strength,
                            timeframe=timeframe,
                            price_level=current_price,
                            supporting_evidence=[f"Ensemble confidence: {confidence:.3f}"],
                            pattern_data={
                                'model_predictions': {model: float(pred[concept_idx]) for model, pred in predictions.items()},
                                'ensemble_confidence': float(confidence)
                            }
                        )
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.error(f"Error in ensemble pattern detection: {e}")
            # Fallback to rule-based detection
            patterns = self._rule_based_pattern_detection(features, symbol, timeframe)
            
        return patterns
        
    def _rule_based_pattern_detection(self, features: pd.DataFrame, symbol: str, timeframe: str) -> List[Pattern]:
        """Rule-based pattern detection as fallback"""
        patterns = []
        
        if features.empty:
            return patterns
            
        current_time = datetime.now()
        
        try:
            # Look for specific ICT patterns in features
            latest_features = features.iloc[-1]
            
            # Fair Value Gap detection
            if 'FVG_Bull_Gap' in features.columns and latest_features.get('FVG_Bull_Gap', 0) == 1:
                patterns.append(Pattern(
                    concept_name="Fair Value Gaps (FVG)",
                    concept_number=6,
                    timestamp=current_time,
                    confidence=0.75,
                    pattern_type='bullish',
                    strength='B',
                    timeframe=timeframe,
                    price_level=0,
                    supporting_evidence=["Bullish FVG detected"],
                    pattern_data={'type': 'bullish_fvg'}
                ))
                
            if 'FVG_Bear_Gap' in features.columns and latest_features.get('FVG_Bear_Gap', 0) == 1:
                patterns.append(Pattern(
                    concept_name="Fair Value Gaps (FVG)",
                    concept_number=6,
                    timestamp=current_time,
                    confidence=0.75,
                    pattern_type='bearish',
                    strength='B',
                    timeframe=timeframe,
                    price_level=0,
                    supporting_evidence=["Bearish FVG detected"],
                    pattern_data={'type': 'bearish_fvg'}
                ))
                
            # Order Block detection
            if 'Bull_OB_Signal' in features.columns and latest_features.get('Bull_OB_Signal', 0) == 1:
                patterns.append(Pattern(
                    concept_name="Order Blocks (Bullish & Bearish)",
                    concept_number=4,
                    timestamp=current_time,
                    confidence=0.70,
                    pattern_type='bullish',
                    strength='B',
                    timeframe=timeframe,
                    price_level=0,
                    supporting_evidence=["Bullish Order Block detected"],
                    pattern_data={'type': 'bullish_ob'}
                ))
                
            # Add more rule-based detections...
            
        except Exception as e:
            logger.error(f"Error in rule-based detection: {e}")
            
        return patterns
        
    def _classify_pattern_strength(self, confidence: float) -> str:
        """Classify pattern strength based on confidence"""
        if confidence >= 0.9:
            return 'A+'
        elif confidence >= 0.8:
            return 'A'
        elif confidence >= 0.7:
            return 'B'
        else:
            return 'C'
            
    def real_time_pattern_detection(self, features: pd.DataFrame, symbol: str, timeframe: str) -> PatternDetectionResult:
        """Main entry point for real-time pattern detection"""
        
        start_time = datetime.now()
        
        # Detect patterns
        patterns = self.detect_patterns_with_ensemble(features, symbol, timeframe)
        
        # Filter high confidence patterns
        high_confidence_patterns = [p for p in patterns if p.confidence >= 0.8]
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update metrics
        self.detection_metrics['total_detections'] += len(patterns)
        self.detection_metrics['high_confidence_detections'] += len(high_confidence_patterns)
        if patterns:
            avg_conf = np.mean([p.confidence for p in patterns])
            self.detection_metrics['avg_confidence'] = (
                self.detection_metrics['avg_confidence'] * 0.9 + avg_conf * 0.1
            )
        self.detection_metrics['avg_execution_time'] = (
            self.detection_metrics['avg_execution_time'] * 0.9 + execution_time * 0.1
        )
        
        result = PatternDetectionResult(
            symbol=symbol,
            timestamp=start_time,
            patterns=patterns,
            total_patterns=len(patterns),
            high_confidence_patterns=len(high_confidence_patterns),
            execution_time_ms=execution_time
        )
        
        logger.info(f"Detected {len(patterns)} patterns for {symbol} in {execution_time:.2f}ms")
        
        return result
        
    def get_detection_metrics(self) -> Dict[str, float]:
        """Get pattern detection performance metrics"""
        return self.detection_metrics.copy()

# Export main classes
__all__ = ['ICTPatternRecognitionAI', 'Pattern', 'PatternDetectionResult', 'LSTMPatternNet', 'TransformerPatternNet']