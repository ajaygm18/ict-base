"""
Model Training Engine for ICT Pattern Recognition
Trains ensemble models on historical stock data
"""

import pandas as pd
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .pattern_detector import LSTMPatternNet, TransformerPatternNet
from .feature_engineer import TechnicalIndicatorEngine, MultiTimeframeFeatureEngine

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    symbols: List[str]
    start_date: str  # 5+ years of data
    end_date: str
    timeframes: List[str]
    sequence_length: int = 50
    test_size: float = 0.2
    cv_folds: int = 5
    
    # Model hyperparameters
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    transformer_d_model: int = 256
    transformer_nhead: int = 8
    transformer_num_layers: int = 4
    
    # Training hyperparameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10

@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float
    validation_loss: float

class ICTPatternLabeler:
    """Creates labeled datasets for ICT pattern training"""
    
    def __init__(self):
        self.feature_engine = TechnicalIndicatorEngine()
        
    def create_labels_for_ict_concepts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary labels for all 65 ICT concepts"""
        
        labels = pd.DataFrame(index=df.index)
        
        # Concept 1: Market Structure (HH, HL, LH, LL)
        hh = (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2))
        hl = (df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) > df['Low'].shift(2))
        lh = (df['High'] < df['High'].shift(1)) & (df['High'].shift(1) < df['High'].shift(2))
        ll = (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2))
        labels['concept_1'] = (hh | hl | lh | ll).astype(int)
        
        # Concept 2: Liquidity (buy-side & sell-side)
        high_break = df['High'] > df['High'].rolling(20).max().shift(1)
        low_break = df['Low'] < df['Low'].rolling(20).min().shift(1)
        labels['concept_2'] = (high_break | low_break).astype(int)
        
        # Concept 3: Liquidity Pools (equal highs/lows)
        equal_highs = abs(df['High'] - df['High'].rolling(20).max().shift(1)) / df['High'] < 0.002
        equal_lows = abs(df['Low'] - df['Low'].rolling(20).min().shift(1)) / df['Low'] < 0.002
        labels['concept_3'] = (equal_highs | equal_lows).astype(int)
        
        # Concept 4: Order Blocks
        down_candle = df['Close'] < df['Open']
        up_candle = df['Close'] > df['Open']
        strong_up_move = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) > 0.03
        strong_down_move = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) < -0.03
        bull_ob = down_candle & strong_up_move.shift(-1)
        bear_ob = up_candle & strong_down_move.shift(-1)
        labels['concept_4'] = (bull_ob | bear_ob).astype(int)
        
        # Concept 5: Breaker Blocks (former resistance becomes support)
        resistance_break = (df['High'] > df['High'].rolling(20).max().shift(1)) & (df['Close'].shift(5) < df['High'].rolling(20).max().shift(1))
        support_break = (df['Low'] < df['Low'].rolling(20).min().shift(1)) & (df['Close'].shift(5) > df['Low'].rolling(20).min().shift(1))
        labels['concept_5'] = (resistance_break | support_break).astype(int)
        
        # Concept 6: Fair Value Gaps (FVG)
        bull_fvg = (df['Low'].shift(1) > df['High'].shift(-1)) & (df['Close'] > df['Close'].shift(2))
        bear_fvg = (df['High'].shift(1) < df['Low'].shift(-1)) & (df['Close'] < df['Close'].shift(2))
        labels['concept_6'] = (bull_fvg | bear_fvg).astype(int)
        
        # Concept 7: Rejection Blocks (strong rejections with wicks)
        upper_wick = df['High'] - np.maximum(df['Open'], df['Close'])
        lower_wick = np.minimum(df['Open'], df['Close']) - df['Low']
        body_size = abs(df['Close'] - df['Open'])
        strong_rejection = (upper_wick > 2 * body_size) | (lower_wick > 2 * body_size)
        labels['concept_7'] = strong_rejection.astype(int)
        
        # For remaining concepts, create simplified labels based on technical patterns
        # This is a simplified approach - in practice, each concept would need detailed labeling logic
        
        for concept_num in range(8, 66):  # Concepts 8-65
            # Create pseudo-labels based on price action and technical indicators
            returns = df['Close'].pct_change()
            volatility = returns.rolling(20).std()
            volume_spike = df['Volume'] > df['Volume'].rolling(20).mean() * 2
            
            # Combine different signals for different concepts
            if concept_num <= 20:  # Core concepts
                signal = (abs(returns) > volatility * 2) & volume_spike
            elif concept_num <= 30:  # Time & Price concepts
                signal = (df['High'] == df['High'].rolling(20).max()) | (df['Low'] == df['Low'].rolling(20).min())
            elif concept_num <= 39:  # Risk management concepts
                signal = abs(returns) > volatility * 3
            elif concept_num <= 50:  # Advanced concepts
                signal = volume_spike & (abs(returns) > volatility * 1.5)
            else:  # Strategy concepts
                signal = (abs(returns) > volatility * 2) & (df['Volume'] > df['Volume'].rolling(10).mean() * 1.5)
                
            labels[f'concept_{concept_num}'] = signal.astype(int)
            
        return labels.fillna(0)
        
    def create_training_dataset(self, symbol: str, start_date: str, end_date: str, timeframe: str = '5m') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create complete training dataset with features and labels"""
        
        # Download historical data
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=timeframe)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
            
        # Create features
        feature_set = self.feature_engine.create_comprehensive_features(df, symbol, timeframe)
        features = feature_set.features
        
        # Create labels
        labels = self.create_labels_for_ict_concepts(df)
        
        # Align features and labels
        common_index = features.index.intersection(labels.index)
        features = features.loc[common_index]
        labels = labels.loc[common_index]
        
        logger.info(f"Created dataset for {symbol}: {len(features)} samples, {len(features.columns)} features, {len(labels.columns)} labels")
        
        return features, labels

class ModelTrainer:
    """Trains ensemble models for ICT pattern recognition"""
    
    def __init__(self, config: TrainingConfig, model_dir: str = "data/models"):
        self.config = config
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.labeler = ICTPatternLabeler()
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
        # Setup device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
    def collect_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect and combine training data from multiple symbols"""
        
        all_features = []
        all_labels = []
        
        logger.info(f"Collecting training data for {len(self.config.symbols)} symbols")
        
        for symbol in self.config.symbols:
            try:
                logger.info(f"Processing {symbol}")
                features, labels = self.labeler.create_training_dataset(
                    symbol, self.config.start_date, self.config.end_date
                )
                
                if not features.empty and not labels.empty:
                    all_features.append(features)
                    all_labels.append(labels)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
                
        if not all_features:
            raise ValueError("No valid training data collected")
            
        # Combine all data
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)
        
        logger.info(f"Combined dataset: {len(combined_features)} samples, {len(combined_features.columns)} features")
        
        return combined_features, combined_labels
        
    def prepare_data_for_training(self, features: pd.DataFrame, labels: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training"""
        
        # Remove any remaining NaN values
        features = features.fillna(0)
        labels = labels.fillna(0)
        
        # Split data (time series split)
        split_idx = int(len(features) * (1 - self.config.test_size))
        
        X_train = features.iloc[:split_idx].values
        y_train = labels.iloc[:split_idx].values
        X_test = features.iloc[split_idx:].values
        y_test = labels.iloc[split_idx:].values
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        logger.info(f"Training data: {X_train_scaled.shape}, Test data: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
        
    def create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM/Transformer training"""
        
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
            
        return np.array(X_seq), np.array(y_seq)
        
    def train_random_forest(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Train Random Forest classifier"""
        
        logger.info("Training Random Forest model")
        start_time = datetime.now()
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        # Grid search for hyperparameters
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='f1_macro', n_jobs=-1)
        
        # Train on all concepts (multi-label)
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        
        # Evaluate
        y_pred = best_rf.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        accuracy = np.mean(y_pred == y_test)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.models['random_forest'] = best_rf
        
        performance = ModelPerformance(
            model_name='RandomForest',
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_time=training_time,
            validation_loss=0.0  # N/A for RF
        )
        
        logger.info(f"Random Forest - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Time: {training_time:.1f}s")
        
        return performance
        
    def train_lstm(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Train LSTM model"""
        
        logger.info("Training LSTM model")
        start_time = datetime.now()
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, self.config.sequence_length)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, self.config.sequence_length)
        
        if len(X_train_seq) == 0:
            logger.warning("Not enough data for LSTM sequences")
            return ModelPerformance('LSTM', 0, 0, 0, 0, 0, float('inf'))
            
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_seq).to(self.device)
        
        # Create model
        input_size = X_train_seq.shape[2]
        num_classes = y_train_seq.shape[1]
        
        model = LSTMPatternNet(
            input_size=input_size,
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            num_classes=num_classes
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            model.train()
            
            # Forward pass
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.model_dir / 'best_lstm.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
                
        # Load best model and evaluate
        model.load_state_dict(torch.load(self.model_dir / 'best_lstm.pth'))
        model.eval()
        
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_predictions = (test_outputs > 0.5).float()
            
            accuracy = (test_predictions == y_test_tensor).float().mean().item()
            
            # Convert to numpy for sklearn metrics
            y_test_np = y_test_tensor.cpu().numpy()
            y_pred_np = test_predictions.cpu().numpy()
            
            precision, recall, f1, _ = precision_recall_fscore_support(y_test_np, y_pred_np, average='macro', zero_division=0)
            
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.models['lstm'] = model
        
        performance = ModelPerformance(
            model_name='LSTM',
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_time=training_time,
            validation_loss=best_val_loss.item()
        )
        
        logger.info(f"LSTM - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Time: {training_time:.1f}s")
        
        return performance
        
    def train_transformer(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """Train Transformer model"""
        
        logger.info("Training Transformer model")
        start_time = datetime.now()
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train, self.config.sequence_length)
        X_test_seq, y_test_seq = self.create_sequences(X_test, y_test, self.config.sequence_length)
        
        if len(X_train_seq) == 0:
            logger.warning("Not enough data for Transformer sequences")
            return ModelPerformance('Transformer', 0, 0, 0, 0, 0, float('inf'))
            
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_seq).to(self.device)
        
        # Create model
        input_size = X_train_seq.shape[2]
        num_classes = y_train_seq.shape[1]
        
        model = TransformerPatternNet(
            input_size=input_size,
            d_model=self.config.transformer_d_model,
            nhead=self.config.transformer_nhead,
            num_layers=self.config.transformer_num_layers,
            num_classes=num_classes
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop (similar to LSTM)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            model.train()
            
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.model_dir / 'best_transformer.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
                
        # Evaluate
        model.load_state_dict(torch.load(self.model_dir / 'best_transformer.pth'))
        model.eval()
        
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_predictions = (test_outputs > 0.5).float()
            
            accuracy = (test_predictions == y_test_tensor).float().mean().item()
            
            y_test_np = y_test_tensor.cpu().numpy()
            y_pred_np = test_predictions.cpu().numpy()
            
            precision, recall, f1, _ = precision_recall_fscore_support(y_test_np, y_pred_np, average='macro', zero_division=0)
            
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.models['transformer'] = model
        
        performance = ModelPerformance(
            model_name='Transformer',
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            training_time=training_time,
            validation_loss=best_val_loss.item()
        )
        
        logger.info(f"Transformer - Accuracy: {accuracy:.3f}, F1: {f1:.3f}, Time: {training_time:.1f}s")
        
        return performance
        
    def train_all_models(self) -> Dict[str, ModelPerformance]:
        """Train all models in the ensemble"""
        
        logger.info("Starting ensemble model training")
        
        # Collect training data
        features, labels = self.collect_training_data()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data_for_training(features, labels)
        
        # Train models
        performances = {}
        
        # Random Forest
        try:
            rf_performance = self.train_random_forest(X_train, X_test, y_train, y_test)
            performances['random_forest'] = rf_performance
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            
        # LSTM
        try:
            lstm_performance = self.train_lstm(X_train, X_test, y_train, y_test)
            performances['lstm'] = lstm_performance
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            
        # Transformer
        try:
            transformer_performance = self.train_transformer(X_train, X_test, y_train, y_test)
            performances['transformer'] = transformer_performance
        except Exception as e:
            logger.error(f"Error training Transformer: {e}")
            
        self.performance_metrics = performances
        
        # Save models and scalers
        self.save_models()
        
        logger.info(f"Training completed. Trained {len(performances)} models.")
        
        return performances
        
    def save_models(self):
        """Save all trained models"""
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
                    
            # Save performance metrics
            if self.performance_metrics:
                with open(self.model_dir / "performance_metrics.pkl", 'wb') as f:
                    pickle.dump(self.performance_metrics, f)
                    
            logger.info("All models and scalers saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")

# Export main classes
__all__ = ['ModelTrainer', 'TrainingConfig', 'ModelPerformance', 'ICTPatternLabeler']