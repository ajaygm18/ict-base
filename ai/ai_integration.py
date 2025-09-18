"""
AI Integration Module
Integrates AI/ML components with the main ICT application
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path

from .feature_engineer import TechnicalIndicatorEngine, MultiTimeframeFeatureEngine, FeatureSet
from .pattern_detector import ICTPatternRecognitionAI, Pattern, PatternDetectionResult
from .model_trainer import ModelTrainer, TrainingConfig, ModelPerformance

logger = logging.getLogger(__name__)

class PatternLabelValidator:
    """Validates the quality of pattern labels for training"""
    
    def __init__(self):
        self.min_samples_per_class = 50
        self.min_class_balance = 0.1  # Min 10% representation
    
    def validate_labels(self, labels: np.ndarray) -> Dict[str, Any]:
        """Validate label quality"""
        try:
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_samples = len(labels)
            
            # Class balance check
            min_count = np.min(counts)
            max_count = np.max(counts)
            balance_ratio = min_count / max_count if max_count > 0 else 0
            
            # Minimum samples per class
            has_min_samples = min_count >= self.min_samples_per_class
            
            # Label distribution
            class_distribution = {str(label): count/total_samples for label, count in zip(unique_labels, counts)}
            
            # Quality score
            quality_score = balance_ratio * 0.5 + (1.0 if has_min_samples else 0.0) * 0.5
            
            return {
                'quality_score': quality_score,
                'class_balance': balance_ratio,
                'min_samples_per_class': has_min_samples,
                'class_distribution': class_distribution,
                'total_samples': total_samples,
                'num_classes': len(unique_labels)
            }
            
        except Exception as e:
            logger.error(f"Error validating labels: {e}")
            return {'quality_score': 0.0, 'error': str(e)}

class AIIntegrationEngine:
    """Professional AI/ML integration with proper validation and backtesting"""
    
    def __init__(self, validation_split: float = 0.2, min_training_samples: int = 5000):
        # Create default training configuration
        default_config = TrainingConfig(
            symbols=['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'],
            start_date='2019-01-01',
            end_date='2024-12-31',
            timeframes=['1h', '1d'],
            sequence_length=50,
            test_size=0.2,
            cv_folds=5
        )
        
        self.model_trainer = ModelTrainer(default_config)
        self.pattern_detector = ICTPatternRecognitionAI()
        self.feature_engine = MultiTimeframeFeatureEngine()
        self.models = {}
        self.model_performance = {}
        self.validation_split = validation_split
        self.min_training_samples = min_training_samples
        self.label_validator = PatternLabelValidator()
        
        # Performance tracking
        self.performance_stats = {
            'avg_feature_creation_time': 0.0,
            'avg_pattern_detection_time': 0.0,
            'cache_hit_rate': 0.0,
            'total_analyses': 0
        }
        
        # Caching
        self.feature_cache = {}
        self.pattern_cache = {}
        
        # Model directory
        self.model_dir = Path(__file__).parent / 'models'
        self.model_dir.mkdir(exist_ok=True)
        
    async def initialize(self):
        """Initialize AI engine - basic setup"""
        try:
            logger.info("AI Integration Engine initialized successfully")
            # Basic initialization - models will be trained on demand
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AI engine: {e}")
            return False
        
    async def initialize_models(self, data_processor):
        """Initialize and train all AI models with proper validation"""
        try:
            logger.info("Starting AI model initialization with validation...")
            
            # Get and validate training data
            training_data = await self._prepare_training_data(data_processor)
            
            if training_data is None or len(training_data) < self.min_training_samples:
                logger.error(f"Insufficient training data: {len(training_data) if training_data is not None else 0} samples, need {self.min_training_samples}")
                return False
            
            # Validate labels for quality
            label_quality = await self._validate_training_labels(training_data)
            if label_quality['quality_score'] < 0.6:
                logger.error(f"Training label quality too low: {label_quality['quality_score']:.2f}")
                return False
            
            # Split data properly
            train_data, val_data = self._split_data_chronologically(training_data)
            
            # Train models with validation
            success = await self._train_all_models_with_validation(train_data, val_data)
            
            if success:
                # Run final validation
                final_metrics = await self._validate_all_models(val_data)
                logger.info(f"AI models initialized successfully. Validation metrics: {final_metrics}")
                return True
            else:
                logger.error("AI model training failed validation")
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            return False
    
    async def _prepare_training_data(self, data_processor) -> Optional[pd.DataFrame]:
        """Prepare high-quality training data with proper ICT labels"""
        try:
            logger.info("Preparing training data with validated ICT patterns...")
            
            # Get historical data (minimum 2 years for robust training)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']  # Liquid symbols for training
            all_training_data = []
            
            for symbol in symbols:
                logger.info(f"Processing training data for {symbol}")
                
                # Get historical data
                stock_data = await data_processor.get_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                
                if stock_data is None or len(stock_data) < 1000:
                    logger.warning(f"Insufficient data for {symbol}: {len(stock_data) if stock_data is not None else 0} samples")
                    continue
                
                # Create features
                feature_set = self.feature_engine.create_comprehensive_features(stock_data, symbol, '1h')
                
                if feature_set.features.empty:
                    logger.warning(f"No features created for {symbol}")
                    continue
                
                # Generate validated ICT labels using actual pattern detection
                labels = await self._generate_validated_ict_labels(stock_data, feature_set, symbol)
                
                if labels is None or len(labels) < 100:
                    logger.warning(f"Insufficient valid labels for {symbol}: {len(labels) if labels is not None else 0}")
                    continue
                
                # Combine features and labels
                training_subset = feature_set.features.copy()
                training_subset['labels'] = labels
                training_subset['symbol'] = symbol
                
                all_training_data.append(training_subset)
                logger.info(f"Added {len(training_subset)} training samples for {symbol}")
            
            if not all_training_data:
                logger.error("No valid training data could be prepared")
                return None
            
            # Combine all data
            combined_data = pd.concat(all_training_data, ignore_index=True)
            
            # Remove rows with missing labels
            combined_data = combined_data.dropna(subset=['labels'])
            
            logger.info(f"Prepared {len(combined_data)} total training samples")
            return combined_data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None
    
    async def _generate_validated_ict_labels(self, stock_data: pd.DataFrame, feature_set: FeatureSet, symbol: str) -> Optional[np.ndarray]:
        """Generate high-quality ICT pattern labels using multiple validation methods"""
        try:
            from ..ict_engine.core_concepts import StockMarketStructureAnalyzer
            
            # Initialize ICT analyzer
            ict_analyzer = StockMarketStructureAnalyzer(symbol)
            
            labels = []
            
            for i in range(50, len(stock_data) - 50):  # Leave buffer for pattern validation
                current_data = stock_data.iloc[:i+1].copy()
                future_data = stock_data.iloc[i:i+50].copy()  # Next 50 periods for validation
                
                try:
                    # Analyze current market structure
                    structure_analysis = ict_analyzer.analyze_market_structure(current_data)
                    
                    # Look for specific high-probability patterns
                    pattern_signals = []
                    
                    # Check for bullish order blocks
                    ob_bullish = ict_analyzer.concept_4_order_blocks_bullish_bearish(current_data)
                    if ob_bullish and len(ob_bullish.get('bullish_blocks', [])) > 0:
                        # Validate with future price action
                        if self._validate_bullish_pattern(current_data.iloc[-1], future_data):
                            pattern_signals.append(1)  # Bullish
                    
                    # Check for bearish order blocks
                    if ob_bullish and len(ob_bullish.get('bearish_blocks', [])) > 0:
                        if self._validate_bearish_pattern(current_data.iloc[-1], future_data):
                            pattern_signals.append(-1)  # Bearish
                    
                    # Check for FVG patterns
                    fvg_result = ict_analyzer.concept_6_fair_value_gaps_fvg_imbalances(current_data)
                    if fvg_result and len(fvg_result.get('fvgs', [])) > 0:
                        latest_fvg = fvg_result['fvgs'][-1]
                        if latest_fvg.get('direction') == 'bullish':
                            if self._validate_bullish_pattern(current_data.iloc[-1], future_data):
                                pattern_signals.append(1)
                        elif latest_fvg.get('direction') == 'bearish':
                            if self._validate_bearish_pattern(current_data.iloc[-1], future_data):
                                pattern_signals.append(-1)
                    
                    # Determine final label
                    if len(pattern_signals) > 0:
                        # Use majority vote
                        bullish_count = sum(1 for s in pattern_signals if s > 0)
                        bearish_count = sum(1 for s in pattern_signals if s < 0)
                        
                        if bullish_count > bearish_count:
                            labels.append(1)  # Bullish
                        elif bearish_count > bullish_count:
                            labels.append(-1)  # Bearish
                        else:
                            labels.append(0)  # Neutral
                    else:
                        labels.append(0)  # No clear pattern
                        
                except Exception as e:
                    labels.append(0)  # Default to neutral on error
            
            return np.array(labels) if labels else None
            
        except Exception as e:
            logger.error(f"Error generating ICT labels for {symbol}: {e}")
            return None
    
    def _validate_bullish_pattern(self, current_candle: pd.Series, future_data: pd.DataFrame) -> bool:
        """Validate bullish pattern with future price action"""
        try:
            entry_price = current_candle['close']
            
            # Check if price moves favorably in next 10-20 periods
            future_highs = future_data['high'].iloc[5:25]  # Skip immediate noise
            
            if len(future_highs) == 0:
                return False
            
            max_future_high = future_highs.max()
            min_favorable_target = entry_price * 1.01  # At least 1% move
            
            return max_future_high >= min_favorable_target
            
        except Exception:
            return False
    
    def _validate_bearish_pattern(self, current_candle: pd.Series, future_data: pd.DataFrame) -> bool:
        """Validate bearish pattern with future price action"""
        try:
            entry_price = current_candle['close']
            
            # Check if price moves favorably in next 10-20 periods
            future_lows = future_data['low'].iloc[5:25]  # Skip immediate noise
            
            if len(future_lows) == 0:
                return False
            
            min_future_low = future_lows.min()
            max_favorable_target = entry_price * 0.99  # At least 1% move down
            
            return min_future_low <= max_favorable_target
            
        except Exception:
            return False
    
    async def _validate_training_labels(self, training_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate the quality of training labels"""
        try:
            labels = training_data['labels'].values
            return self.label_validator.validate_labels(labels)
            
        except Exception as e:
            logger.error(f"Error validating training labels: {e}")
            return {'quality_score': 0.0, 'error': str(e)}
    
    def _split_data_chronologically(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data chronologically (not randomly) for time series"""
        split_idx = int(len(data) * (1 - self.validation_split))
        train_data = data.iloc[:split_idx].copy()
        val_data = data.iloc[split_idx:].copy()
        
        logger.info(f"Split data: {len(train_data)} training, {len(val_data)} validation samples")
        return train_data, val_data
    
    async def _train_all_models_with_validation(self, train_data: pd.DataFrame, val_data: pd.DataFrame) -> bool:
        """Train all models with proper validation"""
        try:
            # Prepare features and labels
            feature_cols = [col for col in train_data.columns if col not in ['labels', 'symbol']]
            
            X_train = train_data[feature_cols].fillna(0).values
            y_train = train_data['labels'].values
            
            X_val = val_data[feature_cols].fillna(0).values
            y_val = val_data['labels'].values
            
            # Train models
            config = TrainingConfig(
                symbols=train_data['symbol'].unique().tolist(),
                sequence_length=50,
                epochs=100,
                batch_size=64,
                learning_rate=0.001,
                validation_split=0.0  # We already split
            )
            
            trainer = ModelTrainer(config, str(self.model_dir))
            
            # Train with validation data
            performances = trainer.train_with_validation(X_train, y_train, X_val, y_val)
            
            # Check if models meet minimum performance
            min_accuracy = 0.55  # Better than random (50%)
            all_passed = True
            
            for model_name, performance in performances.items():
                if performance.validation_accuracy < min_accuracy:
                    logger.warning(f"Model {model_name} failed validation: {performance.validation_accuracy:.3f} < {min_accuracy:.3f}")
                    all_passed = False
                else:
                    logger.info(f"Model {model_name} passed validation: {performance.validation_accuracy:.3f}")
            
            self.model_performance = performances
            return all_passed
            
        except Exception as e:
            logger.error(f"Error training models with validation: {e}")
            return False
    
    async def _validate_all_models(self, val_data: pd.DataFrame) -> Dict[str, float]:
        """Final validation of all models"""
        try:
            metrics = {}
            
            feature_cols = [col for col in val_data.columns if col not in ['labels', 'symbol']]
            X_val = val_data[feature_cols].fillna(0).values
            y_val = val_data['labels'].values
            
            for model_name, performance in self.model_performance.items():
                metrics[f"{model_name}_accuracy"] = performance.validation_accuracy
                metrics[f"{model_name}_precision"] = performance.validation_precision
                metrics[f"{model_name}_recall"] = performance.validation_recall
                metrics[f"{model_name}_f1"] = performance.validation_f1
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in final validation: {e}")
            return {}
            
    def create_features_for_symbol(self, symbol: str, stock_data: pd.DataFrame, timeframe: str = '5m') -> FeatureSet:
        """Create comprehensive features for a symbol"""
        
        cache_key = f"{symbol}_{timeframe}_{hash(str(stock_data.index[-1]))}"
        
        # Check cache
        if cache_key in self.feature_cache:
            self.performance_stats['cache_hit_rate'] = (
                self.performance_stats['cache_hit_rate'] * 0.9 + 1.0 * 0.1
            )
            return self.feature_cache[cache_key]
            
        start_time = datetime.now()
        
        try:
            # Create comprehensive features
            feature_set = self.feature_engine.create_comprehensive_features(
                stock_data, symbol, timeframe
            )
            
            # Cache the result
            self.feature_cache[cache_key] = feature_set
            
            # Update performance stats
            creation_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_stats['avg_feature_creation_time'] = (
                self.performance_stats['avg_feature_creation_time'] * 0.9 + creation_time * 0.1
            )
            self.performance_stats['cache_hit_rate'] = (
                self.performance_stats['cache_hit_rate'] * 0.9 + 0.0 * 0.1
            )
            
            logger.info(f"Created {len(feature_set.feature_names)} features for {symbol} in {creation_time:.2f}ms")
            
            return feature_set
            
        except Exception as e:
            logger.error(f"Error creating features for {symbol}: {e}")
            # Return empty feature set
            return FeatureSet(
                timeframe=timeframe,
                features=pd.DataFrame(),
                feature_names=[],
                creation_timestamp=datetime.now(),
                symbol=symbol
            )
            
    def detect_patterns_for_symbol(self, symbol: str, features: FeatureSet, timeframe: str = '5m') -> PatternDetectionResult:
        """Detect ICT patterns for a symbol"""
        
        cache_key = f"{symbol}_{timeframe}_patterns_{hash(str(features.creation_timestamp))}"
        
        # Check cache
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
            
        start_time = datetime.now()
        
        try:
            # Detect patterns using AI
            result = self.pattern_detector.real_time_pattern_detection(
                features.features, symbol, timeframe
            )
            
            # Cache the result
            self.pattern_cache[cache_key] = result
            
            # Update performance stats
            detection_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_stats['avg_pattern_detection_time'] = (
                self.performance_stats['avg_pattern_detection_time'] * 0.9 + detection_time * 0.1
            )
            self.performance_stats['total_analyses'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting patterns for {symbol}: {e}")
            # Return empty result
            return PatternDetectionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                patterns=[],
                total_patterns=0,
                high_confidence_patterns=0,
                execution_time_ms=0.0
            )
            
    async def analyze_symbol_with_ai(self, symbol: str, stock_data: pd.DataFrame, timeframe: str = '5m') -> Dict[str, Any]:
        """Complete AI analysis for a symbol"""
        
        start_time = datetime.now()
        
        try:
            # Step 1: Create features
            feature_set = self.create_features_for_symbol(symbol, stock_data, timeframe)
            
            if feature_set.features.empty:
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'status': 'error',
                    'message': 'No features could be created',
                    'patterns': [],
                    'feature_count': 0,
                    'analysis_time_ms': 0
                }
                
            # Step 2: Detect patterns
            pattern_result = self.detect_patterns_for_symbol(symbol, feature_set, timeframe)
            
            # Step 3: Prepare response
            analysis_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = {
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'success',
                'timestamp': start_time.isoformat(),
                'feature_count': len(feature_set.feature_names),
                'patterns': [
                    {
                        'concept_name': p.concept_name,
                        'concept_number': p.concept_number,
                        'confidence': p.confidence,
                        'pattern_type': p.pattern_type,
                        'strength': p.strength,
                        'supporting_evidence': p.supporting_evidence,
                        'timestamp': p.timestamp.isoformat()
                    } for p in pattern_result.patterns
                ],
                'pattern_summary': {
                    'total_patterns': pattern_result.total_patterns,
                    'high_confidence_patterns': pattern_result.high_confidence_patterns,
                    'avg_confidence': np.mean([p.confidence for p in pattern_result.patterns]) if pattern_result.patterns else 0.0
                },
                'performance': {
                    'analysis_time_ms': analysis_time,
                    'feature_creation_time_ms': self.performance_stats['avg_feature_creation_time'],
                    'pattern_detection_time_ms': pattern_result.execution_time_ms
                },
                'ai_metrics': self.pattern_detector.get_detection_metrics()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'error',
                'message': str(e),
                'patterns': [],
                'feature_count': 0,
                'analysis_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
            
    async def train_models_for_symbols(self, symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, ModelPerformance]:
        """Train AI models on historical data"""
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 years ago
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Starting model training for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Create training configuration
        config = TrainingConfig(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframes=['5m', '1h', '1d'],
            sequence_length=50,
            epochs=50,  # Reduced for demo
            batch_size=32,
            learning_rate=0.001
        )
        
        # Initialize trainer
        trainer = ModelTrainer(config, str(self.model_dir))
        
        try:
            # Train all models
            performances = trainer.train_all_models()
            
            # Reload the trained models in pattern detector
            self.pattern_detector.load_models()
            
            logger.info("Model training completed successfully")
            return performances
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {}
            
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from trained models"""
        
        try:
            importance_data = {}
            
            # Random Forest feature importance
            if 'random_forest' in self.pattern_detector.models:
                rf_model = self.pattern_detector.models['random_forest']
                if hasattr(rf_model, 'feature_importances_'):
                    importance_data['random_forest'] = rf_model.feature_importances_.tolist()
                    
            return {
                'status': 'success',
                'feature_importance': importance_data,
                'total_features': len(importance_data.get('random_forest', []))
            }
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'feature_importance': {},
                'total_features': 0
            }
            
    def get_ai_performance_stats(self) -> Dict[str, Any]:
        """Get AI performance statistics"""
        
        return {
            'feature_engine': self.performance_stats,
            'pattern_detection': self.pattern_detector.get_detection_metrics(),
            'cache_stats': {
                'feature_cache_size': len(self.feature_cache),
                'pattern_cache_size': len(self.pattern_cache)
            },
            'model_info': {
                'models_loaded': len(self.pattern_detector.models),
                'is_trained': self.pattern_detector.is_trained
            }
        }
        
    def clear_cache(self):
        """Clear feature and pattern caches"""
        self.feature_cache.clear()
        self.pattern_cache.clear()
        logger.info("AI caches cleared")

# Global AI integration instance
ai_engine = AIIntegrationEngine()

# Export main classes and instance
__all__ = ['AIIntegrationEngine', 'ai_engine']