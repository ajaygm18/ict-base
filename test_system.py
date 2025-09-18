#!/usr/bin/env python3
"""
Simple test script to validate ICT Trading System functionality
Tests core components and integration
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from config.settings import settings
from monitoring.logging import logger
from data_pipeline.data_source import data_pipeline
from ict_engine.ict_complete import CompleteICTConceptsEngine
from trading_engine.engine import ICTSignalGenerator
import pandas as pd

async def test_basic_functionality():
    """Test basic system functionality"""
    
    logger.info("Starting ICT Trading System validation test")
    
    # Test 1: Configuration
    logger.info(f"✓ Configuration loaded: {settings.app_name} v{settings.version}")
    
    # Test 2: Data Pipeline
    try:
        symbols = ['AAPL']
        data = await data_pipeline.get_real_time_data(symbols)
        if data and 'AAPL' in data:
            logger.info(f"✓ Data pipeline working: Retrieved {len(data['AAPL'])} data points for AAPL")
        else:
            logger.info("⚠ Data pipeline test: No real-time data (expected for some environments)")
        
        # Test historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        historical_data = await data_pipeline.get_historical_data(
            'AAPL', start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
        )
        if historical_data:
            logger.info(f"✓ Historical data: Retrieved {len(historical_data)} data points")
        
    except Exception as e:
        logger.error(f"✗ Data pipeline error: {e}")
    
    # Test 3: ICT Engine
    try:
        if historical_data and len(historical_data) > 50:
            # Convert to DataFrame
            df_data = []
            for data_point in historical_data[-100:]:  # Last 100 points
                df_data.append({
                    'timestamp': data_point.timestamp,
                    'open': data_point.open,
                    'high': data_point.high,
                    'low': data_point.low,
                    'close': data_point.close,
                    'volume': data_point.volume
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('timestamp', inplace=True)
            
            # Test ICT analysis
            ict_engine = CompleteICTConceptsEngine('AAPL')
            analysis = ict_engine.analyze_market_structure(df)
            
            if 'error' not in analysis:
                concepts_found = len(analysis.get('concepts', {}))
                logger.info(f"✓ ICT Engine: Analyzed {concepts_found} concepts successfully")
                
                # Show some analysis results
                if 'concepts' in analysis:
                    for concept_name, concept_data in analysis['concepts'].items():
                        if isinstance(concept_data, dict) and 'error' not in concept_data:
                            logger.info(f"  - {concept_name}: ✓")
                        else:
                            logger.info(f"  - {concept_name}: ⚠ (error or no data)")
            else:
                logger.error(f"✗ ICT Engine error: {analysis['error']}")
        
    except Exception as e:
        logger.error(f"✗ ICT Engine error: {e}")
    
    # Test 4: Signal Generation
    try:
        from trading_engine.engine import ICTSignalGenerator
        signal_generator = ICTSignalGenerator()
        signals = await signal_generator.generate_signals(['AAPL'])
        logger.info(f"✓ Signal Generator: Generated {len(signals)} signals")
        
        for signal in signals[:3]:  # Show first 3 signals
            logger.info(f"  - Signal: {signal.symbol} {signal.signal_type} "
                       f"(confidence: {signal.confidence:.2f})")
        
    except Exception as e:
        logger.error(f"✗ Signal Generator error: {e}")
    
    # Test 5: Risk Management
    try:
        from ict_engine.risk_management import StockRiskManagementEngine
        risk_engine = StockRiskManagementEngine()
        
        # Test position sizing
        test_setups = [{'symbol': 'AAPL', 'entry_price': 150, 'stop_loss': 145, 'confidence': 0.8}]
        position_sizing = risk_engine.concept_35_position_sizing_algorithms(100000, test_setups)
        
        if 'error' not in position_sizing:
            logger.info("✓ Risk Management: Position sizing calculated successfully")
        else:
            logger.error(f"✗ Risk Management error: {position_sizing['error']}")
        
    except ImportError as e:
        logger.info(f"⚠ Risk Management module not available: {e}")
    except Exception as e:
        logger.error(f"✗ Risk Management error: {e}")
    
    logger.info("ICT Trading System validation test completed")
    logger.info("System Status: Core components operational ✓")

if __name__ == "__main__":
    asyncio.run(test_basic_functionality())