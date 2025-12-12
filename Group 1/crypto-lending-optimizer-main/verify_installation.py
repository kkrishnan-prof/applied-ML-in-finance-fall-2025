"""
Verification Script - Check if all components are working
Run this after setup to verify the system is ready
"""

import sys
from pathlib import Path
from loguru import logger
import importlib

def check_file_exists(filepath: str, description: str) -> bool:
    """Check if a file exists"""
    if Path(filepath).exists():
        logger.info(f"✅ {description}: {filepath}")
        return True
    else:
        logger.error(f"❌ {description} NOT FOUND: {filepath}")
        return False

def check_import(module_name: str) -> bool:
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        logger.info(f"✅ Module '{module_name}' can be imported")
        return True
    except ImportError as e:
        logger.error(f"❌ Module '{module_name}' CANNOT be imported: {e}")
        return False

def main():
    logger.info("\n" + "="*80)
    logger.info("SYSTEM VERIFICATION - Hybrid Lending Rate Optimization")
    logger.info("="*80 + "\n")

    all_good = True

    # Check data files
    logger.info("📁 Checking Data Files...")
    all_good &= check_file_exists("data/raw/btcusd_1-min_data.csv", "Raw data")
    all_good &= check_file_exists("data/processed/processed_data.parquet", "Processed data")

    # Check models
    logger.info("\n🤖 Checking Models...")
    all_good &= check_file_exists("models/volatility_model_lightgbm.pkl", "Volatility model")
    all_good &= check_file_exists("models/revenue_model.pkl", "Revenue model")
    all_good &= check_file_exists("models/volatility_model_metadata.json", "Volatility metadata")
    all_good &= check_file_exists("models/revenue_model_metadata.json", "Revenue metadata")

    # Check source files
    logger.info("\n📝 Checking Source Code...")
    all_good &= check_file_exists("src/data_processor.py", "Data processor")
    all_good &= check_file_exists("src/volatility_models.py", "Volatility models")
    all_good &= check_file_exists("src/model_evaluator.py", "Model evaluator")
    all_good &= check_file_exists("src/revenue_optimizer.py", "Revenue optimizer")
    all_good &= check_file_exists("src/rate_calculator.py", "Rate calculator")

    # Check API
    logger.info("\n🌐 Checking API...")
    all_good &= check_file_exists("api/main.py", "FastAPI application")

    # Check documentation
    logger.info("\n📚 Checking Documentation...")
    all_good &= check_file_exists("README.md", "README")
    all_good &= check_file_exists("IMPLEMENTATION_STATUS.md", "Implementation status")
    all_good &= check_file_exists("NEXTJS_INTEGRATION_GUIDE.md", "NextJS guide")
    all_good &= check_file_exists("EXECUTIVE_SUMMARY.md", "Executive summary")
    all_good &= check_file_exists("config.yaml", "Configuration")
    all_good &= check_file_exists("requirements.txt", "Requirements")

    # Check Python dependencies
    logger.info("\n📦 Checking Python Dependencies...")
    required_modules = [
        'pandas',
        'numpy',
        'scipy',
        'sklearn',
        'lightgbm',
        'xgboost',
        'torch',
        'fastapi',
        'uvicorn',
        'pydantic',
        'yaml',
        'loguru',
        'joblib'
    ]

    for module in required_modules:
        all_good &= check_import(module)

    # Test model loading
    logger.info("\n🔧 Testing Model Loading...")
    try:
        sys.path.insert(0, str(Path.cwd()))
        from src.rate_calculator import HybridRateCalculator

        calculator = HybridRateCalculator()
        logger.info("✅ Hybrid rate calculator initialized successfully")

        # Test with dummy data
        import numpy as np
        dummy_features = np.random.randn(41)
        result = calculator.calculate_rate(dummy_features, leverage=10.0)

        logger.info(f"✅ Rate calculation works! Sample rate: {result['final_rate_daily']*100:.4f}%")

    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        all_good = False

    # Final verdict
    logger.info("\n" + "="*80)
    if all_good:
        logger.info("✅✅✅ ALL CHECKS PASSED - SYSTEM READY! ✅✅✅")
        logger.info("\nYou can now:")
        logger.info("  1. Start the API: PYTHONPATH=. python api/main.py")
        logger.info("  2. Test the API: python test_api.py")
        logger.info("  3. View docs: http://localhost:8000/docs")
    else:
        logger.error("❌ SOME CHECKS FAILED - Please review errors above")
        logger.info("\nTo fix:")
        logger.info("  1. Install dependencies: pip install -r requirements.txt")
        logger.info("  2. Process data: python src/data_processor.py")
        logger.info("  3. Train models: python run_fast_model_comparison.py && python train_revenue_model.py")

    logger.info("="*80 + "\n")

    return all_good


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
