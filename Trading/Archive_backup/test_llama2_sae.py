"""
Test script for Llama-2-7b-hf SAE Trading Analysis
Simple test to verify the functionality works
"""

import sys
import os
import logging

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_simple_trading():
    """Test the simple trading analyzer"""
    try:
        from llama2_sae_simple_trading import LlamaSAESimpleTrading
        
        logger.info("Testing LlamaSAESimpleTrading...")
        
        # Initialize analyzer
        analyzer = LlamaSAESimpleTrading()
        
        if analyzer.sae_model is None:
            logger.error("SAE model not loaded. This is expected if the model path doesn't exist.")
            return False
        
        # Test with a simple financial text
        test_text = "Apple reports record quarterly earnings"
        analysis = analyzer.analyze_financial_text(test_text)
        
        if 'error' not in analysis:
            logger.info(f"✓ Simple trading analysis successful")
            logger.info(f"  Found {analysis['total_activations']} activations")
            logger.info(f"  Max activation: {analysis['max_activation']:.4f}")
            return True
        else:
            logger.error(f"✗ Simple trading analysis failed: {analysis['error']}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Simple trading test failed: {str(e)}")
        return False

def test_integrated_trading():
    """Test the integrated trading analyzer"""
    try:
        from llama2_sae_integrated_trading import LlamaSAEIntegratedTrading
        
        logger.info("Testing LlamaSAEIntegratedTrading...")
        
        # Initialize analyzer
        analyzer = LlamaSAEIntegratedTrading()
        
        if analyzer.sae_model is None:
            logger.error("SAE model not loaded. This is expected if the model path doesn't exist.")
            return False
        
        # Test with simple headlines
        test_headlines = [
            "Apple reports record quarterly earnings",
            "Tesla stock surges 15% after positive earnings report"
        ]
        
        # Test headline analysis
        top_features = analyzer.analyze_headlines_for_features(test_headlines, top_n=5)
        
        if len(top_features) > 0:
            logger.info(f"✓ Integrated trading analysis successful")
            logger.info(f"  Found {len(top_features)} features")
            logger.info(f"  Top feature: {top_features.iloc[0]['feature_label']}")
            return True
        else:
            logger.error("✗ Integrated trading analysis failed: No features found")
            return False
            
    except Exception as e:
        logger.error(f"✗ Integrated trading test failed: {str(e)}")
        return False

def test_model_path():
    """Test if the model path exists"""
    model_path = "/home/nvidia/Documents/Hariom/saetrain/trained_models/llama2_7b_hf_layers4 10 16 22 28_k32_latents400_wikitext103_torchrun"
    
    if os.path.exists(model_path):
        logger.info(f"✓ Model path exists: {model_path}")
        
        # Check for SAE weights
        sae_path = os.path.join(model_path, "layers.16")
        sae_weights_path = os.path.join(sae_path, "sae.safetensors")
        
        if os.path.exists(sae_weights_path):
            logger.info(f"✓ SAE weights found: {sae_weights_path}")
            return True
        else:
            logger.error(f"✗ SAE weights not found: {sae_weights_path}")
            return False
    else:
        logger.error(f"✗ Model path does not exist: {model_path}")
        return False

def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("Testing Llama-2-7b-hf SAE Trading Analysis")
    logger.info("="*60)
    
    # Test 1: Check model path
    logger.info("\n1. Testing model path...")
    model_path_ok = test_model_path()
    
    # Test 2: Simple trading analyzer
    logger.info("\n2. Testing simple trading analyzer...")
    simple_ok = test_simple_trading()
    
    # Test 3: Integrated trading analyzer
    logger.info("\n3. Testing integrated trading analyzer...")
    integrated_ok = test_integrated_trading()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    logger.info(f"Model path exists: {'✓' if model_path_ok else '✗'}")
    logger.info(f"Simple trading: {'✓' if simple_ok else '✗'}")
    logger.info(f"Integrated trading: {'✓' if integrated_ok else '✗'}")
    
    if model_path_ok and simple_ok and integrated_ok:
        logger.info("\n🎉 All tests passed! The SAE trading analysis is working correctly.")
    else:
        logger.info("\n⚠️  Some tests failed. Check the error messages above.")
    
    logger.info("="*60)

if __name__ == "__main__":
    main()










