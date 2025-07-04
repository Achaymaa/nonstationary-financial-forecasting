#!/usr/bin/env python3
"""
🚀 MULTI-HORIZON EVALUATION RUNNER
Short-run (96 steps) vs Long-run (136 steps) Forecasting Comparison

This script runs comprehensive evaluation of RNN and AEFIN models
for both short-run and long-run financial forecasting horizons.

Metrics calculated:
- MAPE (Mean Absolute Percentage Error)
- SMAPE (Symmetric Mean Absolute Percentage Error)
- RMSE (Root Mean Square Error)
- RMSP (Root Mean Square Percentage Error)
- MASE (Mean Absolute Scaled Error)

Output:
- Comprehensive comparison table
- Multi-panel visualization plots
- Detailed CSV results
"""

import os
import sys

def main():
    """Run the multi-horizon evaluation."""
    
    print("=" * 70)
    print("🚀 MULTI-HORIZON FINANCIAL FORECASTING EVALUATION")
    print("=" * 70)
    print("📊 Short Run: 96 steps ahead")
    print("📈 Long Run: 136 steps ahead")
    print("🤖 Models: RNN vs AEFIN")
    print("🔬 Metrics: MAPE, SMAPE, RMSE, RMSP, MASE")
    print("")
    
    # Check if required directories exist
    required_dirs = ['models/rnn', 'models/aefin', 'evaluation', 'output/results', 'output/plots']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            print(f"❌ Missing directory: {dir_path}")
            return
        else:
            print(f"✅ Found: {dir_path}")
    
    print("\n" + "=" * 70)
    print("🏃 STARTING EVALUATION...")
    print("=" * 70)
    
    try:
        # Import and run the evaluation
        from evaluation.multi_horizon_evaluation import main as run_evaluation
        run_evaluation()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n🔧 SETUP INSTRUCTIONS:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Ensure all model files are in place")
        print("3. Check Python environment setup")
        
    except Exception as e:
        print(f"❌ Execution Error: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check if PyTorch is installed: pip install torch")
        print("2. Verify model files exist in models/ directory")
        print("3. Ensure sufficient memory for training")


if __name__ == "__main__":
    main() 