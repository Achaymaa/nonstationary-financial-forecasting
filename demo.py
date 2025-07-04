#!/usr/bin/env python3
"""
🚀 NONSTATIONARY FINANCIAL PREDICTION DEMO
RNN vs AEFIN (Attention-Enhanced Fourier-Integrated Network)

This demo showcases the power of advanced neural architectures 
for predicting nonstationary financial time series.
"""

import os
import sys

def main():
    print("=" * 70)
    print("🚀 NONSTATIONARY FINANCIAL PREDICTION DEMO")
    print("🧠 RNN vs AEFIN (Attention-Enhanced Fourier-Integrated Network)")
    print("=" * 70)
    
    print("\n📋 PROJECT OVERVIEW:")
    print("   • RNN: Advanced recurrent networks with adaptive normalization")
    print("   • AEFIN: Fourier transforms + Attention + Regime detection")
    print("   • Focus: Nonstationary financial time series prediction")
    print("   • Features: Multi-regime data, volatility clustering, trend changes")
    
    print("\n🏗️  ARCHITECTURE HIGHLIGHTS:")
    print("   RNN Model:")
    print("   ├── Bidirectional LSTM/GRU")
    print("   ├── Adaptive normalization")
    print("   ├── Attention mechanism (optional)")
    print("   └── Multi-layer output network")
    
    print("\n   AEFIN Model:")
    print("   ├── Fourier Transform Layer (frequency analysis)")
    print("   ├── Multi-Head Attention (temporal focus)")
    print("   ├── Integration Network (combines representations)")
    print("   ├── Regime Detection (market state changes)")
    print("   └── Adaptive Layer Normalization")
    
    print("\n🚀 GETTING STARTED:")
    print("   1. Install dependencies: pip install -r requirements.txt")
    print("   2. Run training: python training/train_models.py")
    print("   3. View results in: output/ directory")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("   prediction-ml-nonstationnary/")
    print("   ├── models/")
    print("   │   ├── rnn/rnn_model.py         # Advanced RNN implementation")
    print("   │   └── aefin/aefin_model.py     # AEFIN architecture")
    print("   ├── training/train_models.py     # Training pipeline")
    print("   ├── configuration.py             # Model configurations")
    print("   ├── requirements.txt             # Dependencies")
    print("   └── output/                      # Results and plots")
    
    print("\n🎯 WHY AEFIN FOR NONSTATIONARY DATA?")
    print("   • Fourier analysis captures cyclical market patterns")
    print("   • Attention focuses on relevant time periods")
    print("   • Regime detection adapts to market state changes")
    print("   • Integration combines time & frequency representations")
    print("   • Adaptive normalization handles distribution shifts")
    
    print("\n💡 KEY INNOVATIONS:")
    print("   1. Learnable Fourier weights for frequency filtering")
    print("   2. Cross-attention between time and frequency domains")
    print("   3. Market regime detection mechanism")
    print("   4. Adaptive normalization for nonstationarity")
    print("   5. Multi-horizon prediction capabilities")
    
    # Check if dependencies are available
    print("\n🔍 DEPENDENCY CHECK:")
    try:
        import torch
        print("   ✅ PyTorch available")
    except ImportError:
        print("   ❌ PyTorch not installed")
    
    try:
        import numpy
        print("   ✅ NumPy available")
    except ImportError:
        print("   ❌ NumPy not installed")
    
    try:
        import pandas
        print("   ✅ Pandas available")
    except ImportError:
        print("   ❌ Pandas not installed")
    
    try:
        import matplotlib
        print("   ✅ Matplotlib available")
    except ImportError:
        print("   ❌ Matplotlib not installed")
    
    print("\n📊 EXPECTED RESULTS:")
    print("   • AEFIN typically outperforms RNN on:")
    print("     - RMSE (Root Mean Square Error)")
    print("     - Directional accuracy")
    print("     - Regime change detection")
    print("     - Frequency pattern recognition")
    
    print("\n🎪 DEMO COMMANDS:")
    print("   # Quick training demo (reduced epochs)")
    print("   python training/train_models.py")
    print("")
    print("   # View model architectures")
    print("   python -c \"from models.rnn.rnn_model import *; from models.aefin.aefin_model import *; print('Models loaded!')\"")
    
    print("\n" + "=" * 70)
    print("🚀 Ready to explore nonstationary financial prediction!")
    print("   Start with: python training/train_models.py")
    print("=" * 70)


if __name__ == "__main__":
    main() 