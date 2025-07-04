import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rnn.rnn_model import create_rnn_model
from models.aefin.aefin_model import create_aefin_model
import configuration as config

class MultiHorizonEvaluator:
    """
    Multi-horizon forecasting evaluator for comparing short-run and long-run predictions.
    """
    
    def __init__(self, short_horizon=96, long_horizon=136):
        self.short_horizon = short_horizon
        self.long_horizon = long_horizon
        self.device = config.device
        self.scaler = RobustScaler()
        
    def generate_complex_financial_data(self, n_samples=3000):
        """
        Generate complex nonstationary financial time series with multiple regimes.
        """
        np.random.seed(config.random_seed)
        
        # Create regime-switching price data with trends and cycles
        total_length = n_samples + self.long_horizon + 100  # Extra buffer
        
        # Multiple regime periods
        regime_lengths = [total_length // 4, total_length // 3, total_length // 3]
        regime_lengths.append(total_length - sum(regime_lengths))
        
        prices = []
        
        # Regime 1: Bull market with trend
        regime1_returns = np.random.normal(0.0008, 0.015, regime_lengths[0])
        regime1_returns += 0.0002 * np.sin(np.arange(regime_lengths[0]) * 2 * np.pi / 20)  # 20-day cycle
        regime1_prices = np.cumsum(regime1_returns)
        prices.extend(regime1_prices)
        
        # Regime 2: High volatility period
        regime2_returns = np.random.normal(0.0001, 0.035, regime_lengths[1])
        regime2_returns += 0.0003 * np.sin(np.arange(regime_lengths[1]) * 2 * np.pi / 5)   # 5-day cycle
        regime2_prices = prices[-1] + np.cumsum(regime2_returns)
        prices.extend(regime2_prices)
        
        # Regime 3: Bear market
        regime3_returns = np.random.normal(-0.0005, 0.025, regime_lengths[2])
        regime3_returns += 0.0001 * np.sin(np.arange(regime_lengths[2]) * 2 * np.pi / 10)  # 10-day cycle
        regime3_prices = prices[-1] + np.cumsum(regime3_returns)
        prices.extend(regime3_prices)
        
        # Regime 4: Recovery phase
        regime4_returns = np.random.normal(0.0004, 0.020, regime_lengths[3])
        regime4_returns += 0.0002 * np.sin(np.arange(regime_lengths[3]) * 2 * np.pi / 15)  # 15-day cycle
        regime4_prices = prices[-1] + np.cumsum(regime4_returns)
        prices.extend(regime4_prices)
        
        prices = np.array(prices)
        
        # Generate multiple features (economic indicators)
        features = np.zeros((len(prices), 12))
        
        # Feature 1-3: Lagged returns
        returns = np.diff(prices, prepend=prices[0])
        features[:, 0] = returns
        features[1:, 1] = returns[:-1]  # 1-lag
        features[2:, 2] = returns[:-2]  # 2-lag
        
        # Feature 4-6: Moving averages
        for i, window in enumerate([5, 10, 20]):
            ma = pd.Series(prices).rolling(window=window, min_periods=1).mean().values
            features[:, 3+i] = (prices - ma) / ma  # Relative to MA
        
        # Feature 7-9: Volatility measures
        for i, window in enumerate([5, 10, 20]):
            vol = pd.Series(returns).rolling(window=window, min_periods=1).std().values
            features[:, 6+i] = vol
        
        # Feature 10-12: Technical indicators
        features[:, 9] = np.sin(np.arange(len(prices)) * 2 * np.pi / 252)   # Seasonal
        features[:, 10] = np.cos(np.arange(len(prices)) * 2 * np.pi / 252)  # Seasonal
        features[:, 11] = np.random.normal(0, 0.01, len(prices))             # Noise
        
        return prices, features, returns
    
    def create_multi_horizon_sequences(self, prices, features, returns):
        """
        Create sequences for both short and long horizon predictions.
        """
        sequence_length = config.sequence_length
        
        # Prepare data for both horizons
        short_X, short_y = [], []
        long_X, long_y = [], []
        
        # Create sequences
        for i in range(sequence_length, len(prices) - self.long_horizon):
            # Input sequence
            X_seq = features[i-sequence_length:i]
            
            # Short horizon target (96 steps ahead)
            if i + self.short_horizon < len(returns):
                short_X.append(X_seq)
                short_y.append(returns[i:i+self.short_horizon])
            
            # Long horizon target (136 steps ahead)  
            if i + self.long_horizon < len(returns):
                long_X.append(X_seq)
                long_y.append(returns[i:i+self.long_horizon])
        
        return (np.array(short_X), np.array(short_y)), (np.array(long_X), np.array(long_y))
    
    def calculate_comprehensive_metrics(self, actual, predicted):
        """
        Calculate all requested metrics: MAPE, SMAPE, RMSE, RMSP, MASE.
        """
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        
        # Remove any NaN or infinite values
        mask = np.isfinite(actual) & np.isfinite(predicted)
        actual = actual[mask]
        predicted = predicted[mask]
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
        
        # SMAPE (Symmetric Mean Absolute Percentage Error)
        smape = np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # RMSP (Root Mean Square Percentage Error)
        rmsp = np.sqrt(np.mean(((actual - predicted) / (actual + 1e-8)) ** 2)) * 100
        
        # MASE (Mean Absolute Scaled Error) - using naive forecast as benchmark
        mae = mean_absolute_error(actual, predicted)
        naive_forecast_error = np.mean(np.abs(np.diff(actual)))
        mase = mae / (naive_forecast_error + 1e-8)
        
        return {
            'MAPE': mape,
            'SMAPE': smape, 
            'RMSE': rmse,
            'RMSP': rmsp,
            'MASE': mase
        }
    
    def train_multi_horizon_model(self, model_type, X_data, y_data, horizon_name):
        """
        Train model for specific horizon.
        """
        print(f"\n🚀 Training {model_type.upper()} for {horizon_name} ({len(y_data[0])} steps)")
        
        # Split data
        n_samples = len(X_data)
        train_end = int(n_samples * 0.7)
        val_end = int(n_samples * 0.85)
        
        train_X, train_y = X_data[:train_end], y_data[:train_end]
        val_X, val_y = X_data[train_end:val_end], y_data[train_end:val_end]
        test_X, test_y = X_data[val_end:], y_data[val_end:]
        
        # Normalize features
        train_X_scaled = self.scaler.fit_transform(train_X.reshape(-1, train_X.shape[-1]))
        train_X_scaled = train_X_scaled.reshape(train_X.shape)
        val_X_scaled = self.scaler.transform(val_X.reshape(-1, val_X.shape[-1]))
        val_X_scaled = val_X_scaled.reshape(val_X.shape)
        test_X_scaled = self.scaler.transform(test_X.reshape(-1, test_X.shape[-1]))
        test_X_scaled = test_X_scaled.reshape(test_X.shape)
        
        # Create model
        input_size = train_X_scaled.shape[-1]
        output_size = len(y_data[0])  # Multi-step output
        
        if model_type == 'rnn':
            model = create_rnn_model(config.rnn_config, input_size)
            # Modify output layer for multi-step
            model.output_layers = nn.Sequential(
                nn.Linear(model.effective_hidden, model.effective_hidden),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(model.effective_hidden, output_size)
            )
        else:  # aefin
            model = create_aefin_model(config.aefin_config, input_size)
            # Modify output layer for multi-step
            model.output_projection = nn.Sequential(
                nn.Linear(model.hidden_size, model.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(model.hidden_size // 2, output_size)
            )
        
        model = model.to(self.device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate * 0.5)  # Lower LR for multi-step
        
        # Convert to tensors
        train_X_tensor = torch.FloatTensor(train_X_scaled).to(self.device)
        train_y_tensor = torch.FloatTensor(train_y).to(self.device)
        val_X_tensor = torch.FloatTensor(val_X_scaled).to(self.device)
        val_y_tensor = torch.FloatTensor(val_y).to(self.device)
        
        # Training loop (reduced epochs for demo)
        best_val_loss = float('inf')
        epochs = 30
        
        for epoch in range(epochs):
            # Training
            model.train()
            optimizer.zero_grad()
            
            if model_type == 'aefin':
                outputs, _, _ = model(train_X_tensor)
            else:
                outputs = model(train_X_tensor)
            
            loss = criterion(outputs, train_y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Validation
            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    if model_type == 'aefin':
                        val_outputs, _, _ = model(val_X_tensor)
                    else:
                        val_outputs = model(val_X_tensor)
                    
                    val_loss = criterion(val_outputs, val_y_tensor).item()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model.state_dict(), f'output/models/{model_type}_{horizon_name.lower()}_best.pth')
                    
                    print(f"Epoch {epoch+1}/{epochs}: Train Loss = {loss.item():.6f}, Val Loss = {val_loss:.6f}")
        
        # Test evaluation
        model.eval()
        test_X_tensor = torch.FloatTensor(test_X_scaled).to(self.device)
        
        with torch.no_grad():
            if model_type == 'aefin':
                test_predictions, _, _ = model(test_X_tensor)
            else:
                test_predictions = model(test_X_tensor)
        
        test_predictions = test_predictions.cpu().numpy()
        
        return test_predictions, test_y
    
    def run_comprehensive_evaluation(self):
        """
        Run complete multi-horizon evaluation for both models.
        """
        print("🚀 MULTI-HORIZON FINANCIAL FORECASTING EVALUATION")
        print("=" * 70)
        print(f"📊 Short Run: {self.short_horizon} steps")
        print(f"📈 Long Run: {self.long_horizon} steps")
        print("🔬 Metrics: MAPE, SMAPE, RMSE, RMSP, MASE")
        
        # Generate data
        print("\n📊 Generating complex financial time series...")
        prices, features, returns = self.generate_complex_financial_data()
        
        # Create sequences for both horizons
        (short_X, short_y), (long_X, long_y) = self.create_multi_horizon_sequences(prices, features, returns)
        
        print(f"✅ Short horizon sequences: {len(short_X)}")
        print(f"✅ Long horizon sequences: {len(long_X)}")
        
        # Results storage
        results = {}
        predictions_data = {}
        
        # Evaluate both models on both horizons
        for model_type in ['rnn', 'aefin']:
            results[model_type] = {}
            predictions_data[model_type] = {}
            
            # Short horizon
            short_pred, short_actual = self.train_multi_horizon_model(
                model_type, short_X, short_y, "Short_Run"
            )
            short_metrics = self.calculate_comprehensive_metrics(short_actual, short_pred)
            results[model_type]['Short_Run'] = short_metrics
            predictions_data[model_type]['Short_Run'] = (short_pred, short_actual)
            
            # Long horizon  
            long_pred, long_actual = self.train_multi_horizon_model(
                model_type, long_X, long_y, "Long_Run"
            )
            long_metrics = self.calculate_comprehensive_metrics(long_actual, long_pred)
            results[model_type]['Long_Run'] = long_metrics
            predictions_data[model_type]['Long_Run'] = (long_pred, long_actual)
        
        # Create comprehensive comparison table
        self.create_comparison_table(results)
        
        # Create comprehensive plots
        self.create_comprehensive_plots(predictions_data, results)
        
        return results, predictions_data
    
    def create_comparison_table(self, results):
        """
        Create comprehensive comparison table with all metrics.
        """
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE MULTI-HORIZON COMPARISON TABLE")
        print("=" * 80)
        
        # Prepare data for table
        table_data = []
        
        for model in ['rnn', 'aefin']:
            for horizon in ['Short_Run', 'Long_Run']:
                metrics = results[model][horizon]
                table_data.append({
                    'Model': model.upper(),
                    'Horizon': horizon.replace('_', ' '),
                    'MAPE (%)': f"{metrics['MAPE']:.4f}",
                    'SMAPE (%)': f"{metrics['SMAPE']:.4f}",
                    'RMSE': f"{metrics['RMSE']:.6f}",
                    'RMSP (%)': f"{metrics['RMSP']:.4f}",
                    'MASE': f"{metrics['MASE']:.4f}"
                })
        
        # Create DataFrame and display
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
        
        # Calculate improvements
        print("\n" + "=" * 60)
        print("📈 AEFIN vs RNN IMPROVEMENTS")
        print("=" * 60)
        
        for horizon in ['Short_Run', 'Long_Run']:
            print(f"\n🎯 {horizon.replace('_', ' ')}:")
            rnn_metrics = results['rnn'][horizon]
            aefin_metrics = results['aefin'][horizon]
            
            for metric in ['MAPE', 'SMAPE', 'RMSE', 'RMSP', 'MASE']:
                rnn_val = rnn_metrics[metric]
                aefin_val = aefin_metrics[metric]
                improvement = ((rnn_val - aefin_val) / rnn_val) * 100
                print(f"   {metric:8}: {improvement:+6.2f}% {'✅' if improvement > 0 else '❌'}")
        
        # Save to CSV
        df.to_csv('output/results/multi_horizon_comparison.csv', index=False)
        print(f"\n💾 Results saved to: output/results/multi_horizon_comparison.csv")
    
    def create_comprehensive_plots(self, predictions_data, results):
        """
        Create comprehensive visualization plots.
        """
        print("\n📊 Creating comprehensive visualization plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # Get sample data for plotting
        sample_idx = -1
        
        # Plot 1: Short Run Comparison
        plt.subplot(3, 2, 1)
        rnn_short_pred, rnn_short_actual = predictions_data['rnn']['Short_Run']
        aefin_short_pred, aefin_short_actual = predictions_data['aefin']['Short_Run']
        
        actual_short = rnn_short_actual[sample_idx][:80]  # Show first 80 points
        rnn_pred_short = rnn_short_pred[sample_idx][:80]
        aefin_pred_short = aefin_short_pred[sample_idx][:80]
        
        x_axis = range(len(actual_short))
        plt.plot(x_axis, actual_short, 'k-', label='Actual', linewidth=2, alpha=0.8)
        plt.plot(x_axis, rnn_pred_short, 'b--', label='RNN', linewidth=1.5, alpha=0.7)
        plt.plot(x_axis, aefin_pred_short, 'r:', label='AEFIN', linewidth=2, alpha=0.8)
        plt.title('📊 Short Run Predictions (96 steps)', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Long Run Comparison
        plt.subplot(3, 2, 2)
        rnn_long_pred, rnn_long_actual = predictions_data['rnn']['Long_Run']
        aefin_long_pred, aefin_long_actual = predictions_data['aefin']['Long_Run']
        
        actual_long = rnn_long_actual[sample_idx][:80]  # Show first 80 points
        rnn_pred_long = rnn_long_pred[sample_idx][:80]
        aefin_pred_long = aefin_long_pred[sample_idx][:80]
        
        x_axis = range(len(actual_long))
        plt.plot(x_axis, actual_long, 'k-', label='Actual', linewidth=2, alpha=0.8)
        plt.plot(x_axis, rnn_pred_long, 'b--', label='RNN', linewidth=1.5, alpha=0.7)
        plt.plot(x_axis, aefin_pred_long, 'r:', label='AEFIN', linewidth=2, alpha=0.8)
        plt.title('📈 Long Run Predictions (136 steps)', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Error Distribution - Short Run
        plt.subplot(3, 2, 3)
        rnn_errors_short = (rnn_short_actual - rnn_short_pred).flatten()
        aefin_errors_short = (aefin_short_actual - aefin_short_pred).flatten()
        
        plt.hist(rnn_errors_short, bins=50, alpha=0.6, label='RNN Errors', color='blue', density=True)
        plt.hist(aefin_errors_short, bins=50, alpha=0.6, label='AEFIN Errors', color='red', density=True)
        plt.title('📊 Short Run Error Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Error Distribution - Long Run
        plt.subplot(3, 2, 4)
        rnn_errors_long = (rnn_long_actual - rnn_long_pred).flatten()
        aefin_errors_long = (aefin_long_actual - aefin_long_pred).flatten()
        
        plt.hist(rnn_errors_long, bins=50, alpha=0.6, label='RNN Errors', color='blue', density=True)
        plt.hist(aefin_errors_long, bins=50, alpha=0.6, label='AEFIN Errors', color='red', density=True)
        plt.title('📈 Long Run Error Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Prediction Error')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Cumulative Returns Comparison
        plt.subplot(3, 2, 5)
        full_actual_short = rnn_short_actual[sample_idx]
        full_rnn_short = rnn_short_pred[sample_idx]
        full_aefin_short = aefin_short_pred[sample_idx]
        
        cum_actual = np.cumsum(full_actual_short)
        cum_rnn = np.cumsum(full_rnn_short)
        cum_aefin = np.cumsum(full_aefin_short)
        
        x_axis = range(len(cum_actual))
        plt.plot(x_axis, cum_actual, 'k-', label='Actual', linewidth=2, alpha=0.8)
        plt.plot(x_axis, cum_rnn, 'b--', label='RNN', linewidth=1.5, alpha=0.7)
        plt.plot(x_axis, cum_aefin, 'r:', label='AEFIN', linewidth=2, alpha=0.8)
        plt.title('📊 Cumulative Returns - Short Run', fontsize=14, fontweight='bold')
        plt.xlabel('Time Steps')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Metrics Comparison Bar Chart
        plt.subplot(3, 2, 6)
        metrics_names = ['MAPE', 'SMAPE', 'RMSE', 'RMSP', 'MASE']
        
        rnn_short_metrics = [results['rnn']['Short_Run'][m] for m in metrics_names]
        aefin_short_metrics = [results['aefin']['Short_Run'][m] for m in metrics_names]
        
        x_pos = np.arange(len(metrics_names))
        width = 0.35
        
        plt.bar(x_pos - width/2, rnn_short_metrics, width, label='RNN', alpha=0.7, color='blue')
        plt.bar(x_pos + width/2, aefin_short_metrics, width, label='AEFIN', alpha=0.7, color='red')
        
        plt.title('📊 Metrics Comparison - Short Run', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.xticks(x_pos, metrics_names, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('output/plots/multi_horizon_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Comprehensive plots saved to: output/plots/multi_horizon_comprehensive_analysis.png")


def main():
    """
    Main function to run multi-horizon evaluation.
    """
    # Ensure output directories exist
    os.makedirs('output/results', exist_ok=True)
    os.makedirs('output/plots', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    
    # Create evaluator
    evaluator = MultiHorizonEvaluator(short_horizon=96, long_horizon=136)
    
    # Run comprehensive evaluation
    results, predictions_data = evaluator.run_comprehensive_evaluation()
    
    print("\n" + "=" * 70)
    print("🎉 MULTI-HORIZON EVALUATION COMPLETED!")
    print("📊 Check output/results/ for detailed metrics")
    print("📈 Check output/plots/ for comprehensive visualizations")
    print("=" * 70)


if __name__ == "__main__":
    main() 