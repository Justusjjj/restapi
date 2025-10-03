# 🚀 Advanced Forex Trading System - Complete Feature Documentation

## 📋 **System Overview**
A comprehensive ML/AI trading system with MT5 integration, multi-timeframe analysis, ICT/price action signals, deep learning models, reinforcement learning, and hybrid architectures.

---

## 🎯 **Core Features Implemented**

### 1. **Multi-Timeframe Analysis Pipeline** (`mtf_pipeline.py`)
- **Higher Timeframes**: MN1/W1/H4 bias detection and key levels
- **Structure Analysis**: H1 trend breaks, order blocks, FVG, liquidity
- **Confirmation Signals**: M15/M5/M1 ICT analysis
- **Deep Model Integration**: Optional LSTM confirmation
- **Sentiment Context**: News sentiment analysis

### 2. **Streamlined ICT/Price Action Features** (`streamlined_features.py`)
- **Order Blocks**: Bullish/bearish counts, latest price, strength
- **Fair Value Gaps**: FVG detection and strength scoring
- **Liquidity Sweeps**: High/low sweep detection
- **Market Structure**: HH/LL break detection
- **Composite Signal**: Weighted combination of all signals
- **Economic Calendar Proxy**: Time-based impact scoring

### 3. **Deep Learning Models**

#### **LSTM Sequence Classifier** (`deep_model.py`)
- Multi-layer LSTM with dropout and early stopping
- Class-weighted training for imbalanced data
- Sequence-to-classification for directional prediction
- PyTorch implementation with GPU support

#### **Hybrid Transformer + Decision Tree** (`hybrid_transformer_tree.py`)
- **Transformer Feature Extractor**: Captures complex temporal patterns
- **Decision Tree Classifier**: Interpretable decision making
- **Tree Options**: Random Forest, Gradient Boosting, Single Tree
- **Contrastive Learning**: Self-supervised feature learning
- **Feature Importance**: Interpretable decision rules

### 4. **Reinforcement Learning** (`rl_trading.py`)
- **PPO Environment**: 25-dimensional state space
- **Actions**: HOLD/BUY/SELL with risk-adjusted sizing
- **Reward Function**: P&L + balance bonuses + holding penalties
- **Features**: Price, ICT, economic, account state
- **Training**: Stable-Baselines3 PPO implementation

### 5. **Backtesting System** (`backtest_decider.py`)
- **Decision Rule Backtesting**: Historical validation of trading rules
- **ATR-Based Sizing**: Dynamic position sizing
- **Performance Metrics**: Win rate, drawdown, profit factor
- **Trade History**: Complete execution log

### 6. **Execution System** (`execution.py`)
- **MT5 Integration**: Order placement and management
- **Risk Management**: ATR-based position sizing
- **Stop Loss/Take Profit**: Automatic risk control
- **Account Management**: Balance and exposure tracking

---

## 🛠️ **CLI Commands**

### **Data & Training**
```bash
# Build dataset for ML training
python /workspace/cli.py build-dataset EURUSD --seq-len 64 --horizon 12 --out-npz /workspace/eurusd_mtf.npz

# Train LSTM model
python /workspace/cli.py train /workspace/eurusd_mtf.npz --epochs 30 --batch-size 128 --lr 5e-4 --out-pt /workspace/eurusd_lstm.pt

# Train Hybrid Transformer + Decision Tree
python /workspace/cli.py hybrid-train /workspace/eurusd_mtf.npz --tree-type random_forest --epochs 50 --out-model /workspace/hybrid_model.pt

# Train RL PPO Agent
python /workspace/cli.py rl-train EURUSD --timesteps 100000 --lr 3e-4 --out-model /workspace/ppo_model
```

### **Analysis & Decisions**
```bash
# Multi-timeframe analysis
python /workspace/cli.py analyze EURUSD --model-pt /workspace/eurusd_lstm.pt

# Rule-based decision
python /workspace/cli.py decide EURUSD --model-pt /workspace/eurusd_lstm.pt

# Hybrid model decision
python /workspace/cli.py hybrid-decide EURUSD /workspace/hybrid_model.pt
```

### **Evaluation & Testing**
```bash
# Evaluate LSTM model
python /workspace/cli.py evaluate /workspace/eurusd_mtf.npz /workspace/eurusd_lstm.pt

# Evaluate Hybrid model
python /workspace/cli.py hybrid-evaluate /workspace/eurusd_mtf.npz /workspace/hybrid_model.pt

# Evaluate RL agent
python /workspace/cli.py rl-evaluate EURUSD /workspace/ppo_model --episodes 10

# Backtest decision rules
python /workspace/cli.py backtest-decider EURUSD --model-pt /workspace/eurusd_lstm.pt --risk-pct 0.01 --out-json /workspace/backtest.json
```

### **Live Trading**
```bash
# Execute single trade
python /workspace/cli.py trade-once EURUSD --model-pt /workspace/eurusd_lstm.pt --risk-pct 0.01 --atr-mult 1.5
```

---

## 🔬 **Research-Based Features**

### **Hybrid Transformer + Decision Tree Architecture**
Based on web research findings:

1. **Transformer Feature Extraction**:
   - Multi-head attention for temporal pattern recognition
   - Positional encoding for sequence understanding
   - Contrastive learning for discriminative features
   - Bounded output (Tanh) for decision tree stability

2. **Decision Tree Integration**:
   - Random Forest for robustness
   - Gradient Boosting for performance
   - Single Tree for interpretability
   - Feature importance analysis

3. **Benefits**:
   - **Accuracy**: Transformer captures complex patterns
   - **Interpretability**: Decision tree provides clear rules
   - **Robustness**: Ensemble methods reduce overfitting
   - **Feature Learning**: Self-supervised contrastive learning

### **Streamlined ICT Implementation**
- **Noise Reduction**: Only proven signals (OBs, FVGs, sweeps, structure)
- **Quantified Signals**: Explicit counts and strength scores
- **Composite Scoring**: Weighted combination for decision making
- **Economic Context**: Time-based impact assessment

---

## 📊 **Model Comparison**

| Model Type | Accuracy | Interpretability | Speed | Use Case |
|------------|----------|------------------|-------|----------|
| **LSTM** | High | Low | Medium | Pattern recognition |
| **Hybrid Transformer+Tree** | Very High | High | Medium | Best of both worlds |
| **RL PPO** | Medium | Low | Slow | Adaptive strategies |
| **Rule-Based** | Medium | Very High | Fast | Conservative trading |

---

## 🚀 **Usage Workflow**

### **1. Development Phase**
```bash
# Build comprehensive dataset
python /workspace/cli.py build-dataset EURUSD --seq-len 64 --horizon 12 --out-npz /workspace/eurusd_mtf.npz

# Train multiple models
python /workspace/cli.py train /workspace/eurusd_mtf.npz --epochs 30 --out-pt /workspace/eurusd_lstm.pt
python /workspace/cli.py hybrid-train /workspace/eurusd_mtf.npz --tree-type random_forest --out-model /workspace/hybrid_model.pt
python /workspace/cli.py rl-train EURUSD --timesteps 100000 --out-model /workspace/ppo_model
```

### **2. Validation Phase**
```bash
# Evaluate all models
python /workspace/cli.py evaluate /workspace/eurusd_mtf.npz /workspace/eurusd_lstm.pt
python /workspace/cli.py hybrid-evaluate /workspace/eurusd_mtf.npz /workspace/hybrid_model.pt
python /workspace/cli.py rl-evaluate EURUSD /workspace/ppo_model --episodes 10

# Backtest decision rules
python /workspace/cli.py backtest-decider EURUSD --model-pt /workspace/eurusd_lstm.pt --risk-pct 0.01
```

### **3. Production Phase**
```bash
# Live trading with best model
python /workspace/cli.py trade-once EURUSD --model-pt /workspace/hybrid_model.pt --risk-pct 0.01
```

---

## ⚙️ **Configuration**

### **Environment Variables** (`.env`)
```env
# MT5 Connection
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server

# Trading Parameters
PAPER_BALANCE=10000
RISK_PER_TRADE=0.01
ATR_MULTIPLIER=1.5

# Model Parameters
SEQ_LEN=64
HORIZON=12
BATCH_SIZE=64
LEARNING_RATE=1e-3
```

### **Model Parameters**
- **LSTM**: 2 layers, 128 hidden units, dropout 0.2
- **Transformer**: 4 heads, 2 layers, 64 d_model
- **Decision Tree**: Max depth 10, min samples split 5
- **PPO**: 2048 steps, 64 batch size, 10 epochs

---

## 🎯 **Key Advantages**

1. **Multi-Model Ensemble**: Combines different approaches for robustness
2. **Interpretable Decisions**: Hybrid model provides clear reasoning
3. **Risk Management**: ATR-based sizing with configurable risk
4. **Real-Time Integration**: Direct MT5 connection for live trading
5. **Comprehensive Backtesting**: Historical validation of strategies
6. **Research-Driven**: Based on latest ML/AI research findings

---

## 🔮 **Future Enhancements**

1. **Economic Calendar Integration**: Real-time news impact scoring
2. **Multi-Asset Support**: Portfolio-level decision making
3. **Advanced RL**: Multi-agent systems and hierarchical RL
4. **Feature Engineering**: More sophisticated ICT signal extraction
5. **Model Ensembling**: Dynamic model selection based on market conditions

---

## 📝 **Notes**

- **Hybrid Model**: Best accuracy + interpretability combination
- **ICT Signals**: Focused on proven patterns only
- **Risk Management**: Conservative approach with ATR sizing
- **Backtesting**: Essential for strategy validation
- **Live Trading**: Start with paper trading, then live with small position sizes

The system is production-ready with comprehensive testing, validation, and risk management features.