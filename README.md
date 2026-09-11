# PaperNet-AI: End-to-End Research Topic Classification Using GNNs

PaperNet-AI is an end-to-end Deep Learning pipeline built with PyTorch Geometric (PyG) and ONNX for classifying scientific papers into research topics using Graph Attention Networks v2 (GATv2) with Jumping Knowledge.

---

## 📁 Modular Project Structure

```text
PaperNet-AI/
├── config/
│   └── config.yaml                     # Centralized hyperparameter & path configurations
├── utils/
│   ├── __init__.py                     # Package exposures
│   ├── logger.py                       # Project-wide logging (console & timestamped log files)
│   ├── custom_exception.py             # Detailed custom exception with traceback line reporting
│   └── helpers.py                      # Utilities (YAML reader, JSON save/load, seeds, device resolver)
├── src/
│   ├── __init__.py
│   ├── constants/
│   │   └── __init__.py                 # Global constants (e.g., CONFIG_FILE_PATH)
│   ├── entity/
│   │   ├── __init__.py
│   │   └── config_entity.py            # Dataclass configuration entities
│   ├── config/
│   │   ├── __init__.py
│   │   └── configuration.py            # ConfigurationManager reading config.yaml
│   ├── models/
│   │   ├── __init__.py
│   │   └── gat.py                      # GATv2Block & AdvancedCoraGAT model architecture
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py           # Dataset downloading, loading, profiling & visualization
│   │   ├── model_trainer.py            # Training loop with edge dropout, grad clipping, early stopping
│   │   └── model_evaluation.py         # Test split evaluation, confusion matrix, curves, ONNX export
│   └── pipeline/
│       ├── __init__.py
│       ├── stage_01_data_ingestion.py  # Stage 1 execution
│       ├── stage_02_model_trainer.py   # Stage 2 execution
│       ├── stage_03_model_evaluation.py# Stage 3 execution
│       ├── training_pipeline.py        # End-to-end multi-stage training pipeline orchestrator
│       └── prediction_pipeline.py      # Inference pipeline supporting PyTorch (.pt) and ONNX Runtime
├── artifacts/
│   ├── data/                           # Cora dataset files
│   ├── models/                         # PyTorch (.pt) checkpoint and ONNX (.onnx) model
│   └── evaluation/                     # Metrics, classification report, plots
├── logs/                               # Execution runtime logs
├── notebooks/                          # Original exploratory notebooks
├── main.py                             # Pipeline runner entrypoint
├── predict.py                          # Sample inference script
├── requirements.txt                    # Project dependencies
└── setup.py                            # Package installer
```

---

## ⚙️ Configuration (`config/config.yaml`)

All parameters are decoupled from code into `config/config.yaml`:
- **Data Ingestion**: Dataset root path and dataset name (`cora`).
- **Model Architecture**: Number of GAT layers, hidden dimensions, attention heads, dropout rates, Jumping Knowledge mode (`cat`).
- **Model Trainer**: Epochs, learning rate, weight decay, edge dropout rate, label smoothing, gradient clipping, early stopping patience, LR scheduler parameters.
- **Model Evaluation**: Metrics output path, classification report path, confusion matrix plot path, training curves plot path, ONNX export parameters and opset version.

---

## 🚀 How to Run

### 1. Install Environment & Package
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Run End-to-End Training & Evaluation Pipeline
```bash
python main.py
```
This automatically runs:
- **Stage 01**: Data Ingestion and graph profiling.
- **Stage 02**: GATv2 model training with early stopping and best checkpoint persistence.
- **Stage 03**: Model evaluation on test split, metric saving (`metrics.json`), classification report (`classification_report.txt`), confusion matrix plot, training curves plot, and ONNX model export with dynamic axes.

### 3. Run Inference / Predictions
```bash
python predict.py
```
Performs sample inference on test nodes using both the PyTorch checkpoint (`.pt`) and high-performance ONNX Runtime session (`.onnx`).
