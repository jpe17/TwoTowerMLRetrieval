# WandB Sweep Setup for Two-Tower ML Retrieval

This directory contains the WandB sweep configuration for hyperparameter tuning of the two-tower retrieval model.

## Files

- `sweep_config.yaml` - Hyperparameter sweep configuration
- `sweep_train.py` - Training function called by sweep agents
- `run_sweep.py` - Main script to start the sweep
- `test_sweep.py` - Test script to verify setup
- `README.md` - This file

## Quick Start

1. **Test the setup** (recommended):
   ```bash
   cd backend/sweep
   python test_sweep.py
   ```

2. **Start a sweep**:
   ```bash
   cd backend/sweep  # Important: run from sweep directory
   python run_sweep.py
   ```

3. **Or run from project root**:
   ```bash
   python backend/sweep/run_sweep.py
   ```

## Configuration

The sweep is configured in `sweep_config.yaml` with the following hyperparameters:

- `LR`: Learning rates [0.0001, 0.0005, 0.001, 0.002]
- `BATCH_SIZE`: Batch sizes [32, 64, 128]
- `HIDDEN_DIM`: Hidden dimensions [32, 64, 128, 256]
- `MARGIN`: Triplet loss margins [0.2, 0.5, 1.0]
- `RNN_TYPE`: RNN types ["GRU", "LSTM"]
- `NUM_LAYERS`: Number of layers [1, 2, 3]
- `DROPOUT`: Dropout rates [0.0, 0.1, 0.2, 0.3]
- `SUBSAMPLE_RATIO`: Data subsampling [0.01, 0.02, 0.05]
- `SHARED_ENCODER`: Whether to share encoders [false, true]

## Environment Setup

Make sure you have:

1. A `.env` file with `WANDB_API_KEY=your_key_here`
2. All required dependencies installed
3. Data files in the correct locations (as specified in `backend/config.json`)

## Monitoring

Once started, you can monitor your sweep at:
```
https://wandb.ai/your-username/two-tower-ml-retrieval/sweeps/SWEEP_ID
```

The sweep will optimize for the lowest validation loss by default.

## Troubleshooting

If you encounter import errors:
1. Run `python test_sweep.py` to diagnose issues
2. Check that all required files exist in the `backend/` directory
3. Verify your Python path includes the project root

## Customization

To modify the sweep:
1. Edit `sweep_config.yaml` to change hyperparameters
2. Edit `sweep_train.py` to modify the training loop
3. Update the metric in `sweep_config.yaml` if needed

## Early Termination

The sweep uses Hyperband early termination to stop poorly performing runs:
- `min_iter`: 1 epoch minimum
- `max_iter`: 5 epochs maximum

This helps save compute resources by stopping bad configurations early. 