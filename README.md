# Deep Reinforcement Learning for MIMO Communication with Low-Resolution ADCs

This repository contains the official codebase for our paper:

**Deep Reinforcement Learning for MIMO Communication with Low-Resolution ADCs**

📄 **[Link to arxiv pre-print]([https://arxiv.org/abs/your-paper-id](https://arxiv.org/abs/2504.18957)**

**Abstract:**

Multiple-input multiple-output (MIMO) wireless systems conventionally use high-resolution analog-to-digital converters (ADCs) at the receiver side to faithfully digitize received signals prior to digital signal processing. However, the power consumption of high-resolution ADCs increases significantly as the bandwidth is increased, particularly in millimeter wave communications systems. A combination of two mitigating approaches has been considered in the literature: i) to use hybrid beamforming to reduce the number of ADCs, and ii) to use low-resolution ADCs to reduce per ADC power consumption.
Lowering the number and resolution of the ADCs naturally reduces the communication rate of the system, leading to a tradeoff between ADC power consumption and communication rate. Prior works have shown that optimizing over the hybrid beamforming matrix and ADC thresholds may reduce the aforementioned rate-loss significantly. A key challenge is the complexity of optimization over all choices of beamforming matrices and threshold vectors. This work proposes a reinforcement learning (RL) architecture to perform the optimization. The proposed approach integrates deep neural network-based mutual information estimators for reward calculation with policy gradient methods for reinforcement learning. The approach is robust to dynamic channel statistics and noisy CSI estimates. It is shown theoretically that greedy RL methods converge to the globally optimal policy. Extensive empirical evaluations are provided demonstrating that the performance of the proposed RL-based approach closely matches exhaustive search optimization across the solution space.

![Reinforcement Learning Framework Overview](figures/rl_overview.png)

---
## 📁 Repository Structure

```plaintext
RL_MIMO/
├── experiments_jupyter_notebooks/      # Jupyter notebooks used for experiments
│   ├── plotting_simulations.ipynb
│   ├── simulation-runner.ipynb
│   ├── training_cortical.ipynb
│   └── training_policies.ipynb
├── experiments_py/                    # Scripts for reproducible training, inference, and plotting
│   ├── train_policy.py
│   ├── train_cortical.py
│   ├── run_simulations.py
│   └── plot_results.py
├── figures/                           # PDF/PNG figures for paper and README
├── simulation_results/                # Saved output from simulations
│   ├── 1D/                            
│   ├── 2D/                            
│   └── H/                             
├── models/                            # Trained policies and CORTICAL estimators
│   ├── policy_models/
│   └── cortical_models/
├── src/                               # Core source code
│   ├── ba_estimator.py                # Blahut-Arimoto mutual information estimator
│   ├── channel.py                     # Channel model 
│   ├── cortical_estimator.py          # CORTICAL MI estimator
│   ├── rl_environment.py              # Environment setup for RL training
│   ├── rl_policy.py                   # Policy and unified policy definitions
│   ├── simulation_runner.py           # Simulation functions
│   ├── utils.py                       # Helper functions 
├── functions.py                       # Legacy loader shim for model compatibility
├── requirements.txt                   # Python dependencies
└── README.md                          # Project overview and documentation

```

---

## 🚀 Quickstart

1. **Clone the repository**

    ```bash
    git clone https://github.com/mtemp009-2001/RL_MIMO.git
    cd RL_MIMO
    ```

2. **Install dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Train a policy**

You can either use the notebook:

- `experiments_jupyter_notebooks/training_policies.ipynb`

Or run from terminal using:

```bash
python experiments_py/train_policy.py \
    --dimension 2 \
    --num_thresholds 4 \
    --mi_estimator BA \
    --run_id run10
```

**Key arguments**:
- `--dimension`: Input space dimension (1 or 2)
- `--num_thresholds`: Number of quantization thresholds
- `--mi_estimator`: Mutual information estimator to use (`BA` or `CORTICAL`)
- `--run_id`: Identifier for saving the policy


4. **Train a CORTICAL mutual information estimator**

Via notebook:

- `experiments_jupyter_notebooks/training_cortical.ipynb`

Or from terminal:

```bash
python experiments_py/train_cortical.py \
    --dimension 1 \
    --num_thresholds 1 \
    --epochs 100 \
    --subepochs 10 \
    --verbose 1
```

**Key arguments**:
- `--sample_size`, `--num_batches`: Size of training batches and count
- `--epochs`, `--subepochs`: Number of epochs for overall model and subepochs for Discriminator
- `--dimension`, `--num_thresholds`: Match the downstream use case


5. **Run inference simulations using a trained policy**

From notebook:

- `experiments_jupyter_notebooks/simulation-runner.ipynb`

Or from terminal:

```bash
python experiments_py/run_simulations.py \
    --dimension 2 \
    --num_thresholds 4 \
    --mi_estimator BA \
    --channel_type identity-csi \
    --run_id run10 \
    --num_sims 10 \
    --box_param 1.5
```

**Key arguments**:
- `--channel_type`: One of `identity-csi`, `noisy-csi-0.01`, `noisy-csi-0.05`, `noisy-csi-0.1`, `changing-csi-smooth-0.01`, or `changing-csi-smooth-0.05`.
- `--num_sims`: Number of simulations to run
- `--box_param`: Bounding box for determining centroids and bounding regions

6. **Plot mutual information vs SNR curves**

From notebook:

- `experiments_jupyter_notebooks/plotting_simulations.ipynb`

Or from terminal:

```bash
python experiments_py/plot_results.py \
    --dimensions=2 \
    --num_thresholds_list=4 \
    --mi_estimators=BA \
    --channel_types=identity-csi \
    --run_ids=run10 \
    --sim_counts=0,1,2,3,4,5,6,7,8,9
```

**Key arguments**:
- `--dimensions`: One or more input dimensions (comma-separated)
- `--num_thresholds_list`: Threshold configs to compare
- `--mi_estimators`: Estimators to plot (`BA`, `CORTICAL`, can also include `BruteForce`, `PSK`, or `QUAM` for comparison)
- `--run_ids`: Identifiers used during training
- `--channel_types`: Channels to include
- `--sim_counts`: Which simulation files to aggregate
- `--num_std`: Width of shaded confidence band (in std dev)

By default, the script will open the plot in a window. Note that BruteForce, PSK, and QUAM all use the BA estimator (but without using the RL policies).


7. **Legacy compatibility**

If loading archived models used in the paper, ensure the file `functions.py` exists. This file is included for backward compatibility and should not be used in new development.

---
