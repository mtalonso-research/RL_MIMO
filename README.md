# Deep Reinforcement Learning for MIMO Communication with Low-Resolution ADCs

This repository contains the official codebase for our paper:

**Deep Reinforcement Learning for MIMO Communication with Low-Resolution ADCs**

📄 **[Link to paper / preprint](https://arxiv.org/abs/your-paper-id)**

Multiple-input multiple-output (MIMO) wireless systems conventionally use high-resolution analog-to-digital converters (ADCs) at the receiver side to faithfully digitize received signals prior to digital signal processing. However, the power consumption of high-resolution ADCs increases significantly as the bandwidth is increased, particularly in millimeter wave communications systems. A combination of two mitigating approaches has been considered in the literature: i) to use hybrid beamforming to reduce the number of ADCs, and ii) to use low-resolution ADCs to reduce per ADC power consumption.
Lowering the number and resolution of the ADCs naturally reduces the communication rate of the system, leading to a tradeoff between ADC power consumption and communication rate. Prior works have shown that optimizing over the hybrid beamforming matrix and ADC thresholds may reduce the aforementioned rate-loss significantly. A key challenge is the complexity of optimization over all choices of beamforming matrices and threshold vectors. This work proposes a reinforcement learning (RL) architecture to perform the optimization. The proposed approach integrates deep neural network-based mutual information estimators for reward calculation with policy gradient methods for reinforcement learning. The approach is robust to dynamic channel statistics and noisy CSI estimates. It is shown theoretically that greedy RL methods converge to the globally optimal policy. Extensive empirical evaluations are provided demonstrating that the performance of the proposed RL-based approach closely matches exhaustive search optimization across the solution space.

![Reinforcement Learning Framework Overview](figures/rl_overview.png)

---

## 📁 Repository Structure


---

## 🚀 Quickstart

1. **Clone the repository**
2. **Install dependencies**
We recommend using a virtual environment.

---

## 📖 Citation

If you use this code in your research, please cite:


---

## 🪪 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙋 Contact

For questions or collaborations, feel free to reach out via your-email@example.com or open an issue.
