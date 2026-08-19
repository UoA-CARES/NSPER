# NSPER: Novelty and Surprise Prioritized Experience Replay

[![arXiv](https://img.shields.io/badge/arXiv-2608.17373-b31b1b.svg)](https://arxiv.org/abs/2608.17373)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![DeepMind Control Suite](https://img.shields.io/badge/Benchmark-DeepMind%20Control%20Suite-blue)](https://github.com/google-deepmind/dm_control)

**Intrinsic-signal-based experience prioritization for sample-efficient image-based reinforcement learning.**

> **Replay experiences because they are informative — not only because they produce a large TD error.**

This repository provides the PyTorch implementation of **Novelty and Surprise Prioritized Experience Replay (NSPER)** and its intrinsic-reward extension **NSPER+R**.

NSPER combines two complementary intrinsic learning signals:

- **Novelty** — identifies observations that are poorly represented by the agent.
- **Surprise** — identifies transitions that are difficult for the agent to predict.

These signals are combined into a **Novelty–Surprise Signal (NSS)** that determines which experiences should receive greater replay priority.

In **NSPER+R**, the same signal is additionally used as an intrinsic reward, connecting **experience prioritization** and **exploration** through a shared measure of informational value.

The method is implemented within **PixelTD3** and evaluated on image-based continuous-control tasks from the **DeepMind Control Suite**.

---

## Paper


**Integrating Novelty and Surprise for Experience Prioritization and Exploration in Image-Based Reinforcement Learning**

**Hoda Yamani, Henry Williams, Bruce A. MacDonald**

*International Journal of Computer and Systems Engineering*,  
Vol. 20, No. 4, pp. 439–447, 2026.

- **Published version:** [International Journal of Computer and Systems Engineering (WASET)](https://publications.waset.org/10014461/integrating-novelty-and-surprise-for-experience-prioritization-and-exploration-in-image-based-reinforcement-learning)
- **Preprint:** [arXiv:2608.17373](https://arxiv.org/abs/2608.17373)

This work also forms part of the PhD thesis chapter:

> **Novelty and Surprise for Experience Prioritisation and Exploration**

---

## Motivation

Experience replay enables off-policy reinforcement learning agents to reuse previously collected interactions. However, the usefulness of replay depends strongly on **which experiences are selected for learning**.

Traditional Prioritized Experience Replay (PER) commonly uses the magnitude of the temporal-difference (TD) error as a proxy for transition importance. In high-dimensional continuous-control problems, however, value-based errors do not necessarily capture the broader informational value of an experience.

NSPER instead considers two complementary questions:

> **Is this observation unfamiliar?**

> **Did something happen that the agent failed to predict?**

The first is captured by **novelty**, while the second is captured by **surprise**.

Together, they identify transitions that expose gaps in the agent's current representation or predictive understanding of the environment.

---

## Method Overview

<p align="center">
  <img src="readme_media/NSPER.png" width="900" alt="NSPER architecture">
</p>

NSPER augments PixelTD3 with:

1. a convolutional **encoder–decoder** for representation learning;
2. reconstruction-based **novelty estimation**;
3. an ensemble of latent dynamics models for **surprise estimation**;
4. a **Novelty–Surprise Signal (NSS)**;
5. an NSS-driven prioritized replay buffer; and
6. optionally, NSS-based intrinsic rewards.

The learned representation therefore supports policy learning, novelty estimation, transition prediction, and experience prioritization.

---

## Novelty and Surprise

### Novelty: Representational Unfamiliarity

Given an image observation $s_t$, the encoder maps it into a latent representation:

```math
z_t = \mathrm{Enc}(s_t).
```

The decoder then reconstructs the observation:

```math
\hat{s}_t = \mathrm{Dec}(z_t).
```

Novelty is measured using the Structural Similarity Index (SSIM):

```math
\mathrm{Novelty}(s_t)
=
1 - \mathrm{SSIM}(s_t,\hat{s}_t).
```

A higher novelty value indicates that the observation is less well reconstructed and therefore less familiar to the agent.

---

### Surprise: Predictive Mismatch

While novelty captures representational unfamiliarity, surprise measures how strongly an observed transition differs from the agent's prediction.

Given the current latent representation $z_t$ and action $a_t$, an ensemble of $E$ predictive dynamics models estimates the next latent state:

```math
\tilde{z}_{t+1}^{(e)}
=
f^{(e)}(z_t,a_t),
\qquad
e=1,\ldots,E.
```

The ensemble mean prediction is:

```math
\bar{z}_{t+1}
=
\frac{1}{E}
\sum_{e=1}^{E}
\tilde{z}_{t+1}^{(e)}.
```

Surprise is defined as the prediction error between the predicted and observed next latent states:

```math
\mathrm{Surprise}(s_t,a_t)
=
\left\|
\bar{z}_{t+1}
-
z_{t+1}
\right\|_2^2.
```

A higher surprise value indicates that the transition is difficult for the current dynamics model to predict.

---

## Novelty–Surprise Signal

Novelty and surprise capture complementary aspects of experience:

| Signal | Captures | Interpretation |
|---|---|---|
| **Novelty** | Representational unfamiliarity | The observation is not yet represented well |
| **Surprise** | Predictive mismatch | The transition differs from what was expected |

NSPER combines these signals into the **Novelty–Surprise Signal (NSS)**:

```math
\mathrm{NSS}(s_t,a_t)
=
\lambda_N\,\mathrm{Novelty}(s_t)
+
\lambda_S\,\mathrm{Surprise}(s_t,a_t).
```

For the experiments reported in this work:

```math
\lambda_N = \lambda_S = 1.
```

Therefore:

```math
\mathrm{NSS}(s_t,a_t)
=
\mathrm{Novelty}(s_t)
+
\mathrm{Surprise}(s_t,a_t).
```

Novelty and surprise therefore contribute equally to the prioritization signal in the reported experiments.

---

## NSPER: Novelty–Surprise Prioritization

Standard TD-error Prioritized Experience Replay assigns transition priority according to:

```math
\sigma_i
=
|\delta_i| + \epsilon,
```

where $\delta_i$ is the TD error and $\epsilon > 0$ ensures that every transition retains a non-zero probability of being replayed.

NSPER retains the general PER framework but replaces the TD-error-based priority with the Novelty–Surprise Signal:

```math
\sigma_i
=
\mathrm{NSS}_i + \epsilon.
```

Transitions with greater novelty or surprise therefore receive higher replay priority.

This shifts experience prioritization away from purely **value-based errors** and toward the **intrinsic informational content** of an experience.

---

## NSPER+R: Prioritization and Exploration

In NSPER, NSS is used exclusively for replay prioritization:

```math
r_t
=
r_t^{\mathrm{ext}}.
```

NSPER+R additionally uses the same Novelty–Surprise Signal as an intrinsic reward:

```math
r_t
=
r_t^{\mathrm{ext}}
+
\mathrm{NSS}(s_t,a_t).
```

The two proposed variants can therefore be summarized as:

| Method | Replay Priority | Intrinsic Reward |
|---|---|---|
| **NSPER** | Novelty + Surprise | No |
| **NSPER+R** | Novelty + Surprise | Novelty + Surprise |

NSS influences learning in two complementary ways:

- **Experience selection:** Which previously collected transitions should be replayed more frequently?
- **Exploration:** Which informative transitions should the agent seek during interaction?

NSPER+R connects these two processes through the same intrinsic learning signal.

---

## Representation Learning

NSPER jointly learns representations for reconstruction and transition prediction.

The reconstruction objective is:

```math
\mathcal{L}_{\mathrm{rec}}
=
\left\|
s_t-\hat{s}_t
\right\|_2^2.
```

The predictive ensemble is trained using the dynamics prediction objective:

```math
\mathcal{L}_{\mathrm{dyn}}
=
\frac{1}{E}
\sum_{e=1}^{E}
\left\|
\tilde{z}_{t+1}^{(e)}
-
z_{t+1}
\right\|_2^2.
```

The combined representation-learning objective is:

```math
\mathcal{L}_{\mathrm{rep}}
=
\mathcal{L}_{\mathrm{rec}}
+
\mathcal{L}_{\mathrm{dyn}}.
```

This joint optimization allows the learned representation, novelty estimates, surprise estimates, and reinforcement learning policy to evolve together during training.

---

## Experimental Evaluation

NSPER and NSPER+R are evaluated on five image-based continuous-control tasks from the **DeepMind Control Suite**.

### Tasks

| Task | Domain | Main Challenge |
|---|---|---|
| **Cartpole-Balance** | Classic control | Stabilizing an underactuated system |
| **Finger-Spin** | Manipulation | Fine continuous torque control |
| **Ball-in-Cup** | Manipulation | Precise timing and coordination |
| **Walker-Walk** | Locomotion | Dynamic balance and coordination |
| **Cheetah-Run** | Locomotion | Stable high-speed control |

### Experimental Configuration

| Parameter | Value |
|---|---|
| Observation | 84 × 84 RGB images |
| Frame stacking | 3 consecutive frames |
| Training horizon | 1,000,000 environment steps |
| Replay buffer capacity | 1,000,000 transitions |
| Batch size | 128 |
| Actor learning rate | 1 × 10⁻⁴ |
| Critic learning rate | 1 × 10⁻³ |
| Prioritization exponent α | 0.7 |
| Importance-sampling exponent β | 0.4 |
| Evaluation frequency | Every 10,000 steps |
| Evaluation episodes | 10 |
| Independent random seeds | 5 |

---

## Baselines

NSPER and NSPER+R are compared with several experience replay and intrinsic-reward strategies under a common PixelTD3 framework:

- **Uniform** — uniform experience replay
- **Uniform+R (NaSATD3)** — uniform replay with intrinsic rewards
- **TD-PER** — TD-error-based prioritized experience replay
- **TD-PER+R** — TD-PER combined with intrinsic rewards
- **RPE-PER** — reward-prediction-error prioritized experience replay
- **RPE-PER+R** — RPE-PER combined with intrinsic rewards
- **CCLF** — contrastive curiosity-driven learning

All methods use the same implementation and training pipeline to provide a controlled comparison.

---

## Main Findings

The experimental results demonstrate that intrinsic learning signals can provide useful criteria not only for exploration, but also for **experience selection**.

### Overall Performance

- **NSPER improves PixelTD3 relative to the evaluated baseline methods on four of the five tasks.**
- On the remaining task, NSPER achieves performance comparable to the strongest competing approach.
- **NSPER+R achieves the strongest overall performance among the evaluated methods.**
- NSPER maintains more consistent performance across diverse tasks than relying exclusively on TD-error prioritization.

### Ablation Study

The ablation study compares:

- **NoveltyPER**
- **SurprisePER**
- **NSPER**

with and without intrinsic rewards.

The results show that:

- novelty and surprise provide **complementary information**;
- NoveltyPER generally outperforms SurprisePER;
- combining novelty and surprise generally performs better than using either signal individually; and
- combining both signals for prioritization and intrinsic reward provides the strongest overall performance.

These results support the central idea behind NSPER:

> **Informative experiences can be identified through what the agent does not yet represent well and what it cannot yet predict well.**

---

## Repository Structure

```text
NSPER/
├── networks/                  # Neural-network components
├── utils/                     # Replay and utility components
├── readme_media/              # README figures
│
├── nsper_td3.py               # NSPER implementation
├── nsper_td3_tderr.py         # TD-error comparison implementation
│
├── train_loop.py              # Main NSPER / NSPER+R experiments
├── train_loop_novelty.py      # Novelty-only ablation
├── train_loop_surprise.py     # Surprise-only ablation
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/UoA-CARES/NSPER.git
cd NSPER
```

Create an isolated Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The implementation uses **PyTorch**, **MuJoCo**, and the **DeepMind Control Suite** for image-based continuous-control experiments.

---

## Reproducing the Experiments

The repository contains separate training scripts for the main method and the ablation experiments:

```text
train_loop.py
train_loop_novelty.py
train_loop_surprise.py
```

- `train_loop.py` — NSPER and NSPER+R
- `train_loop_novelty.py` — novelty-only prioritization
- `train_loop_surprise.py` — surprise-only prioritization

The experimental configuration used in the paper is summarized above.

> **Note:** See the training scripts for the currently supported command-line arguments and experiment configuration.

---

## Research Context

NSPER was developed as part of research on **sample-efficient reinforcement learning**, with a particular focus on improving how agents select and reuse informative experiences.

The work forms part of the doctoral research chapter:

> **Novelty and Surprise for Experience Prioritisation and Exploration**

and accompanies the paper:

> **Integrating Novelty and Surprise for Experience Prioritization and Exploration in Image-Based Reinforcement Learning**

The broader motivation is to move experience replay beyond purely value-based criteria toward signals that capture the **informational content of interaction**.

---

## Paper

**Integrating Novelty and Surprise for Experience Prioritization and Exploration in Image-Based Reinforcement Learning**

Hoda Yamani, Henry Williams, and Bruce A. MacDonald

*International Journal of Computer and Systems Engineering*,  
Vol. 20, No. 4, pp. 439–447, 2026.

- **Published version:** World Academy of Science, Engineering and Technology
- **Preprint:** arXiv:2608.17373

---

## Authors

**Hoda Yamani**  
**Henry Williams**  
**Bruce A. MacDonald**

Robot Learning Team and CARES Robotics Lab  
Department of Electrical, Computer, and Software Engineering  
University of Auckland  
Auckland, New Zealand

---

## Acknowledgements

This research was conducted within the **Robot Learning Team** and **CARES Robotics Lab** at the University of Auckland.
