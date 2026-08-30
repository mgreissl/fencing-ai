# Using Deep Learning to Referee Fencing

An automated computer vision system for referee decision prediction in Olympic fencing bouts from single-camera broadcast video. The model classifies weapon exchanges into referee outcomes: **Left (`L`)**, **Right (`R`)**, or **Simultaneous (`T`)** according to FIE rules.

![Fencing AI Exchange Example](resources/example_clip.gif)

---

## 1. Background: Priority (Right of Way) in Foil
Unlike Épée—where points are determined solely by electrical contact timing and double-touches award points to both competitors—Foil (and Sabre) have conventional rules of Right of Way (Priorité) [3]:

* **Initiation & The Attack:** The fencer who first begins an offensive movement (characterized by arm extension continuously threatening the opponent's valid torso target while advancing) possesses priority.
* **The Parry-Riposte:** A defender cannot simply hit into an attack. To gain priority, the defender must first deflect the incoming blade (Parry). A successful parry extinguishes the opponent's attack and confers priority to the defender's return touch (Riposte).
* **Point-in-Line:** A defensive fencer who establishes an extended arm and blade threatening the target before the opponent initiates an attack holds priority until their blade is deflected or moved.
* **Simultaneous Actions (Simultané):** When both fencers initiate offensive actions concurrently without prior blade contact, neither holds priority. Even if both colored scoring lights illuminate, no point is awarded (**Tie** / **`T`**), and action resumes from the En Garde lines.

The Computer Vision Challenge: An automated referee cannot rely solely on the apparatus lights. It must analyze the footwork acceleration, blade feints, parry contacts, and arm extension timings to determine priority.

---

## 2. System Architecture & Methodology

This project expands upon work by Sholto Douglas in [SholtoD/fencing-AI](https://github.com/SholtoD/fencing-AI). 

- **Spatio-Temporal Representation**: Leverages a pre-trained **VideoMAE** [1] (3D Vision Transformer) operating on spacetime tubelet patches ($16 \times 16 \times 2$).
- **Parameter-Efficient Fine-Tuning**: Rank-8 LoRA adapters applied to query/value projections across layers 8–11 update only **0.58%** of model weights (~501K parameters) [2].
- **Bilateral Spatial-Difference Pooling**: Resolves left-right orientation symmetry collapse by extracting global scene context combined with lateral difference features:
  $$\mathbf{z} = \left[ \mathbf{z}_{\text{global}}, \; \mathbf{z}_{\text{left}} - \mathbf{z}_{\text{right}} \right] \in \mathbb{R}^{1536}$$
  This feature representation is mathematically anti-symmetric under horizontal reflections ($\mathbf{z}_{\text{diff}} \to -\mathbf{z}_{\text{diff}}$), enforcing robust separation between Left and Right touches.
- **End-Weighted Temporal Sampling**: Allocates 25% of sampled frames to approach footwork and 75% densely over the terminal blade clash (last 400–600 ms).

---

## 3. Experimental Evaluation

| Model Architecture | Adaptation Method | Test Accuracy | Peak Val Accuracy |
| :--- | :--- | :---: | :---: |
| **Fencing AI (2017)** | InceptionV3 + Rainbow Flow + LSTM | ~60.0% *(w/ leakage)* | — |
| **VideoMAE** | Frozen Backbone + MLP Head | 57.66% | 60.57% |
| **VideoMAE + LoRA** | Bilateral Spatial Pooling + LoRA (8–11) | **63.31%** | **72.76%** |

---

## 4. Dataset Summary

Curated from 1,490 international tournament broadcasts (World Cups, Grand Prix, Olympic bouts):

| Weapon | Matches | Action Clips | Left (`L`) | Tie (`T`) | Right (`R`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Foil** | 409 | 1,591 | 558 (35.1%) | 458 (28.8%) | 575 (36.1%) |
| **Sabre** | 644 | 346 | 93 (26.9%) | 144 (41.6%) | 109 (31.5%) |
| **Epee** | 437 | 124 | 58 (46.8%) | 16 (12.9%) | 50 (40.3%) |
| **Total** | **1,490** | **2,061** | **709 (34.4%)** | **618 (30.0%)** | **734 (35.6%)** |

---

## 5. Reproduction & Execution

### Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Video Ingestion & Action Extraction
```bash

python 2-cut_and_label.py --weapon foil
```

### Model Training & Evaluation
```bash
# Train Foil model
python 6-train_AI.py data.weapon=foil training.epochs=20 training.batch_size=4

# Train multi-weapon model
python 6-train_AI.py data.weapon=multi training.epochs=25 training.batch_size=4

# Evaluate checkpoint against held-out test distribution
python 7-evaluate_AI.py data.weapon=foil
```

---

## 6. References

[1] Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.* NeurIPS 2022. [[arXiv:2203.12602]](https://arxiv.org/abs/2203.12602)

[2] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [[arXiv:2106.09685]](https://arxiv.org/abs/2106.09685)

[3] Fédération Internationale d'Escrime. *Technical Rules.* [[fie.org]](https://fie.org/fie/documents/rules)
