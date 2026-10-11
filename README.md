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
  This achieves approximate flip-relatedness through left-right difference features. Exact equivariance requires test-time averaging: average $p(L|x)$ with $p(R|\text{flip}(x))$ (and vice versa), either at inference or as a consistency loss during training. Note: ViT token features are not flip-equivariant, and fencers do not stay on fixed screen halves due to camera pans and center-line crossings.
- **Temporal Sampling**: The default `onset_window` mode samples frames evenly in a window around the light onset (`window_pre_s` before, `window_post_s` after). `end_weighted` (25% of frames early, 75% in the second half of the clip) and `uniform` remain available via `data.sampling_mode`.

---

## 3. Experimental Evaluation

> **Stale numbers.** The results and dataset counts below were measured with the old clip-level random split, where clips of the same match could land in train and test (leakage, #2), without confidence intervals, and before label, sampling and overlay-mask fixes. They are not comparable with the current pipeline and must be re-measured with the match-level split (`data.group_by=match` or grouped CV), then reported with bootstrap intervals (see "Statistical Evaluation").

| Model Architecture | Adaptation Method | Test Accuracy | Peak Val Accuracy |
| :--- | :--- | :---: | :---: |
| **Fencing AI (2017)** | InceptionV3 + Rainbow Flow + LSTM | ~60.0% *(w/ leakage)* | — |
| **VideoMAE** | Frozen Backbone + MLP Head | 58.85% | 60.50% |
| **VideoMAE + LoRA (Foil)** | Bilateral Spatial Pooling + LoRA (8–11) | **63.79%** | **72.76%** |
| **VideoMAE + LoRA (Multi-Weapon)** | Bilateral Pooling + LoRA + Weapon Embed | 51.62% | 58.31% |

---

## 4. Dataset Summary

*Counts predate the manifest pipeline and the stricter label checks; regenerate them from `manifest.csv`.*

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

### Command Line Interface

All steps run through one CLI, `python -m src <command>` (or `fencing-ai <command>` after `pip install -e .`). Arguments after the command are passed to the underlying module unchanged; `python -m src <command> --help` lists them.

| Command | Runs | Purpose |
| :--- | :--- | :--- |
| `harvest` | `scripts/harvest_fencingvision.py` | collect and verify match URLs into `{weapon}_videos.txt` |
| `preprocess` | `scripts/preprocess_pipeline.py` | download each match, cut clips at the light onset, label them, write the manifest |
| `precompute-kinematics` | `scripts/precompute_kinematics.py` | cache pose-based kinematic features per clip |
| `train` | `src/train.py` (Hydra) | train; `key=value` overrides |
| `evaluate` | `src/evaluate.py` (Hydra) | evaluate a checkpoint on its saved split |
| `bootstrap` | `scripts/bootstrap_eval.py` | match-level bootstrap CIs, paired McNemar test |
| `ablation` | `scripts/run_ablation.py` | run the ablation grid (reports "not available" until the script exists) |
| `audit-sample` / `audit-report` | `scripts/audit_sample.py` / `audit_report.py` | manual label audit |

The numbered scripts `2-cut_and_label.py`, `6-train_AI.py` and `7-evaluate_AI.py` still work as thin wrappers that print a deprecation note. `0-find_vids.py` and `1-download_vids.py` were replaced by `harvest` and `preprocess` and moved to `legacy/` (see `legacy/README.md`).

### Video Ingestion & Action Extraction
```bash
python -m src harvest --weapon foil --workers 3
python -m src preprocess --weapon foil [--max-matches 20] [--pre-s 1.0] [--post-s 0.25] [--clip-fps 25]
```

`preprocess` downloads each match itself (360p, no cookies unless `--cookies-browser` or `--cookies-file` is given; `--remote-components` lets yt-dlp fetch its JS challenge solver), saves clips at the source fps unless `--clip-fps` is given, and deletes the source video afterwards. Rejected exchanges are logged with a reason to `state/rejections_{weapon}.csv`, progress to `state/processed_{weapon}_urls.txt`.

### Clip Manifest

Every clip is described in `data/clips/{weapon}/manifest.csv` (one row per clip) instead of being encoded in its file name. Clip files are named after the video id.

| Column | Meaning |
| :--- | :--- |
| `clip_path`, `label`, `weapon`, `match_id` | required: clip file, `L`/`T`/`R`, `foil`/`sabre`/`epee`, source match |
| `source_url`, `tournament` | provenance; `tournament` enables `data.group_by=tournament` |
| `hit_type` | light pattern: `On-On`, `On-No`, `No-On`, `On-Off`, `Off-On` |
| `score_{left,right}_{before,after}` | scoreboard reads around the decision |
| `light_time_s`, `light_frame_in_clip` | light onset in the source video and in the clip (used by `onset_window` sampling) |
| `source_fps`, `clip_fps`, `n_frames` | timing of the saved clip |
| `split` | optional; used when `data.use_manifest_split=true` and every row has one |

If no manifest exists, clips are discovered by scanning `*.mp4` and parsing the legacy file name (no onset information, so sampling falls back to `uniform`).

### Kinematics Cache

The kinematic stream reads cached pose features. Fill the cache once before training (cache entries are keyed by clip and kinematics settings):

```bash
python -m src precompute-kinematics --clips_dir data/clips/foil --cache_dir data/kinematics
```

With `model.kinematics.compute_if_missing=false` (default) a missing cache entry raises an error; set it to `true` to compute pose features on the fly.

### Model Training & Evaluation
```bash
# Train Foil model
python -m src train data.weapon=foil training.epochs=20 training.batch_size=4

# Train multi-weapon model (epee excluded by default, see data.exclude_weapons)
python -m src train data.weapon=multi training.epochs=25 training.batch_size=4

# Ablations: video only, kinematics only, both (configs/experiment/)
python -m src train +experiment=video_only

# Evaluate checkpoints/best_model.pt (training.checkpoint_dir) on its saved test split
python -m src evaluate data.weapon=foil
```

Training saves the split assignment next to the checkpoint as `split_manifest.csv` (in `training.checkpoint_dir`). **Evaluation requires this file** and fails if it is missing: the test split is never recomputed at evaluation time. `eval.split` selects which split to evaluate (`test` by default). Evaluation writes `predictions_<split>.csv` and `metrics_<split>.json` to `eval.output_dir` (default: the checkpoint directory).

### Splits and Configuration

Defaults live in `configs/default.yaml`; every key can be overridden on the command line (`python -m src train key=value`).

| Key | Default | Effect |
| :--- | :--- | :--- |
| `data.group_by` | `match` | clips of one match (or `tournament`) never span splits; splits are class-stratified at group level |
| `data.split_seed`, `train_ratio`, `val_ratio`, `test_ratio` | 42, 0.7, 0.15, 0.15 | split reproducibility and sizes |
| `data.use_manifest_split` | `false` | take the `split` column of the manifest if every row has one |
| `data.cv_folds`, `data.cv_fold` | 0, 0 | grouped k-fold CV; with `cv_folds>0` fold `cv_fold` is test, the next fold val, the rest train |
| `data.exclude_weapons` | `[epee]` | dropped when `data.weapon=multi` (epee has no right of way, #22) |
| `data.sampling_mode` | `onset_window` | `onset_window` samples evenly in `[onset - window_pre_s, onset + window_post_s]`; `end_weighted` is denser in the second half; `uniform` spans the whole clip. `onset_window` needs `light_frame_in_clip` and `clip_fps` in the manifest, else it falls back to `uniform` |
| `data.window_pre_s`, `data.window_post_s` | 1.0, 0.25 | window around the light onset in seconds |
| `data.overlay_mask.enabled` | `true` | blank the scoreboard and light rows before resize and flip |
| `data.overlay_mask.y_min_frac` | 0.82 | first masked row as a fraction of frame height. When unset (`null`), it is derived from the overlay geometry in `configs/overlay/fie_default.yaml` (score digit top minus 14 px, 295/360) |
| `training.seed` | 42 | seeds python, numpy, torch, shuffling and workers |
| `training.selection_metric` | `macro_f1` | checkpoint selection and early stopping (`macro_f1` or `accuracy`) |
| `eval.flip_tta` | `true` | average `f(x)` with the L/R-swapped `f(flip(x))` in validation and evaluation (2x forward passes); makes the prediction flip-consistent (#19) |
| `eval.split` | `test` | split of the saved `split_manifest.csv` to evaluate |
| `model.use_video`, `model.use_kinematics` | `true`, `true` | switch streams (ablation presets in `configs/experiment/`) |
| `model.kinematics.cache_dir` | `data/kinematics` | cache written by `precompute-kinematics` |
| `model.kinematics.settings` | `null` | overrides of the kinematics defaults (part of the cache key) |
| `model.kinematics.compute_if_missing` | `false` | compute missing cache entries on the fly instead of raising |

Overlay geometry (light boxes, score digit crops, thresholds) is configured per broadcast template in `configs/overlay/`.

### Statistical Evaluation

A single test accuracy on a few hundred clips from a handful of matches is not enough to rank models (#17). Evaluation writes `predictions_<split>.csv`; `bootstrap` resamples matches (not clips) from them:

```bash
# Match-level bootstrap CIs for accuracy and macro-F1
python -m src bootstrap checkpoints/predictions_test.csv

# Pool several CV folds
python -m src bootstrap fold0/predictions_test.csv fold1/predictions_test.csv

# Paired comparison of two models on the same clips (McNemar test + bootstrap difference)
python -m src bootstrap a/predictions_test.csv --compare b/predictions_test.csv
```

Options: `--n-boot` (default 2000), `--seed`.

### Label Audit

The labels come from scoreboard OCR and carry an unknown noise rate (#11). Draw a stratified sample, label it by hand, then estimate noise rates:

```bash
python -m src audit-sample --manifest data/clips/foil/manifest.csv --n-clips 200
# fill human_label in audit/audit_sheet.csv (L/T/R, X = not identifiable); review clips via audit/index.html
python -m src audit-report --sheet audit/audit_sheet.csv --output-json audit/report.json
```

`audit-report` prints per-class and per-weapon noise rates with Wilson 95% intervals and the confusion matrix (human vs. automatic label).

---

## 6. Data Source and Usage Restrictions

The preprocessing pipeline downloads FIE broadcast videos from YouTube with yt-dlp. By default it uses no cookies. If YouTube rate-limits or asks for a login, pass `--cookies-browser {firefox,edge,chrome}` or `--cookies-file PATH` (a Netscape cookie export; more robust than reading a running browser's locked database).

- **YouTube Terms of Service**: automated downloading, in particular with session cookies, may conflict with YouTube's terms of service.
- **Copyright**: the broadcast footage is owned by the FIE and the respective broadcasters. Whether a use is covered by a research exception depends on the jurisdiction and has not been clarified for this project.
- **Redistribution**: do not redistribute downloaded videos, cut clips, or models trained on them without clarifying the rights first. Formal research access from the FIE is the cleanest option.

---

## 7. Licensing

The kinematic stream uses [ultralytics](https://github.com/ultralytics/ultralytics), which is licensed under **AGPL-3.0**. Distributing this project, or offering it as a network service, with ultralytics included triggers the AGPL obligations (source availability under a compatible license) unless an ultralytics enterprise license is obtained. The pose weights (`yolo11n-pose.pt`) are downloaded at runtime and fall under the same terms.

---

## 8. References

[1] Tong, Z., Song, Y., Wang, J., & Wang, L. (2022). *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training.* NeurIPS 2022. [[arXiv:2203.12602]](https://arxiv.org/abs/2203.12602)

[2] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. [[arXiv:2106.09685]](https://arxiv.org/abs/2106.09685)

[3] Fédération Internationale d'Escrime. *Technical Rules.* [[fie.org]](https://fie.org/fie/documents/rules)
