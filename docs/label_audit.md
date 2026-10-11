# Label Audit Workflow

## Overview

All labels in the fencing-ai dataset are produced automatically by light-template matching plus scoreboard OCR. To estimate label noise and establish a clean evaluation set, we manually audit a stratified random sample of clips.

## Workflow

### 1. Sample Clips

Generate a stratified random sample (e.g., 200 clips) and write an HTML index for manual review:

```bash
python scripts/audit_sample.py \
  --manifest data/clips/foil/manifest.csv \
  --n-clips 200 \
  --seed 42 \
  --output-dir audit
```

Options:
- `--manifest` — Load clips from a specific manifest (default: scans `data/clips/{weapon}/manifest.csv`)
- `--n-clips` — Sample size (default: 200)
- `--seed` — Random seed for reproducibility (default: 42)
- `--stratify-by-hittype` — Include hit type in stratification (optional)
- `--output-dir` — Output directory (default: `audit/`)

This generates:
- **`audit/audit_sheet.csv`** — CSV with one row per clip, columns: `clip_path`, `weapon`, `hit_type`, `auto_label`, `human_label` (empty), `notes` (empty)
- **`audit/index.html`** — Embeds each clip as a playable `<video>` tag with metadata so you can preview before labeling

### 2. Manual Review & Labeling

Open `audit/index.html` in a browser and review each clip. For each clip, fill in `audit/audit_sheet.csv`:

- **`human_label`** — Your assessment of the correct label: `L` (left), `T` (touch/neutral), or `R` (right)
  - If the clip is ambiguous or unidentifiable, write `X` (this is tracked separately in the report)
- **`notes`** — Optional: brief comment on why you chose this label (e.g., "unclear due to low light")

Example filled row:
```
clip_path,weapon,hit_type,auto_label,human_label,notes
data/clips/foil/L_foil_m1_0.mp4,foil,On-On,L,L,clear parry-riposte
data/clips/foil/T_foil_m2_1.mp4,foil,On-No,T,R,light came up late on opponent advance
data/clips/foil/L_foil_m3_2.mp4,foil,On-On,L,X,obscured by another fencer
```

### 3. Generate Report

Once you've filled the sheet, compute noise rates and generate a report:

```bash
python scripts/audit_report.py \
  --sheet audit/audit_sheet.csv \
  --output-json audit/audit_report.json
```

This prints:
- **Overall label noise rate** with 95% confidence interval (Wilson score)
- **Per-class noise rates** (L, T, R)
- **Per-weapon noise rates** (foil, sabre, epee)
- **Confusion matrix** (human label vs. auto label)
- **Count of unidentifiable clips** (human_label = "X")

Example output:
```
# Label Audit Report

## Summary
- **Total clips audited:** 200
- **Total errors:** 12
- **Overall noise rate:** 6.00% [3.24%, 10.46%]
- **Unidentifiable:** 3

## Per-Class Noise Rate

| Label | N Clips | N Errors | Error Rate | 95% CI |
|-------|---------|----------|------------|--------|
| L | 70 | 4 | 5.71% | [1.84%, 13.75%] |
| T | 65 | 5 | 7.69% | [2.95%, 16.87%] |
| R | 62 | 3 | 4.84% | [1.26%, 12.87%] |

## Per-Weapon Noise Rate

| Weapon | N Clips | N Errors | Error Rate | 95% CI |
|--------|---------|----------|------------|--------|
| foil | 100 | 6 | 6.00% | [2.41%, 12.90%] |
| sabre | 60 | 4 | 6.67% | [2.07%, 15.80%] |
| epee | 37 | 2 | 5.41% | [0.95%, 17.87%] |
```

Outputs:
- Markdown summary to stdout (can be saved to a file or pasted into a report)
- `audit/audit_report.json` with structured results for further analysis

## Interpreting the Report

1. **Overall noise rate**: If > 5%, consider retraining or post-processing the auto-labeler
2. **Per-class noise rates**: Imbalanced noise rates suggest the auto-labeler is biased toward one class
3. **Confidence intervals**: Wider CIs indicate smaller sample sizes; consider auditing more clips for that class/weapon
4. **Confusion matrix**: Shows which mislabels are most common (e.g., L→T suggests temporal misalignment in light detection)
5. **Unidentifiable clips**: Consider removing these from the dataset or marking them as ambiguous

## Next Steps

Once the audit is complete:

1. **Set aside the audited clips** for a held-out evaluation set (use the filled `audit_sheet.csv` as ground truth)
2. **Train models** on the remaining dataset (non-audited clips)
3. **Evaluate on the audited clips** to get a clean estimate of model accuracy
4. **Fix auto-labeler issues** identified by the report (if > 10% noise rate)

## Example: Multi-Weapon Audit

To audit all weapons at once:

```bash
# Sample from each weapon proportionally
python scripts/audit_sample.py \
  --n-clips 60 \
  --output-dir audit

# This will load and sample from data/clips/{foil,sabre,epee}/manifest.csv
```

Then follow steps 2–3 above. The report will break down noise rates by weapon.
