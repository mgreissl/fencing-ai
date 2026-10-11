# Legacy scripts

Archived, not maintained. Use the CLI (`python -m src <command>`) instead.

| Old script | Replacement |
| :--- | :--- |
| `0-find_vids.py` (collect playlist URLs) | `python -m src harvest --weapon foil` (also verifies title weapon code and the broadcast overlay) |
| `1-download_vids.py` (download into `precut/`) | `python -m src preprocess --weapon foil` (downloads each match itself, cuts, labels, deletes the source video) |
| `2-cut_and_label.py` | `python -m src preprocess` (root wrapper kept, prints a deprecation note) |
| `6-train_AI.py` | `python -m src train` (root wrapper kept) |
| `7-evaluate_AI.py` | `python -m src evaluate` (root wrapper kept) |
