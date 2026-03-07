# Examples

- [OpenADMET-LogD.ipynb](OpenADMET-LogD.ipynb) — A simple example of how to train and evaluate a Graph Transformer model on the OpenADMET LogD endpoint using `gt-pyg`.
- [train_logd.ipynb](train_logd.ipynb) — Single-task training for **LogD** prediction (tag `v1.6.0`).
- [train_ksol.ipynb](train_ksol.ipynb) — Single-task training for **KSOL** (solubility) prediction (tag `v1.6.0`). The model is trained on the log-transformed endpoint `LogS = log10((KSOL + 1) * 1e-6)`.
- [compare_predictions.ipynb](compare_predictions.ipynb) — Compares single-task models (`v1.6.0`, one model per endpoint) against the beardy-polonium ensemble submission (`v1.3.0`, ensemble of multi-task models) on LogD and log-transformed KSOL. Metrics are reported on the full test set, the leaderboard (public) subset, and the private subset.
