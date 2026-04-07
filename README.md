# Privacy Leakage via Selective Weight Tampering (FL-LM)

This project gives you a **safe, research-oriented simulation scaffold** for experimenting with:

- Federated training of a toy language model
- Selective weight tampering attacks during aggregation
- Privacy leakage indicators based on canary memorization

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.run_experiment --config configs/baseline.json
```

Run one experiment and append metrics to CSV:

```bash
python -m scripts.run_experiment --config configs/baseline.json --csv results/single_run.csv
```

Run a parameter sweep and generate plots:

```bash
python -m scripts.run_sweep --config configs/baseline.json --grid configs/sweep_grid.json --outdir results/sweep
```

## What this scaffold includes

- `fl_privacy_tampering/data.py`: synthetic client text + canary generation
- `fl_privacy_tampering/model.py`: tiny next-token language model (numpy)
- `fl_privacy_tampering/federated.py`: local training + FedAvg loop
- `fl_privacy_tampering/attacks.py`: selective parameter tampering hooks
- `fl_privacy_tampering/leakage.py`: privacy leakage metrics
- `scripts/run_experiment.py`: CLI runner with JSON config
- `scripts/run_sweep.py`: grid runner + CSV + plots

## Notes

- This is intentionally a toy setup for controlled experiments.
- Use it to compare attack intensity, target layers, and client selection.
- Extend to real LMs/frameworks once your hypotheses are validated.

## Metrics included

- Canary leakage: canary loss, control loss, exposure gap, target rank
- Membership inference: member/non-member mean loss and ROC-AUC
