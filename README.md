# AdversarialStudy-FL

Repository for experiments around adversarial behavior, privacy leakage, and robustness in federated/vertical federated learning.

## Subprojects and Main Folders

- `FL-Privacy-Leakage/` - Privacy leakage and tampering-focused FL experiments, configs, scripts, and notes.
- `byzantine-fl/` - ByzFL framework and demos for Byzantine-robust federated learning, attacks, and robust aggregators.
- `gan_attack_fl/` - GAN-based attack implementations and supporting modules for FL settings.
- `GAN-Attack-FederatedLearning/` - Additional GAN attack project workspace and related assets.
- `vfl_hidden_correlations/` - Vertical FL experiments for reconstructing/inferring hidden cross-party correlations.
- `configs/` - Shared experiment configuration files (JSON) used by scripts and notebooks.
- `scripts/` - Entry-point Python scripts to run baselines, sweeps, and attack experiments.
- `results/` - Generated outputs such as CSV summaries and experiment artifacts.

## Data and Notebooks

- `transaction_data.csv` / `sample_transaction_data.csv` - Transaction datasets used by tabular FL experiments.
- `*.ipynb` in root - Exploratory and demonstrative notebooks (benign FL, GAN attack FL, transaction FL, graph embedding analyses).

## Notes

- Project-specific implementation/observation docs are included as `*_plan-implementation-and-observations.md`.
- Most subprojects include their own local README files with setup and usage details.

