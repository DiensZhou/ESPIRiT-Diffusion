# ESPIRiT Diffusion

This repository contains code for VWI MRI reconstruction with ESPIRiT/SENSE coil
sensitivity maps and score-based diffusion sampling.

The GitHub package is source-focused and includes two anonymized example files
for demonstration.

## Contents

- `models/`, `configs/`, `utils/`, `src/`, `op/`: model, config, data, and CUDA
  extension code.
- `main.py`: train/sample entry point using `ml_collections` configs.
- `generate_smaps.py`: BART ESPIRiT CSM generation helper.
- `mask_random/`, `mask_cartesian/`, `mask_caipi/`: sampling masks.
- `example1.mat`, `example2.mat`: anonymized examples for demonstrating
  the expected input format.
- `results/2026_04_24T17_03_51_ncsnpp_vesde_bart_True_alpha_348_std_N_100zeropadding/checkpoints/checkpoint_400.pth`:
  pretrained checkpoint tracked with Git LFS.

## Example Data

The repository includes two anonymized example files. They are provided only to
demonstrate the expected input format.

Each example contains:

- `ksp`: `(1, 302, 256, 18)`, complex64
- `csm`: `(1, 302, 256, 18, 4)`, complex64
- `kernel`: `(1, 5, 5, 18, 18)`, complex64

No volunteer identifiers or acquisition-source fields are included in the
example files.

## Setup

Install PyTorch for your CUDA version first, then install the remaining Python
dependencies:

```bash
pip install -r requirements.txt
```

If you need to estimate ESPIRiT maps with `generate_smaps.py`, install BART and
ensure both the `bart` command and the BART Python module are available.

## Run Reconstruction

Run ESPIRiT-Diffusion reconstruction for both included examples on GPU 0:

```bash
bash test_fastMRI.sh espiritdiff "" 0
```

The empty second argument keeps the default example list from
`configs/espiritdiff/ncsnpp_continuous.py`, which runs both `example1.mat` and
`example2.mat`.

Run one example only:

```bash
bash test_fastMRI.sh espiritdiff example1.mat 0
bash test_fastMRI.sh espiritdiff example2.mat 0
```

Reconstruction outputs are saved under `recon_result/`.

## Notes

- Reconstruction outputs are written to `recon_result/`, which is ignored by
  Git.
- Runtime outputs under `results/` are ignored by Git except for the included
  pretrained checkpoint listed above.
- The default config in `configs/espiritdiff/ncsnpp_continuous.py` uses the
  included example `.mat` files and can be overridden from the command line.
