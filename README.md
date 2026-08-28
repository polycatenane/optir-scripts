# O-PTIR Hyperspectral Analysis

`cellpose_peak_fitting.py` is a Marimo notebook for segmenting O-PTIR AC image stacks with Cellpose, extracting the AC spectra enclosed by each segmentation label, correcting them with timestamp-matched IR-power profiles, and fitting constrained Gaussian peaks.

Install the analysis environment on a CUDA-capable computer before opening the notebook.

```bash
pip install marimo "cellpose==3.1.1.1" accelerate numpy pandas scipy matplotlib
marimo edit cellpose_peak_fitting.py
```

Open the notebook, adjust the controls if needed, and click **Run AC analysis**. Accelerate selects the execution device; the default Cellpose model is `bact_fluor_cp3`.
