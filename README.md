![MCWs (MiCroWire Sorter)](logo.png)
# MCWs (MiCroWire Sorter)
_A new framework for automated and reliable spike sorting in human intracerebral recordings_

MCWs (MiCroWire Sorter) is a modular pipeline for automated spike sorting of human intracerebral microwire recordings.  
It is designed to handle large-scale clinical datasets and to provide transparent, quality-controlled single-unit isolation.

---

## Project status

This repository contains a **beta version** of MCWs (MiCroWire Sorter).

- You may encounter bugs or unexpected behavior.
- If you find an issue, please open a GitHub issue with a short description, error message, and (when possible) a minimal example.
- We are actively developing this pipeline and will continue to update and refine it.

---

## Pipeline overview

MCWs implements an end-to-end workflow:

1. **Initial Data Loading & Pre-processing**
   - Data-driven notch filtering and basic cleaning
   - Channel selection and basic QC

2. **Spike Detection**
   - Automated thresholding and temporal filtering
   - Candidate spike extraction per channel

3. **Artifact Rejection**
   - **Bundle-level** rejection of obvious non-neural events  
   - **Within-channel** rejection using waveform shape and decision logic

4. **Feature Extraction**
   - Waveform- and time-domain features  
   - Distribution-based features (e.g., GMM fits, variability measures)  
   - Dimensionality reduction / selection for clustering

5. **Clustering (Super Paramagnetic Clustering; SPC)**
   - Unsupervised clustering of spike features
   - Automatic identification of putative units

6. **Refine Sorting**
   - Merge/split operations guided by waveform similarity and firing statistics  
   - Task-specific response profiles (e.g., peri-event histograms)

7. **Quality Metrics**
   - Unit stability and isolation indices  
   - Waveform consistency and firing rate characteristics  
   - Visual summaries for manual review

---

## Key innovations

- **Fully modular human-microwire pipeline**: every stage (detection, artifact rejection, feature extraction, clustering, refinement) is separable and can be swapped or extended.
- **Rich artifact and quality control**: two-level artifact rejection and extensive unit quality metrics designed specifically for noisy human intracerebral recordings.

---

## Data format requirements

The current implementation expects:

- Neural data in **NC5 `int16`** format, and  
- A corresponding `NSx.mat` file containing all metadata required to read the data  
  (e.g., sampling rate, channel map, recording duration, etc.). See ["Sample_of_NSx.mat"](https://github.com/Reylab/MCWs/blob/main/Sample_of_NSx.mat) for an exemplary file.

If your recordings are in a **different format**, you will need to:

1. Implement a **parser** that converts your raw data into **NC5 `int16`** files.
2. Create a compatible `NSx.mat` file with the necessary metadata fields.

Once you have `*.nc5` (`int16`) and the associated `NSx.mat`, you can use this repository **without further changes** to the main pipeline.

---

