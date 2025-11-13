# MCWs Python Pipeline - Quick Start

## Overview
The MCWs pipeline processes neural recordings through spike detection, artifact removal, clustering, and quality assessment. All steps run from `main_Python.ipynb`.

## Setup
1. Place raw data files in `input/` folder
2. Install dependencies: `pip install -r requirements.txt`
3. Run cells sequentially in `main_Python.ipynb`

## Configuration
- **Parallel processing**: Set `use_parallel = True/False` (default: True)
- **Plotting**: Configure `save_plots` and `show_plots` parameters
- **Parameters**: Adjust thresholds, cluster counts, and other settings in configuration cell

## Pipeline Functions

### Starting from NSX Files (Recommended: Run all steps)
1. **NSX Parsing** - Extracts channel data from Blackrock .nsx files
2. **Channel Reading** - Loads and preprocesses continuous data
3. **Spike Detection** - Detects threshold crossings and extracts waveforms
4. **Collision Detection** - Removes multi-channel artifacts (events appearing across multiple channels simultaneously)
5. **Quarantine (Artifact Removal)** - Masks suspicious spikes based on waveform shape criteria
6. **Clustering** - GMM-based feature extraction and clustering
   - **Note**: First run may be slow due to distribution fitting
   - Fitted distributions are saved and reused for faster subsequent runs
7. **Quality Metrics** - Calculates isolation, SNR, ISI violations, etc.
8. **Rescue** - Re-evaluates quarantined spikes and assigns them to clusters if appropriate
9. **Final Metrics** - Updated quality metrics including rescued spikes

### Starting from Preprocessed Data (.mat files)
- **Skip to Channel Reading** (around cell/line 176)
- All subsequent functions remain the same

## Outputs
- **Data**: `output/times_*.mat`, `cluster_*.mat`, `metrics_*.csv`
- **Reports**: `output/*_report.pdf` with waveforms, features, ISI histograms

## Quick Example
```python
# Process all channels with default settings
process_all_channels(channels='all')

# Process specific channel
process_single_channel(channel=328)
```

