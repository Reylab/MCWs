# MCW Data Processing Pipeline

The pipeline is executed via `processing_steps_MCW_reduced`. This function acts as a wrapper for all relevant subfunctions, which can be run sequentially or independently.

## General Configuration

- **Parallel Processing:** Set to `true` by default. Change the `parallel` variable to `false` to run serially.
- **Visualization:** Plot generation (PNG/MATLAB figures) can be toggled via parameter inputs.
- **Setup:** Ensure the directory containing this code is added to your MATLAB path.

---

## Workflow: Starting with NSX File

Recommended for raw data processing.

**Setup:**
1. Update the path variable to match your local machine.
2. Set the directory to the location of the NSX file.

**Steps:**

1. **NSX Parsing:** Initial file parsing.
2. **NCX Channel Reading:** Reads channel data.
3. **Spike Detection:** Identifies potential events.
4. **Collision Detection:** Flags events occurring simultaneously across multiple channels (likely artifacts).
5. **Quarantine Spikes:** - Removes spikes that do not meet peak shape criteria.
   - Creates a mask to exclude them from initial clustering; they are reviewed later.
6. **Clustering:** - Uses GMM (Gaussian Mixture Model) feature extraction.
   - *Note:* Requires distribution fitting. This is computed once and saved, but may be slow on personal laptops.
7. **Metrics:** Prints initial clustering and spike shape metrics.
8. **Rescue Mask:** Re-evaluates quarantined spikes to see if they fit into established clusters.
9. **Final Metrics:** Prints updated metrics including rescued spikes.

---

## Workflow: NCX or Single Channel

For pre-parsed NCX files or single-channel data:

- **Start:** Begin execution at **line 176**.
- **Process:** Steps from *Spike Detection* onwards remain identical to the NSX workflow.

---

## Project Structure

A typical directory setup for this pipeline looks like this:

```text
MATLAB pipeline/
├── processing_steps_MCW_reduced.m   # Primary entry point
├── batch_files/                     # Core processing scripts
│   ├── Get_spikes.m                 # Spike detection
│   ├── Do_features.m                # Feature extraction
│   ├── Do_clustering.m              # Clustering
│   ├── rescue_spikes.m              # Rescue quarantined spikes
│   ├── compute_metrics_batch.m      # Metrics computation
│   ├── separate_collisions.m        # Collision detection
│   ├── clustering/                  # Clustering subfunctions
│   ├── compute metrics/             # Metrics subfunctions
│   ├── feature extraction/          # Feature extraction subfunctions
│   └── setup/                       # Setup and parameter functions
├── codes_emu/                       # EMU-specific utilities
├── useful_functions/                # Helper functions
└── data/
    └── experiment_file.nsx          # Raw data (Set MATLAB path here)