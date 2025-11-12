# Processing Pipeline
 
### Matlab Requirements  
 
- MATLAB >= 9.6 (R2019a)  
- Signal Processing Toolbox
- DSP System Toolbox
- Parallel Computing Toolbox
- MATLAB Parallel Server
- Polyspace Bug Finder
- [NPMK](https://github.com/BlackrockNeurotech/NPMK) for Blackrock recordings.
- Neuroshare for Ripple recordings.

### Python Requirements

```
pip install -r requirements.txt
```

## Steps

1. Parse recording using parse_ripple.m (Ripple recording) or parse_NSx.m (Blacrock recording)
2. Use new_check_lfp_power_NSX.m to create figures to analize the power spectrum and calculate notches
3. Open processing_pipeline.ipynb and follow the cells there. 

The processing pipeline includes comments for new Python users and details about how to use the spikeinterface library.