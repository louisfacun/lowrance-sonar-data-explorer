# Lowrance Sonar Data Explorer

## Quick start
0. Run `pip install -r requirements.txt`.
1. Edit `SL2` file path at `default.yaml`.
2. Run `run_lowmapper.ipynb` via `Run All`.
3. Folder outputs are at `runs` folder.

# PingMapper Features

| Features | Implemented |
| --- | ----------- |
| Decode Lowrance® (tested on SL2) | Done |
| Export all ping records from .DAT and .SON files to .CSV. | Done | 
| Automatically detect depth (i.e. Zheng et al. 2021) and shadows in side scan channels. | Using sensor & Shadow not yet |
| Correct sonar backscatter with Empiracle Gain Normalization. | Done |
| Export un-rectified sonar tiles with water column present (WCP) AND/OR export un-rectified sonograms with water column removed (WCR) using Lowrance depth estimates OR automated depth detections. | WCP Done, WCR need to fix |
| Export speed corrected un-rectified sonograms. |
| Smooth and interpolate GPS track points. |
Export georectified WCP (spatially inaccurate due to presence of water column) AND/OR WCR sonar imagery for use in GIS w/wo shadows removed. | WCP Done, WCR not yet |
| Mosaic georectified sonar imagery. | |
| Automatically segment and classify substrate patches. | |


# Added Features
| Features | Implemented |
| - | - |
| Plot Tracks as shape file | |
| Plot bathymetry map | |
