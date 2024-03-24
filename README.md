# Lowrance Sonar Data Explorer

## Quick start
0. Run `pip install -r requirements.txt`.
1. Edit `SL2` file path at `default.yaml`.
2. Run `run_lowmapper.ipynb` via `Run All`.
3. Folder outputs are at `runs` folder.

## Features
- Save ping records as .csv files all channels and each channels.
- Save high res sonograms for each channels.
- Save georeferenced sidescan as PNG and GeoTif images.

## TODO
- Apply preprocessing from PINGMAPPER's methods:"
  - Depth detection
  - Shadow removal
  - Coordinates correction
  - Others
- Apply PINGMAPPER's methods to remove water column.
- Apply segmentation (mapping of substrate) PINGMAPPER's river model. 