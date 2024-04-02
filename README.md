# Lowrance Sonar Data Explorer

## Quick start
0. Run `pip install -r requirements.txt`.
1. Edit `SL2` file path at `default.yaml`.
2. Run `run_lowmapper.ipynb` via `Run All`.
3. Folder outputs are at `runs` folder.

## Features
- (raw) Save ping records as .csv files all channels and each channels.
- (raw) Save high res sonograms for each channels.
- (raw) Save Georeferenced sidescan as PNG and GeoTif images.
- Apply Empirical Gain Normalization with Min-Max and Percentile Clip stretch (PingMapper).

## TODO
- Apply preprocessing from PINGMAPPER's methods:"
  - Depth detection (for removing water column)
  - Shadow removal (for removing water column)
  - Coordinates correction
  - Others
- Apply PINGMAPPER's methods to remove water column.
- Apply segmentation (mapping of substrate) PINGMAPPER's river mode (wcp or wcr)
- Georeferenced track points as shapefile (using water depth for the color map).
- Plot tracks as graphs.
