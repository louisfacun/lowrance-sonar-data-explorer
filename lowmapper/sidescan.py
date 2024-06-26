import math

import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from tqdm import tqdm

from .exporter import Exporter
from .utils import x_to_longitude, y_to_latitude


# Process designed for sidescan
class SideScan:
    def __init__(self, df, config):
        # Expects data from sonar.sidescan_df
        # The data is a 2d numpy array where each rows are pings and row's values
        # It is the intensity of the signal (or frames in 0-255 range).
        self.df = df
        self.df = self.df.reset_index(drop=True)
        self._add_cum_distance_travelled()
        self._preprocess_points()
        self.image = np.stack(self.df["frames"])  # Sidescan image.
        self.config = config

    def _wcr(self, port, starboard):
        """Remove water column from the sidescan image.

        Args:
            image (numpy.ndarray): Sidescan image.

        Returns:
            numpy.ndarray: Sidescan image with water column removed.
        """

        depths = self.df["water_depth"]
        absolute_range = self.df["max_range"].iloc[0]

        # STARBOARD
        print("Slant correcting starboard...")
        srcDat = np.zeros((starboard.shape[0], starboard.shape[1])).astype(np.float32)
        horizontal_dist = np.sqrt(absolute_range**2 - depths**2).astype(int)

        for i in tqdm(range(starboard.shape[0])):
            depth = ((depths[i] * starboard.shape[1]) / absolute_range).astype(int)
            dd = depth**2
            slant_corrected = np.full(starboard.shape[1], np.nan, dtype=np.float32)
            dataExtent = 0

            # Iterate each sonar/ping return
            for j in range(starboard.shape[1]):
                if j >= depth:
                    intensity = starboard[i, j]
                    horizontal_dist = int(round(math.sqrt(j**2 - dd), 0))
                    slant_corrected[horizontal_dist] = intensity
                    dataExtent = horizontal_dist
            slant_corrected[dataExtent:] = 0
            valid = ~np.isnan(slant_corrected)
            slant_corrected[~valid] = np.interp(
                np.flatnonzero(~valid), np.flatnonzero(valid), slant_corrected[valid]
            )
            srcDat[i] = slant_corrected
        starboard_wcr = srcDat

        # PORT
        print("Slant correcting port...")
        port = np.fliplr(port)

        srcDat = np.zeros((port.shape[0], port.shape[1])).astype(np.float32)
        horizontal_dist = np.sqrt(absolute_range**2 - depths**2).astype(int)

        for i in tqdm(range(port.shape[0])):
            depth = ((depths[i] * port.shape[1]) / absolute_range).astype(int)
            dd = depth**2
            slant_corrected = np.full(port.shape[1], np.nan, dtype=np.float32)
            dataExtent = 0

            # Iterate each sonar/ping return
            for j in range(port.shape[1]):
                if j >= depth:
                    intensity = port[i, j]
                    horizontal_dist = int(round(math.sqrt(j**2 - dd), 0))
                    slant_corrected[horizontal_dist] = intensity
                    dataExtent = horizontal_dist
            slant_corrected[dataExtent:] = 0
            valid = ~np.isnan(slant_corrected)
            slant_corrected[~valid] = np.interp(
                np.flatnonzero(~valid), np.flatnonzero(valid), slant_corrected[valid]
            )
            srcDat[i] = slant_corrected
        port_wcr = np.fliplr(srcDat)

        # # Get first row' as all values are same for all rows
        # absolute_range = self.df["max_range"].iloc[0]

        # if not all(self.df["max_range"] == absolute_range):
        #     raise ValueError(
        #         "Values in 'max_range' column are not consistent across all rows."
        #     )

        # # Assumes port and starboard has same size
        # # TODO: Check if same size
        # n_pings = port.shape[0]  # number of pings in a session
        # n_frame = port.shape[1]  # number of pixel in a frame

        # depths = self.df["water_depth"]

        # # Horizontal distance of max range
        # horizontal_dist = np.sqrt(absolute_range**2 - depths**2).astype(int)

        # # Get equally spaced distances of horizontal
        # port_horizontal_dists = np.linspace(
        #     horizontal_dist, depths, n_frame, axis=1
        # ).astype(int)
        # starboard_horizontal_dists = np.linspace(
        #     depths, horizontal_dist, n_frame, axis=1
        # ).astype(int)

        # # Convert to horizontal distances to pixel value
        # port_horizontal_dists = (
        #     (port_horizontal_dists * n_frame) / absolute_range
        # ).astype(int)
        # starboard_horizontal_dists = (
        #     (starboard_horizontal_dists * n_frame) / absolute_range
        # ).astype(int)

        # port_wcr = np.zeros((n_pings, n_frame))
        # starboard_wcr = np.zeros((n_pings, n_frame))

        # for n in range(n_pings):
        #     port_frame = np.zeros((n_frame))
        #     starboard_frame = np.zeros((n_frame))

        #     port_horizontals = []
        #     starboard_horizontals = []

        #     for m in range(n_frame):
        #         if -port_horizontal_dists[n][m] not in port_horizontals:
        #             port_frame[m] = port[n][-port_horizontal_dists[n][m]]
        #             port_horizontals.append(-port_horizontal_dists[n][m])

        #         if not starboard_horizontal_dists[n][m] not in starboard_horizontals:
        #             starboard_frame[m] = starboard[n][starboard_horizontal_dists[n][m]]
        #             starboard_horizontals.append(starboard_horizontal_dists[n][m])

        #     port_valid = ~np.isnan(port_frame)
        #     port_frame[~port_valid] = np.interp(np.flatnonzero(~port_valid), np.flatnonzero(port_valid), port_frame[port_valid])

        #     starboard_valid = ~np.isnan(starboard_frame)
        #     starboard_frame[~starboard_valid] = np.interp(np.flatnonzero(~starboard_valid), np.flatnonzero(starboard_valid), starboard_frame[starboard_valid])

        #     # convert float values to int of port_frame and starboard_frame
        #     port_frame = port_frame.astype(int)
        #     starboard_frame = starboard_frame.astype(int)

        #     port_wcr[n] = port_frame
        #     starboard_wcr[n] = starboard_frame
        port_wcr = port_wcr.astype(np.uint8)
        starboard_wcr = starboard_wcr.astype(np.uint8)

        port_wcr[port_wcr == 0] = 1
        starboard_wcr[starboard_wcr == 0] = 1

        return port_wcr, starboard_wcr

        # port_depths_px = ((depths * frame_px) / absolute_range).astype(int)
        # starboard_depths_px = ((depths * frame_px) / absolute_range).astype(int)

        # for i, (port_depth_px, starboard_depth_px) in enumerate(zip(port_depths_px, starboard_depths_px)):
        #     port_wcr[i] = np.concatenate(
        #         (
        #             np.zeros(port_depth_px).astype(int), # black
        #             port_wcr[i, :-port_depth_px]
        #         )
        #     )
        #     starboard_wcr[i] = np.concatenate(
        #         (
        #             starboard_wcr[i, starboard_depth_px:],
        #             np.zeros((starboard_depth_px)).astype(int) # black
        #         )
        #     )

    def _port(self):
        """Get port side of the sidescan image.

        Returns:
            numpy.ndarray: Port side of the sidescan image.
        """
        return self.image[:, : self.image.shape[1] // 2]

    def _starboard(self):
        """Get starboard side of the sidescan image.

        Returns:
            numpy.ndarray: Starboard side of the sidescan image.
        """
        return self.image[:, self.image.shape[1] // 2 :]

    def _egn(self, image):
        """Apply Empirical Gain Normalization to the sidescan image.

        Args:
            np.2d array

        Returns:
            np.2d array

        Note:
            Normalized data by mean, not necessarily in the range [0, 1].

        Reference:
            http://ss08.ccom.unh.edu/images/stories/abstracts/02_finlayson_empirical_backscatter_pmbs.pdf
        """
        mean_per_column = np.mean(image, axis=0)
        normalized_by_mean_image = image / mean_per_column

        # Convert in the range [0, 1]
        min_value = np.min(normalized_by_mean_image)
        max_value = np.max(normalized_by_mean_image)

        final = (normalized_by_mean_image - min_value) / (max_value - min_value)

        return (final * 255).astype(np.uint8)

    def _egn_min_max_stretch(self, image, min_value, max_value):
        """Apply min-max stretching to the sidescan EGN normalized image.

        Returns:
            numpy.ndarray: Stretched data in the range [0, 1].
        """

        # _min = np.min(self.egn_data)
        # _max = np.max(self.egn_data)
        stretched_data = (image - min_value) / (max_value - min_value)

        return (stretched_data * 255).astype(np.uint8)

    def _egn_percentile_clip_stretch(self, image, min_percent, max_percent):
        """Apply percentile clipping to the sidescan EGN normalized image.

        Args:
            min_percent (float): Lower percentile to clip (in 0-100%).
            max_percent (float): Upper percentile to clip (in 0-100%).

        Returns:
            numpy.ndarray: Stretched data in the range [0, 1].
        """
        # note: this gets min - max percentile values from the whole image
        min_value, max_value = np.percentile(image, (min_percent, max_percent))

        clipped_data = np.clip(image, min_value, max_value)
        stretched_data = (clipped_data - min_value) / (max_value - min_value)

        return (stretched_data * 255).astype(np.uint8)

    def _create_sonograms(self):
        image = self.image

        sonograms = {}

        # Remove shadows
        if self.config["remove_shadows"]:
            pass
            # print("Removing shadow on side scan sonogram...")

        # Water column present
        if self.config["water_column_present"]:
            print("Creating water column present...")
            sonograms["sidescan_wcp"] = image

        # Water column removed
        if self.config["water_column_removed"]:
            print("Creating water column removed...")
            port_wcr, starboard_wcr = self._wcr(self._port(), self._starboard())
            image = np.concatenate((port_wcr, starboard_wcr), axis=1)
            sonograms["sidescan_wcr"] = image

        # WCP and WCR
        for file_name, image in sonograms.items():
            # Apply EGN
            if self.config["egn"]:
                print(f"Applying EGN on {file_name}...")
                image = self._egn(image)

                # Apply EGN stretch
                if self.config["egn_stretch"] is not None:
                    # Apply min-max stretch

                    if self.config["egn_stretch"] == 1:
                        print(f"Applying EGN min-max stretch on {file_name}...")
                        pass  # TODO

                    # Apply percentile clip stretch
                    elif self.config["egn_stretch"] == 2:
                        print(f"Applying EGN percentile clip stretch on {file_name}...")
                        image = self._egn_percentile_clip_stretch(
                            image,
                            self.config["egn_stretch_factor_min"],
                            self.config["egn_stretch_factor_max"],
                        )
                    else:
                        # TODO: Raise error
                        # Stretch not found only hotdog
                        pass

            # Apply speed correction
            print(f"Applying speed correction on {file_name}...")
            image = self._apply_speed_correction(image)

            # Save
            sonograms[file_name] = image

        return sonograms

    def sonograms(self):
        "Export sonograms"
        sonograms = self._create_sonograms()

        exporter = Exporter(self.config)
        exporter.export_sidescan_sonograms(sonograms)

        return sonograms  # to be optionally pass to georeferencing

    def georeference(self, sonograms):
        "export georeferenced"
        """
            args: output from sonograms (egned and/or stretched)
        """

        exporter = Exporter(self.config)

        for file_name, image in sonograms.items():
            georeferenced_image, bounds = self._create_georeference(
                image, self.config["resolution"]
            )

            exporter.export_georeferenced_sidescan(
                georeferenced_image, file_name, bounds
            )

    def _create_georeference(self, image, resolution=1):
        """Georeference the sidescan image.

        Args:
            resolution (int, optional): Resolution in meters per pixel. Defaults to 1.

        Returns:
            numpy.ndarray: Georeferenced sidescan image.
        """
        # Determine the coordinates of each pixel
        distances = np.stack(
            [
                np.linspace(start, stop, len(f))
                for start, stop, f in zip(
                    self.df["min_range"], self.df["max_range"], self.df["frames"]
                )
            ]
        )
        sidescan_z = image  # image is the intensities
        sidescan_x = np.expand_dims(self.df["x"], axis=1) + distances * np.cos(
            np.expand_dims(self.df["gps_heading"], axis=1)
        )
        sidescan_y = np.expand_dims(self.df["y"], axis=1) - distances * np.sin(
            np.expand_dims(self.df["gps_heading"], axis=1)
        )

        # TODO: utils
        sidescan_long = x_to_longitude(sidescan_x)
        sidescan_lat = y_to_latitude(sidescan_y)

        # Used to adjust values to the correct pixel coordinates
        min_x = np.min(sidescan_x)
        min_y = np.min(sidescan_y)

        # Scale the coordinates to pixel values by resolution (meter per pixel)
        # resolution = 1
        sidescan_x_scaled = ((sidescan_x - min_x) / resolution).astype(int)
        sidescan_y_scaled = ((sidescan_y - min_y) / resolution).astype(int)

        # Determine the width and height of the image from scaled by resolution
        min_x_scaled = np.min(sidescan_x_scaled)
        max_x_scaled = np.max(sidescan_x_scaled)
        min_y_scaled = np.min(sidescan_y_scaled)
        max_y_scaled = np.max(sidescan_y_scaled)

        width = max_x_scaled - min_x_scaled
        height = max_y_scaled - min_y_scaled

        # Clip coordinates to ensure they are within the bounds of the image
        # Because pixel coordinates starts at 0
        sidescan_x_scaled = np.clip(sidescan_x_scaled, 0, width - 1)
        sidescan_y_scaled = np.clip(sidescan_y_scaled, 0, height - 1)

        # Blank transparent image
        image = np.zeros((height, width, 4), dtype=np.uint8)

        # Assign pixel values
        # image[sidescan_y_scaled, sidescan_x_scaled, 0] = sidescan_z
        # image[sidescan_y_scaled, sidescan_x_scaled, 1] = sidescan_z
        # image[sidescan_y_scaled, sidescan_x_scaled, 2] = sidescan_z
        # image[sidescan_y_scaled, sidescan_x_scaled, 3] = 255

        # Get pixel distance from center
        half_len = len(sidescan_z[0]) // 2
        decreasing = np.arange(half_len, 0, -1)
        increasing = np.arange(1, half_len + 1)
        sidescan_d = np.concatenate((decreasing, increasing))

        # Assign pixel values
        image_coords = {}  # track image coord

        for ping in tqdm(range(len(sidescan_z))):
            for y, x, z, d in zip(
                sidescan_y_scaled[ping],
                sidescan_x_scaled[ping],
                sidescan_z[ping],
                sidescan_d,
            ):
                coord = (y, x)
                if coord not in image_coords or (
                    coord in image_coords and d < image_coords[coord]
                ):
                    image[y, x, 0] = z
                    image[y, x, 1] = z
                    image[y, x, 2] = z
                    image[y, x, 3] = 255

                    image_coords[coord] = d

        # since x y coordinate actually starts from bottom left,
        # while image application of pixel coordinates starts from top left,
        # we fix the orientation by flipping vertically
        image = np.flipud(image)

        # Determine the bounds of lat long
        min_long = np.min(sidescan_long)
        max_long = np.max(sidescan_long)
        min_lat = np.min(sidescan_lat)
        max_lat = np.max(sidescan_lat)

        return image, (min_long, max_long, min_lat, max_lat)

    def _add_cum_distance_travelled(self):
        """Add columns cumulative distance traveled."""
        # Calculate the time elapsed between consecutive pings
        self.df = self.df.copy()

        self.df["time_elapsed"] = (
            self.df["datetime"] - self.df["datetime"].shift()
        ).fillna(pd.Timedelta(seconds=0))

        # Convert the time elapsed to seconds
        self.df["time_elapsed"] = self.df["time_elapsed"].dt.total_seconds()

        # Calculate the cumulative sum of the time elapsed
        self.df["cum_time_elapsed"] = self.df["time_elapsed"].cumsum()

        # Calculate distance traveled between consecutive rows
        self.df["dist_traveled"] = (
            self.df["gps_speed"] * self.df["cum_time_elapsed"].diff()
        )

        # Fill NaN values in the first row with 0
        self.df["dist_traveled"] = self.df["dist_traveled"].fillna(0)

        # Calculate cumulative distance traveled
        self.df["cum_dist_traveled"] = self.df["dist_traveled"].cumsum()

    def _apply_speed_correction(self, image):
        """Apply speed correction to the sidescan image by interpolating the
        frames based on the distance traveled."""
        cum_dist_traveled = self.df["cum_dist_traveled"].values
        frames = image

        # Define new evenly spaced distances
        new_distances = np.linspace(
            cum_dist_traveled.min(), cum_dist_traveled.max(), frames.shape[0]
        )

        # Interpolate along the row indices
        indices = np.arange(frames.shape[0])
        interpolated_indices = np.interp(new_distances, cum_dist_traveled, indices)

        # Round and cast to integers to use as indices
        interpolated_indices = np.round(interpolated_indices).astype(int)

        # Interpolate the frames based on the interpolated indices
        interpolated_frames = frames[interpolated_indices]

        return interpolated_frames

    def _preprocess_points(self):
        # Determine duplicates
        duplicates = self.df.duplicated(["x", "y"], keep="first")
        self.df.loc[duplicates, ["x", "y"]] = np.nan

        # Interpolate 'x'
        valid = ~np.isnan(self.df["x"])
        spl = splrep(self.df[valid]["datetime"], self.df[valid]["x"], k=3)
        self.df["x"] = np.where(
            np.isnan(self.df["x"]), splev(self.df["datetime"], spl), self.df["x"]
        )

        # Interpolate 'y'
        valid = ~np.isnan(self.df["y"])
        spl = splrep(self.df[valid]["datetime"], self.df[valid]["y"], k=3)
        self.df["y"] = np.where(
            np.isnan(self.df["y"]), splev(self.df["datetime"], spl), self.df["y"]
        )

    def __getattr__(self, attr):
        return getattr(self.image, attr)
