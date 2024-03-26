import os

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

class Exporter:
    def __init__(self, sonar, project_name):
        self.sonar = sonar
        self.sonar_df = sonar.df # all_channels
        self.project_name = project_name
        self.runs_folder = 'runs'

        self.sub_folders = ['csvs', 'sonograms', 'sidescan']
        self.create_folder_if_not_exists()

        self.primary_df = self.sonar_df.query(f"survey == 'primary'")
        self.downscan_df = self.sonar_df.query(f"survey == 'downscan'")
        self.sidescan_df = self.sonar_df.query(f"survey == 'sidescan'")

        # Define export paths
        self.csv_filenames = {
            'sonar_df': 'all_channels.csv',
            'primary_df': 'primary.csv',
            'downscan_df': 'downscan.csv',
            'sidescan_df': 'sidescan.csv'
        }

        # CSVs
        self.csv_export_path = os.path.join(
            self.runs_folder, self.project_name, self.sub_folders[0])
        
        # Images (plots, sonograms, etc.)
        self.image_export_path = os.path.join(
            self.runs_folder, self.project_name, self.sub_folders[1])
        
        # For side scan images (commonly high res and georeferenced)
        self.sidescan_export_path = os.path.join(
            self.runs_folder, self.project_name, self.sub_folders[2])


    def create_folder_if_not_exists(self):
        project_folder = os.path.join(self.runs_folder, self.project_name)

        # Create project folder if it doesn't exist
        if not os.path.exists(project_folder):
            os.makedirs(project_folder)

        # Create subfolders if they don't exist
        for sub_folder in self.sub_folders:
            folder_path = os.path.join(project_folder, sub_folder)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)


    # CSVs
    def export_data_to_csv(self):
        for dataframe, filename in self.csv_filenames.items():
            df = getattr(self, dataframe)  # Access DataFrame using its name
            df.to_csv(os.path.join(self.csv_export_path, filename))

    # Images
    def export_all_images(self):
        self.export_primary_image()
        self.export_downscan_image()
        self.export_sidescan_image()


    # TODO: move processing to separate file
    def process_different_zoom_levels(self, df):
        """Used to resize the `primary` and `downscan` sonar images to match
        the different zoom levels; based on max_range value.
        """
        max_range_value = df['max_range'].max()
        max_range_ratio = np.array(df['max_range'] / max_range_value)
        image = np.stack(df["frames"])

        def downsample_row(row, ratio):
            indices = np.linspace(0, len(row)-1, int(len(row) *ratio)).astype(int)
            #indices = np.linspace(0, len(row)-1, int(len(row) * (1 - ratio))).astype(int)
            return row[indices]

        # Downsample each row and find the length of the longest row
        # note: this is faster than using img libraries resize as we are always
        # downsampling
        downsampled_rows = [downsample_row(row, ratio) for row, ratio in zip(image, max_range_ratio)]
        max_length = max(len(row) for row in downsampled_rows)
        #max_length = max_frame_size[0]

        # Pad the downsampled rows with zeros to match the length of the longest row
        image_resized = np.array([np.pad(row, (0, max_length - len(row)), 'constant') for row in downsampled_rows])

        return image_resized.transpose()
    

    def export_primary_image(self):
        image = self.process_different_zoom_levels(self.primary_df)
        img = Image.fromarray(image)
        img.save(f'{self.image_export_path}/primary.jpg')


    def export_downscan_image(self):
        image = self.process_different_zoom_levels(self.downscan_df)
        img = Image.fromarray(image)
        img.save(f'{self.image_export_path}/downscan.jpg')


    def export_sidescan_image(self):
        image = np.stack(self.sidescan_df['frames'])
        image = image.transpose()
        img = Image.fromarray(image)
       
        img.save(f'{self.image_export_path}/sidescan.jpg')


    def export_georeferenced_sidescan(self):
        data = self.sidescan_df
        dist = [np.linspace(start, stop, num = len(f)) for start, stop, f in zip(
            data["min_range"],
            data["max_range"],
            data["frames"]
        )]
        dist_stack = np.stack(dist)
        sidescan_z = self.sonar.image("sidescan") # Todo: rename, not image

        sidescan_x = np.expand_dims(data["x"], axis=1) + dist_stack * np.cos(
            np.expand_dims(data["gps_heading"], axis=1))
        sidescan_y = np.expand_dims(data["y"], axis=1) - dist_stack * np.sin(
            np.expand_dims(data["gps_heading"], axis=1))

        # TODO: utils
        sidescan_long = self.sonar._x2lon(sidescan_x)
        sidescan_lat = self.sonar._y2lat(sidescan_y)

        # Used to adjust values to the correct pixel coordinates
        min_x = np.min(sidescan_x)
        min_y = np.min(sidescan_y)

        # Scale the coordinates to pixel values by resolution (meter per pixel)
        resolution = 1
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

        # We want transparent image
        image = np.zeros((height, width, 4), dtype=np.uint8)

        # Assign pixel values
        image[sidescan_y_scaled, sidescan_x_scaled, 0] = sidescan_z
        image[sidescan_y_scaled, sidescan_x_scaled, 1] = sidescan_z
        image[sidescan_y_scaled, sidescan_x_scaled, 2] = sidescan_z
        image[sidescan_y_scaled, sidescan_x_scaled, 3] = 255

        # since x y coordinate are actually starts bottom left, 
        # while image application of pixel coordinates starts from top left,
        # we fix the orientation
        image_np = np.flipud(image)

        # Create a Pillow image from the NumPy array
        image = Image.fromarray(image_np, 'RGBA')

        # Save the image as PNG with transparency
        image.save(f'{self.sidescan_export_path}/sidescan.png', 'PNG')

        # GEOTIF
        # Determine the bounds of lat long
        min_long = np.min(sidescan_long)
        max_long = np.max(sidescan_long)
        min_lat = np.min(sidescan_lat)
        max_lat = np.max(sidescan_lat)
        
        transform = from_bounds(min_long, min_lat, max_long, max_lat, width, height)

        with rasterio.open(
            f'{self.sidescan_export_path}/sidescan.tif',
            'w',
            driver='GTiff',
            height=image_np.shape[0],
            width=image_np.shape[1],
            count=4,
            dtype=image_np.dtype,
            crs='EPSG:4326',
            transform=transform,
        ) as dst:
            dst.write(image_np.transpose(2, 0, 1))