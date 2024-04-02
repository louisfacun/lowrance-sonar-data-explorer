import os
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

class Exporter:
    def __init__(self, config):
        #self.sonar = sonar
        #self.sonar_df = sonar.df # all_channels
        self.save_path = config['save_path']
        
        if config['filename_as_project_name']:
            self.project_name = Path(config['sonar_path']).stem
        else:
            self.project_name = config['project_name']
            
        self.csvs_folder = os.path.join(
            self.save_path, 
            self.project_name,
            'csvs'
        )
        self.sonograms_folder = os.path.join(
            self.save_path, 
            self.project_name,
            'sonograms'
        )
        self.sidescan_folder = os.path.join(
            self.save_path, 
            self.project_name,
            'sidescan'
        )
        if not os.path.exists(self.csvs_folder):
            os.makedirs(self.csvs_folder)
        if not os.path.exists(self.sonograms_folder):
            os.makedirs(self.sonograms_folder)
        if not os.path.exists(self.sidescan_folder):
            os.makedirs(self.sidescan_folder)

        # self.sub_folders = ['csvs', 'sonograms', 'sidescan']
        # self.create_folder_if_not_exists()

        # self.primary_df = self.sonar_df.query(f"survey == 'primary'")
        # self.downscan_df = self.sonar_df.query(f"survey == 'downscan'")
        # self.sidescan_df = self.sonar_df.query(f"survey == 'sidescan'")

        # Define export paths
        # self.csv_filenames = {
        #     'sonar_df': 'all_channels.csv',
        #     'primary_df': 'primary.csv',
        #     'downscan_df': 'downscan.csv',
        #     'sidescan_df': 'sidescan.csv'
        # }

        # CSVs
        # self.csv_export_path = os.path.join(
        #     self.runs_folder, self.project_name, self.sub_folders[0])
        
        # # Images (plots, sonograms, etc.)
        # self.image_export_path = os.path.join(
        #     self.runs_folder, self.project_name, self.sub_folders[1])
        
        # # For side scan images (commonly high res and georeferenced)
        # self.sidescan_export_path = os.path.join(
        #     self.runs_folder, self.project_name, self.sub_folders[2])
            
    def export_csvs(self, csvs):
        """
        Example:
            csvs = {
            'all.csv': all_df,
            'primary.csv': primary_df,
            'downscan.csv': downscan_df,
            'sidescan.csv': sidescan_df,
        }
        """
        print('Exporting csv(s)...')
        for file_name, df in csvs.items():
            df.to_csv(f'{self.csvs_folder}/{file_name}.csv')


    def export_sidescan_sonograms(self, sonograms):
        """
        Args:
            sonograms: dict with key(filename), value(image numpy)
        """
        print('Exporting sonograms(s)...')
        for file_name, image in sonograms.items():
            img = Image.fromarray(image)
            img.save(f'{self.sonograms_folder}/{file_name}.jpg')


    # def create_folder_if_not_exists(self):
    #     project_folder = os.path.join(self.runs_folder, self.project_name)

    #     # Create project folder if it doesn't exist
    #     if not os.path.exists(project_folder):
    #         os.makedirs(project_folder)

    #     # Create subfolders if they don't exist
    #     # for sub_folder in self.sub_folders:
    #     #     folder_path = os.path.join(project_folder, sub_folder)
    #     #     if not os.path.exists(folder_path):
    #     #         os.makedirs(folder_path)

    # CSVs
    # def export_data_to_csv(self):
    #     for dataframe, filename in self.csv_filenames.items():
    #         df = getattr(self, dataframe)  # Access DataFrame using its name
    #         df.to_csv(os.path.join(self.csv_export_path, filename))


    # # Images
    # def export_all_images(self):
    #     self.export_primary_image()
    #     self.export_downscan_image()
    #     self.export_sidescan_image()


    # # TODO: move processing to separate file
    # def process_different_zoom_levels(self, df):
    #     """Used to resize the `primary` and `downscan` sonar images to match
    #     the different zoom levels; based on max_range value.
    #     """
    #     max_range_value = df['max_range'].max()
    #     max_range_ratio = np.array(df['max_range'] / max_range_value)
    #     image = np.stack(df["frames"])

    #     def downsample_row(row, ratio):
    #         indices = np.linspace(0, len(row)-1, int(len(row) *ratio)).astype(int)
    #         #indices = np.linspace(0, len(row)-1, int(len(row) * (1 - ratio))).astype(int)
    #         return row[indices]

    #     # Downsample each row and find the length of the longest row
    #     # note: this is faster than using img libraries resize as we are always
    #     # downsampling
    #     downsampled_rows = [downsample_row(row, ratio) for row, ratio in zip(image, max_range_ratio)]
    #     max_length = max(len(row) for row in downsampled_rows)
    #     #max_length = max_frame_size[0]

    #     # Pad the downsampled rows with zeros to match the length of the longest row
    #     image_resized = np.array([np.pad(row, (0, max_length - len(row)), 'constant') for row in downsampled_rows])

    #     return image_resized.transpose()
    

    # def export_primary_image(self):
    #     image = self.process_different_zoom_levels(self.primary_df)
    #     img = Image.fromarray(image)
    #     img.save(f'{self.image_export_path}/primary.jpg')


    # def export_downscan_image(self):
    #     image = self.process_different_zoom_levels(self.downscan_df)
    #     img = Image.fromarray(image)
    #     img.save(f'{self.image_export_path}/downscan.jpg')



    def export_georeferenced_sidescan(self, image_np, bounds):
        print('Exporting georeferenced sidescan(s)...')
        image = Image.fromarray(image_np, 'RGBA')
        width, height = image.size
        
        # Save the image as PNG with transparency
        image.save(f'{self.sidescan_folder}/sidescan.png', 'PNG')

        # GEOTIF
        min_long, max_long, min_lat, max_lat = bounds
        
        transform = from_bounds(min_long, min_lat, max_long, max_lat, width, height)

        with rasterio.open(
            f'{self.sidescan_folder}/sidescan.tif',
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