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
            'georeferenced'
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
            sonograms: dict with key(filename), value(image numpy) of WCR AND WCP
        """
        print('Exporting `sidescan` channel sonogram(s)...')
        # TODO: put looping outside
        for file_name, image in sonograms.items():
            img = Image.fromarray(image)
            img.save(f'{self.sonograms_folder}/{file_name}.jpg')


    def export_primary_sonogram(self, image):
        """
        Args:
            image: numpy
        """
        print('Exporting `primary` channel sonogram...')
        img = Image.fromarray(image)
        img.save(f'{self.sonograms_folder}/primary.jpg')


    def export_downscan_sonogram(self, image):
        """
        Args:
            image: numpy
        """
        print('Exporting `downscan` channel sonogram...')
        img = Image.fromarray(image)
        img.save(f'{self.sonograms_folder}/downscan.jpg')


    def export_georeferenced_sidescan(self, image_np, file_name, bounds):
        print('Exporting georeferenced sidescan(s)...')
        image = Image.fromarray(image_np, 'RGBA')
        width, height = image.size
        
        # Save the image as PNG with transparency
        image.save(f'{self.sidescan_folder}/{file_name}.png', 'PNG')

        # GEOTIF
        min_long, max_long, min_lat, max_lat = bounds
        
        transform = from_bounds(min_long, min_lat, max_long, max_lat, width, height)

        with rasterio.open(
            f'{self.sidescan_folder}/{file_name}.tif',
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