import os
from pathlib import Path

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

import geopandas as gpd
from shapely.geometry import Point

class Exporter:
    def __init__(self, config):
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
        self.shapefiles_folder = os.path.join(
            self.save_path, 
            self.project_name,
            'shapefiles'
        )
        if not os.path.exists(self.csvs_folder):
            os.makedirs(self.csvs_folder)
        if not os.path.exists(self.sonograms_folder):
            os.makedirs(self.sonograms_folder)
        if not os.path.exists(self.sidescan_folder):
            os.makedirs(self.sidescan_folder)
        if not os.path.exists(self.shapefiles_folder):
            os.makedirs(self.shapefiles_folder)

            
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


    def export_points_shapefile(self, gdf):
        print('Exporting points shapefile...')
        gdf.to_file(f'{self.shapefiles_folder}/points.shp')