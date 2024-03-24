import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_bounds

class Exporter:
    def __init__(self, sonar, name):
        self.sonar = sonar
        self.sonar_df = sonar.df

        # TODO: put the creation of directory here
        # csv
        self.csv_export_path = f'runs/{name}/csvs'
        self.primary_df = self.sonar_df.query(f"survey == 'primary'")
        self.downscan_df = self.sonar_df.query(f"survey == 'downscan'")
        self.sidescan_df = self.sonar_df.query(f"survey == 'sidescan'")

        # sonograms
        self.image_export_path = f'runs/{name}/sonograms'

        # side scan
        self.sidescan_export_path = f'runs/{name}/sidescan'

    # CSV
    def export_all_to_csv(self):
        self.sonar_df.to_csv(
            f'{self.csv_export_path}/all_channels.csv')
        

    def export_primary_to_csv(self):
        self.primary_df.to_csv(
            f'{self.csv_export_path}/primary.csv')
        

    def export_downscan_to_csv(self):
        self.downscan_df.to_csv(
            f'{self.csv_export_path}/downscan.csv')
        

    def export_sidescan_to_csv(self):
        self.sidescan_df.to_csv(
            f'{self.csv_export_path}/sidescan.csv')


    def export_multiple_to_csv(self):
        self.export_all_to_csv()
        self.export_primary_to_csv()
        self.export_downscan_to_csv()
        self.export_sidescan_to_csv()

    # Images
    def export_all_images(self):
        self.export_primary_image()
        self.export_downscan_image()
        self.export_sidescan_image()


    def export_primary_image(self):
        image_data = np.array(self.primary_df['frames'].tolist(), dtype=np.uint8)
        img = Image.fromarray(image_data)
        img = img.rotate(90, expand=True)
        img.save(f'{self.image_export_path}/primary.jpg')


    def export_downscan_image(self):
        image_data = np.array(self.downscan_df['frames'].tolist(), dtype=np.uint8)
        img = Image.fromarray(image_data)
        img = img.rotate(90, expand=True)
        img.save(f'{self.image_export_path}/downscan.jpg')


    def export_sidescan_image(self):
        image_data = np.array(self.sidescan_df['frames'].tolist(), dtype=np.uint8)
        img = Image.fromarray(image_data)
        img = img.rotate(90, expand=True)
        img.save(f'{self.image_export_path}/sidescan.jpg')


    def export_georeferenced_sidescan(self):
        data = self.sidescan_df
        dist = [np.linspace(start, stop, num = len(f)) for start, stop, f in zip(
            data["min_range"],
            data["max_range"],
            data["frames"]
        )]
        dist_stack = np.stack(dist)
        sidescan_z = self.sonar.image("sidescan")

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