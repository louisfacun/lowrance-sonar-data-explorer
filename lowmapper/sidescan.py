import numpy as np


# Process designed for sidescan
class SideScan:
    def __init__(self, data):
        # Assumes data from sonar.image("sidescan")
        # data is a 2d numpy array where each rows are pings and row's values
        # are the intensity of the signal (or frames in 0-255 range).
        self.data = data
        self.egn_data = None
    

    def apply_egn(self):
        """Apply Empirical Gain Normalization to the sidescan image.

        Note: 
            Normalized data by mean, not necessarily in the range [0, 1].

        Reference:
            http://ss08.ccom.unh.edu/images/stories/abstracts/02_finlayson_empirical_backscatter_pmbs.pdf
        """

        mean_per_column = np.mean(self.data, axis=0)
        normalized_data = self.data / mean_per_column

        self.egn_data = normalized_data


    def apply_min_max_stretch(self):
        """Apply min-max stretching to the sidescan EGN normalized image.
        
        
        Returns:
            numpy.ndarray: Stretched data in the range [0, 1].
        """
    
        min_value = np.min(self.egn_data)
        max_value = np.max(self.egn_data)
        stretched_data = (self.egn_data - min_value) / (max_value - min_value)

        return stretched_data
    

    def apply_percentile_clip(self, min_percent, max_percent):
        """Apply percentile clipping to the sidescan EGN normalized image.
        
        Args:
            min_percent (float): Lower percentile to clip (in 0-100%).
            max_percent (float): Upper percentile to clip (in 0-100%).
            
        Returns:
            numpy.ndarray: Stretched data in the range [0, 1].

        """
        min_value, max_value = np.percentile(
            self.egn_data, (min_percent, max_percent))
        
        clipped_data = np.clip(self.egn_data, min_value, max_value)
        stretched_data = (clipped_data - min_value) / (max_value - min_value)

        return stretched_data
