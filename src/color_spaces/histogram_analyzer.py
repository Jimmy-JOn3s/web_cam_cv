"""
Histogram Analyzer - Analyzes and displays image histograms
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


class HistogramAnalyzer:
    """Handles histogram calculation and visualization."""
    
    def __init__(self):
        """Initialize histogram analyzer."""
        self.hist_size = 256
        self.hist_range = (0, 256)
        
    def calculate_histogram(self, image, channels=None, mask=None):
        """
        Calculate histogram for given image.
        
        Args:
            image (numpy.ndarray): Input image
            channels (list): List of channel indices to calculate histogram for
            mask (numpy.ndarray): Mask for histogram calculation
            
        Returns:
            list: List of histograms for each channel
        """
        if len(image.shape) == 2:  # Grayscale
            hist = cv2.calcHist([image], [0], mask, [self.hist_size], self.hist_range)
            return [hist]
        else:  # Color image
            if channels is None:
                channels = [0, 1, 2]  # All channels
            
            histograms = []
            for channel in channels:
                hist = cv2.calcHist([image], [channel], mask, [self.hist_size], self.hist_range)
                histograms.append(hist)
            
            return histograms
    
    def create_histogram_image(self, image, width=512, height=400):
        """
        Create a visual representation of the histogram.
        
        Args:
            image (numpy.ndarray): Input image
            width (int): Width of histogram image
            height (int): Height of histogram image
            
        Returns:
            numpy.ndarray: Histogram visualization image
        """
        hist_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(image.shape) == 2:  # Grayscale
            hist = self.calculate_histogram(image)[0]
            
            # Normalize histogram
            cv2.normalize(hist, hist, 0, height, cv2.NORM_MINMAX)
            
            bin_width = int(width / self.hist_size)
            
            for i in range(1, self.hist_size):
                pt1 = (i * bin_width, height)
                pt2 = (i * bin_width, height - int(hist[i]))
                cv2.line(hist_image, pt1, pt2, (255, 255, 255), 1)
        
        else:  # Color image
            histograms = self.calculate_histogram(image)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
            
            # Normalize histograms
            for hist in histograms:
                cv2.normalize(hist, hist, 0, height, cv2.NORM_MINMAX)
            
            bin_width = int(width / self.hist_size)
            
            for c, (hist, color) in enumerate(zip(histograms, colors)):
                for i in range(1, self.hist_size):
                    pt1 = (i * bin_width, height)
                    pt2 = (i * bin_width, height - int(hist[i]))
                    cv2.line(hist_image, pt1, pt2, color, 1)
        
        return hist_image
    
    def display_histogram_matplotlib(self, image, title="Histogram"):
        """
        Display histogram using matplotlib.
        
        Args:
            image (numpy.ndarray): Input image
            title (str): Title for the plot
        """
        plt.figure(figsize=(10, 6))
        
        if len(image.shape) == 2:  # Grayscale
            hist = self.calculate_histogram(image)[0]
            plt.plot(hist, color='black')
            plt.title(f'{title} - Grayscale')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
        else:  # Color image
            histograms = self.calculate_histogram(image)
            colors = ['blue', 'green', 'red']
            labels = ['Blue', 'Green', 'Red']
            
            for hist, color, label in zip(histograms, colors, labels):
                plt.plot(hist, color=color, label=label)
            
            plt.title(f'{title} - Color Channels')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
            plt.legend()
        
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def get_histogram_stats(self, image):
        """
        Get statistical information from histogram.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            dict: Dictionary containing histogram statistics
        """
        stats = {}
        
        if len(image.shape) == 2:  # Grayscale
            stats['mean'] = np.mean(image)
            stats['std'] = np.std(image)
            stats['min'] = np.min(image)
            stats['max'] = np.max(image)
        else:  # Color image
            for i, channel in enumerate(['Blue', 'Green', 'Red']):
                channel_data = image[:, :, i]
                stats[f'{channel}_mean'] = np.mean(channel_data)
                stats[f'{channel}_std'] = np.std(channel_data)
                stats[f'{channel}_min'] = np.min(channel_data)
                stats[f'{channel}_max'] = np.max(channel_data)
        
        return stats
