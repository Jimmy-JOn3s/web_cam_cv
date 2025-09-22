"""
Base Filter - Abstract base class for all image filters
"""
from abc import ABC, abstractmethod
import numpy as np


class BaseFilter(ABC):
    """Abstract base class for image filters."""
    
    def __init__(self, name):
        """
        Initialize base filter.
        
        Args:
            name (str): Name of the filter
        """
        self.name = name
        self.parameters = {}
    
    @abstractmethod
    def apply(self, image, **kwargs):
        """
        Apply the filter to an image.
        
        Args:
            image (numpy.ndarray): Input image
            **kwargs: Filter-specific parameters
            
        Returns:
            numpy.ndarray: Filtered image
        """
        pass
    
    @abstractmethod
    def get_parameter_info(self):
        """
        Get information about filter parameters.
        
        Returns:
            dict: Dictionary containing parameter information
        """
        pass
    
    def set_parameter(self, param_name, value):
        """
        Set a filter parameter.
        
        Args:
            param_name (str): Parameter name
            value: Parameter value
        """
        self.parameters[param_name] = value
    
    def get_parameter(self, param_name, default=None):
        """
        Get a filter parameter value.
        
        Args:
            param_name (str): Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        return self.parameters.get(param_name, default)
    
    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.parameters.clear()
    
    def __str__(self):
        """String representation of the filter."""
        return f"{self.name} Filter - Parameters: {self.parameters}"
