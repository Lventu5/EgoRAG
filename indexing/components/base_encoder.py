import logging
from abc import ABC, abstractmethod

class BaseEncoder(ABC):
    """
    Abstract Base Class for a modular encoder component.
    Ensures all components share a common interface for loading models 
    and performing encoding.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None
        logging.info(f"Initialized {self.__class__.__name__} on {self.device}")

    @abstractmethod
    def load_models(self):
        """
        Abstract method to load all necessary models and processors
        into memory (e.g., onto self.device).
        """
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        """
        Abstract method to perform the encoding logic.
        """
        pass