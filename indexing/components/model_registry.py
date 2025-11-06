"""
ModelRegistry: Centralized model lifecycle management for EgoRAG.

This module provides a singleton registry for managing model loading and unloading,
reducing memory fragmentation and ensuring models are loaded only once.
"""

from typing import Any, Callable, Dict, Optional
import torch
from threading import Lock


class ModelRegistry:
    """
    Singleton registry for managing model loading and lifecycle.
    
    Models are lazily loaded on first request and cached for subsequent access.
    Supports manual unloading of individual models or all models at once.
    """
    
    _instance: Optional['ModelRegistry'] = None
    _lock: Lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelRegistry, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._models: Dict[str, Any] = {}
        self._loaders: Dict[str, Callable[[], Any]] = {}
        self._initialized = True
    
    def register(self, name: str, loader: Callable[[], Any]) -> None:
        """
        Register a model loader function.
        
        Args:
            name: Unique identifier for the model
            loader: Callable that returns the loaded model when invoked
        """
        self._loaders[name] = loader
    
    def get(self, name: str) -> Any:
        """
        Get a model by name, loading it lazily if not already cached.
        
        Args:
            name: Unique identifier for the model
            
        Returns:
            The loaded model
            
        Raises:
            KeyError: If no loader is registered for this name
        """
        if name not in self._loaders:
            raise KeyError(f"No loader registered for model '{name}'")
        
        if name not in self._models:
            self._models[name] = self._loaders[name]()
        
        return self._models[name]
    
    def unload(self, name: str) -> None:
        """
        Unload a specific model from memory.
        
        Args:
            name: Unique identifier for the model to unload
        """
        if name in self._models:
            model = self._models.pop(name)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def unload_all(self) -> None:
        """Unload all models from memory and clear cache."""
        for name in list(self._models.keys()):
            self.unload(name)
        self._models.clear()
    
    def is_loaded(self, name: str) -> bool:
        """Check if a model is currently loaded."""
        return name in self._models
    
    def registered_models(self) -> list:
        """Get list of all registered model names."""
        return list(self._loaders.keys())


class GPUMemoryGuard:
    """
    Context manager for GPU memory management.
    
    Automatically clears CUDA cache on entry and exit to reduce fragmentation.
    Useful for wrapping inference blocks to ensure memory is released.
    
    Example:
        with GPUMemoryGuard():
            # Perform inference
            outputs = model(inputs)
    """
    
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False


# Global singleton instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global ModelRegistry singleton instance."""
    return _registry
