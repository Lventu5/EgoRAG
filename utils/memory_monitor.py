"""
Memory Monitoring Utilities

Provides tools to monitor and log both CPU RAM and GPU memory usage in real-time.
"""
import torch
import psutil
import logging
import os
from functools import wraps
from contextlib import contextmanager
from typing import Optional


class MemoryMonitor:
    """Monitor and log CPU RAM and GPU memory usage."""
    
    @staticmethod
    def get_cpu_memory_info() -> dict:
        """
        Get current CPU RAM statistics for the current process.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        virtual_mem = psutil.virtual_memory()
        
        rss_mb = mem_info.rss / 1024**2  # Resident Set Size (actual physical memory used)
        vms_mb = mem_info.vms / 1024**2  # Virtual Memory Size
        available_mb = virtual_mem.available / 1024**2
        total_mb = virtual_mem.total / 1024**2
        used_pct = virtual_mem.percent
        
        return {
            "rss": rss_mb,  # Physical RAM used by process
            "vms": vms_mb,  # Virtual memory used by process
            "available": available_mb,  # Available RAM in system
            "total": total_mb,  # Total RAM in system
            "used_pct": used_pct  # System-wide memory usage percentage
        }
    
    @staticmethod
    def get_gpu_memory_info(device: int = 0) -> dict:
        """
        Get current GPU memory statistics.
        
        Args:
            device: GPU device index
            
        Returns:
            Dictionary with memory statistics in MB
        """
        if not torch.cuda.is_available():
            return {
                "allocated": 0,
                "reserved": 0,
                "free": 0,
                "total": 0,
                "allocated_pct": 0,
                "reserved_pct": 0
            }
        
        allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
        total = torch.cuda.get_device_properties(device).total_memory / 1024**2  # MB
        free = total - reserved
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "total": total,
            "allocated_pct": (allocated / total * 100) if total > 0 else 0,
            "reserved_pct": (reserved / total * 100) if total > 0 else 0
        }
    
    @staticmethod
    def log_memory(prefix: str = "", device: int = 0, level: int = logging.INFO):
        """
        Log current CPU RAM and GPU memory usage.
        
        Args:
            prefix: Prefix string for the log message
            device: GPU device index
            level: Logging level
        """
        cpu_info = MemoryMonitor.get_cpu_memory_info()
        
        # CPU RAM log
        cpu_msg = (
            f"{prefix}CPU RAM: "
            f"Process RSS: {cpu_info['rss']:.0f}MB, "
            f"Process VMS: {cpu_info['vms']:.0f}MB, "
            f"System: {cpu_info['used_pct']:.1f}% used, "
            f"Available: {cpu_info['available']:.0f}MB / {cpu_info['total']:.0f}MB"
        )
        logging.log(level, cpu_msg)
        
        # GPU memory log (only if available)
        if torch.cuda.is_available():
            gpu_info = MemoryMonitor.get_gpu_memory_info(device)
            gpu_msg = (
                f"{prefix}GPU Memory: "
                f"Allocated: {gpu_info['allocated']:.0f}MB ({gpu_info['allocated_pct']:.1f}%), "
                f"Reserved: {gpu_info['reserved']:.0f}MB ({gpu_info['reserved_pct']:.1f}%), "
                f"Free: {gpu_info['free']:.0f}MB / {gpu_info['total']:.0f}MB"
            )
            logging.log(level, gpu_msg)


def log_memory_usage(prefix: str = "", device: int = 0):
    """
    Decorator to log CPU RAM and GPU memory before and after function execution.
    
    Args:
        prefix: Prefix for log messages
        device: GPU device index
        
    Example:
        @log_memory_usage(prefix="[Video Encoding]")
        def encode_video(frames):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{prefix} {func.__name__}" if prefix else func.__name__
            
            # Log before
            MemoryMonitor.log_memory(f"[BEFORE {func_name}] ", device)
            
            try:
                result = func(*args, **kwargs)
                
                # Log after success
                MemoryMonitor.log_memory(f"[AFTER {func_name}] ", device)
                
                return result
            except Exception as e:
                # Log on error
                MemoryMonitor.log_memory(f"[ERROR in {func_name}] ", device, logging.ERROR)
                raise
        
        return wrapper
    return decorator


@contextmanager
def monitor_memory(description: str, device: int = 0):
    """
    Context manager to monitor CPU RAM and GPU memory during a code block.
    
    Args:
        description: Description of the operation being monitored
        device: GPU device index
        
    Example:
        with monitor_memory("Loading LLaVA model"):
            model = load_model()
    """
    # Memory before
    cpu_before = MemoryMonitor.get_cpu_memory_info()
    gpu_before = MemoryMonitor.get_gpu_memory_info(device) if torch.cuda.is_available() else None
    
    logging.info(
        f"[START {description}] "
        f"CPU RAM: {cpu_before['rss']:.0f}MB process, {cpu_before['available']:.0f}MB available | "
        + (f"GPU: {gpu_before['allocated']:.0f}MB allocated, {gpu_before['free']:.0f}MB free" 
           if gpu_before else "GPU: N/A")
    )
    
    try:
        yield
    finally:
        # Memory after
        cpu_after = MemoryMonitor.get_cpu_memory_info()
        gpu_after = MemoryMonitor.get_gpu_memory_info(device) if torch.cuda.is_available() else None
        
        cpu_delta = cpu_after['rss'] - cpu_before['rss']
        cpu_sign = "+" if cpu_delta >= 0 else ""
        
        gpu_msg = ""
        if gpu_after:
            gpu_delta = gpu_after['allocated'] - gpu_before['allocated']
            gpu_sign = "+" if gpu_delta >= 0 else ""
            gpu_msg = f" | GPU: {gpu_after['allocated']:.0f}MB allocated (Δ {gpu_sign}{gpu_delta:.0f}MB), {gpu_after['free']:.0f}MB free"
        
        logging.info(
            f"[END {description}] "
            f"CPU RAM: {cpu_after['rss']:.0f}MB process (Δ {cpu_sign}{cpu_delta:.0f}MB), "
            f"{cpu_after['available']:.0f}MB available"
            f"{gpu_msg}"
        )


def check_memory_available(required_cpu_mb: float = 0, required_gpu_mb: float = 0, device: int = 0) -> dict:
    """
    Check if enough CPU RAM and GPU memory is available.
    
    Args:
        required_cpu_mb: Required CPU RAM in MB
        required_gpu_mb: Required GPU memory in MB
        device: GPU device index
        
    Returns:
        Dictionary with 'cpu_ok' and 'gpu_ok' boolean values
    """
    result = {"cpu_ok": True, "gpu_ok": True}
    
    # Check CPU RAM
    if required_cpu_mb > 0:
        cpu_info = MemoryMonitor.get_cpu_memory_info()
        if cpu_info['available'] < required_cpu_mb:
            logging.warning(
                f"Insufficient CPU RAM: {cpu_info['available']:.0f}MB available, "
                f"{required_cpu_mb:.0f}MB required"
            )
            result["cpu_ok"] = False
    
    # Check GPU memory
    if required_gpu_mb > 0 and torch.cuda.is_available():
        gpu_info = MemoryMonitor.get_gpu_memory_info(device)
        if gpu_info['free'] < required_gpu_mb:
            logging.warning(
                f"Insufficient GPU memory: {gpu_info['free']:.0f}MB available, "
                f"{required_gpu_mb:.0f}MB required"
            )
            result["gpu_ok"] = False
    
    return result
