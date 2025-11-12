"""
Cache management utilities for fast model loading.
Copies HuggingFace models from slow network storage to fast local SSD.
"""
import os
import subprocess
import shutil
from typing import Dict, List


# Models larger than this threshold (in GB) will be copied to fast cache
LARGE_MODEL_THRESHOLD_GB = 5.0

# Mapping of model types to their HuggingFace model names
MODEL_REGISTRY = {
    "llava-video": "models--llava-hf--LLaVA-NeXT-Video-7B-hf",
    "qwen2-vl": "models--Qwen--Qwen2-VL-7B-Instruct",
    "xclip": "models--microsoft--xclip-base-patch16",
    "clip-vit-h": "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K",
    "clip-vit-base": "models--openai--clip-vit-base-patch32",
    "whisper-base": "models--openai--whisper-base",
    "whisper-large-v3": "models--openai--whisper-large-v3",
    "clap": "models--laion--clap-htsat-unfused",
    "sentence-transformers": "models--sentence-transformers--all-MiniLM-L6-v2",
    "blip": "models--Salesforce--blip-image-captioning-base",
}


def _get_model_size_gb(model_path: str) -> float:
    """Get size of model directory in GB."""
    if not os.path.exists(model_path):
        return 0.0
    
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    
    return total_size / (1024 ** 3)  # Convert to GB


def _detect_required_models() -> Dict[str, str]:
    """
    Detect which models will be used based on config.
    Returns dict of {model_type: model_dir_name}
    """
    from configuration.config import CONFIG
    
    required = {}
    
    # Video model
    video_model = CONFIG.indexing.video.model_name
    if video_model == "llava-video":
        required["llava-video"] = MODEL_REGISTRY["llava-video"]
    elif video_model == "qwen2-vl":
        required["qwen2-vl"] = MODEL_REGISTRY["qwen2-vl"]
    elif video_model == "xclip":
        required["xclip"] = MODEL_REGISTRY["xclip"]
    
    # CLIP for keyframes (always used)
    required["clip-vit-base"] = MODEL_REGISTRY["clip-vit-base"]
    
    # Audio models (always used)
    # Check which whisper model is configured
    asr_model_id = CONFIG.indexing.audio.asr_model_id
    if "whisper-large-v3" in asr_model_id:
        required["whisper-large-v3"] = MODEL_REGISTRY["whisper-large-v3"]
    else:
        required["whisper-base"] = MODEL_REGISTRY["whisper-base"]
    required["clap"] = MODEL_REGISTRY["clap"]
    
    # Text model (always used)
    required["sentence-transformers"] = MODEL_REGISTRY["sentence-transformers"]
    
    # Caption model
    captioner = CONFIG.indexing.caption.use_captioner
    if captioner == "captioner1":
        required["blip"] = MODEL_REGISTRY["blip"]
    elif captioner == "captioner2":
        # Captioner2 can use LLaVA or Qwen2-VL - check which one
        caption2_model = CONFIG.indexing.caption.caption2_model_id
        if "Qwen2-VL" in caption2_model and video_model != "qwen2-vl":
            # Qwen2-VL for caption but not for video - need to include it
            required["qwen2-vl"] = MODEL_REGISTRY["qwen2-vl"]
        elif "LLaVA" in caption2_model and video_model != "llava-video":
            # LLaVA for caption but not for video - need to include it
            required["llava-video"] = MODEL_REGISTRY["llava-video"]
        # Otherwise it's already included from video_model
    
    # Retrieval models (may differ from indexing)
    retrieval_video_model = CONFIG.retrieval.video_model_id
    if "CLIP-ViT-H-14" in retrieval_video_model:
        required["clip-vit-h"] = MODEL_REGISTRY["clip-vit-h"]
    
    return required


def setup_smart_cache(force_copy_all: bool = False, verbose: bool = True):
    """
    Smart cache setup that only copies large models to fast local SSD.
    
    Args:
        force_copy_all: If True, copy all models regardless of size
        verbose: If True, print detailed progress information
    
    Returns:
        Dictionary with original cache path for restoration
    """
    SLOW_CACHE = "/cluster/scratch/tnanni/hub"
    FAST_CACHE = f"/tmp/{os.environ.get('USER', 'tnanni')}/hf_cache"
    
    if verbose:
        print("\n" + "="*60)
        print("SMART CACHE SETUP")
        print("="*60)
    
    # Store original cache location
    original_hf_home = os.environ.get("HF_HOME", SLOW_CACHE)
    original_transformers_cache = os.environ.get("TRANSFORMERS_CACHE", SLOW_CACHE)
    
    os.makedirs(FAST_CACHE, exist_ok=True)
    
    # Detect required models
    required_models = _detect_required_models()
    
    if verbose:
        print(f"Detected {len(required_models)} required models")
    
    # Categorize models by size
    large_models = []
    small_models = []
    
    for model_type, model_dir in required_models.items():
        slow_path = os.path.join(SLOW_CACHE, model_dir)
        
        if not os.path.exists(slow_path):
            if verbose:
                print(f"⚠️  {model_type}: not found in cache, will download on demand")
            continue
        
        size_gb = _get_model_size_gb(slow_path)
        
        if force_copy_all or size_gb > LARGE_MODEL_THRESHOLD_GB:
            large_models.append((model_type, model_dir, size_gb))
        else:
            small_models.append((model_type, model_dir, size_gb))
    
    if verbose and large_models:
        print(f"\n📦 Large models (>{LARGE_MODEL_THRESHOLD_GB}GB) - will copy to fast cache:")
        for model_type, _, size in large_models:
            print(f"   • {model_type}: {size:.1f}GB")
    
    if verbose and small_models:
        print(f"\n💾 Small models (<{LARGE_MODEL_THRESHOLD_GB}GB) - will use from scratch:")
        for model_type, _, size in small_models:
            print(f"   • {model_type}: {size:.2f}GB")
    
    # Copy large models to fast cache
    for model_type, model_dir, size_gb in large_models:
        slow_path = os.path.join(SLOW_CACHE, model_dir)
        fast_path = os.path.join(FAST_CACHE, model_dir)
        
        if os.path.exists(fast_path):
            if verbose:
                print(f"✓ {model_type} already in fast cache")
            continue
        
        if verbose:
            print(f"\n📥 Copying {model_type} ({size_gb:.1f}GB) to fast cache...")
        
        try:
            # Use rsync with progress
            cmd = ["rsync", "-ah"]
            if verbose:
                cmd.append("--info=progress2")
            cmd.extend([slow_path + "/", fast_path + "/"])
            
            subprocess.run(cmd, check=True)
            
            if verbose:
                print(f"✓ {model_type} copied successfully")
        except subprocess.CalledProcessError:
            if verbose:
                print(f"⚠️  rsync failed for {model_type}, trying shutil...")
            shutil.copytree(slow_path, fast_path)
            if verbose:
                print(f"✓ {model_type} copied via shutil")
    
    # Create symlinks for small models (avoid unnecessary copying)
    for model_type, model_dir, _ in small_models:
        slow_path = os.path.join(SLOW_CACHE, model_dir)
        fast_path = os.path.join(FAST_CACHE, model_dir)
        
        if os.path.exists(fast_path):
            continue
        
        try:
            os.symlink(slow_path, fast_path)
            if verbose:
                print(f"🔗 Linked {model_type} (symlink to scratch)")
        except OSError:
            # Symlink failed, not critical
            pass
    
    # Set environment to use fast cache
    os.environ["HF_HOME"] = FAST_CACHE
    os.environ["TRANSFORMERS_CACHE"] = FAST_CACHE
    
    if verbose:
        print(f"\n✓ Cache environment set to: {FAST_CACHE}")
        print("="*60 + "\n")
    
    return {
        "original_hf_home": original_hf_home,
        "original_transformers_cache": original_transformers_cache,
        "fast_cache_path": FAST_CACHE
    }


def cleanup_smart_cache(cache_info: Dict[str, str] = None, verbose: bool = True):
    """
    Clean up fast cache and restore original environment.
    Only removes copied files, not symlinks or original cache.
    
    Args:
        cache_info: Dictionary returned by setup_smart_cache()
        verbose: If True, print progress information
    """
    FAST_CACHE = f"/tmp/{os.environ.get('USER', 'tnanni')}/hf_cache"
    
    if verbose:
        print("\n" + "="*60)
        print("CLEANING UP FAST CACHE")
        print("="*60)
    
    if os.path.exists(FAST_CACHE):
        if verbose:
            print(f"Removing: {FAST_CACHE}")
        
        # Remove only actual copied files, symlinks will be deleted automatically
        try:
            shutil.rmtree(FAST_CACHE)
            if verbose:
                print("✓ Fast cache cleaned up")
        except Exception as e:
            if verbose:
                print(f"⚠️  Error during cleanup: {e}")
    
    # Restore original environment if provided
    if cache_info:
        os.environ["HF_HOME"] = cache_info["original_hf_home"]
        os.environ["TRANSFORMERS_CACHE"] = cache_info["original_transformers_cache"]
        if verbose:
            print(f"✓ Restored original cache path")
    
    if verbose:
        print("="*60 + "\n")


# Legacy functions for backward compatibility
def setup_fast_cache():
    """
    Setup fast local cache for LLaVA and other HuggingFace models.
    
    Copies models from /cluster/scratch/tnanni/hub (slow NFS) to /tmp/$USER/hf_cache (fast local SSD).
    This reduces LLaVA loading time from 27 minutes to ~60 seconds.
    
    Only runs if using llava-video model in config.
    Sets HF_HOME and TRANSFORMERS_CACHE environment variables.
    """
    from configuration.config import CONFIG
    
    # Only setup cache if using llava-video
    if CONFIG.indexing.video.model_name != "llava-video":
        print("Not using LLaVA, skipping cache setup")
        return
    
    print("\n" + "="*60)
    print("SETTING UP FAST LOCAL CACHE FOR LLAVA")
    print("="*60)
    
    SLOW_CACHE = "/cluster/scratch/tnanni/hub"
    FAST_CACHE = f"/tmp/{os.environ.get('USER', 'tnanni')}/hf_cache"
    
    os.makedirs(FAST_CACHE, exist_ok=True)
    
    # Copy LLaVA to local SSD if not already there
    llava_model = "models--llava-hf--LLaVA-NeXT-Video-7B-hf"
    fast_llava_path = os.path.join(FAST_CACHE, llava_model)
    slow_llava_path = os.path.join(SLOW_CACHE, llava_model)
    
    if not os.path.exists(fast_llava_path):
        print(f"Copying LLaVA (14GB) from scratch to local SSD...")
        print(f"Source: {slow_llava_path}")
        print(f"Dest:   {fast_llava_path}")
        print("This will take 2-3 minutes...")
        
        # Use rsync for faster copying with progress
        try:
            subprocess.run([
                "rsync", "-ah", "--info=progress2",
                slow_llava_path + "/",
                fast_llava_path + "/"
            ], check=True)
            print("✓ LLaVA copied to local cache")
        except subprocess.CalledProcessError as e:
            print(f"Warning: rsync failed, trying shutil.copytree...")
            shutil.copytree(slow_llava_path, fast_llava_path)
            print("✓ LLaVA copied to local cache (via shutil)")
    else:
        print(f"✓ LLaVA already in local cache: {fast_llava_path}")
    
    # Copy other smaller models
    other_models = [
        "models--openai--clip-vit-base-patch32",
        "models--openai--whisper-base",
        "models--laion--clap-htsat-unfused",
        "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K",
        "models--sentence-transformers--all-MiniLM-L6-v2"
    ]
    
    for model in other_models:
        slow_path = os.path.join(SLOW_CACHE, model)
        fast_path = os.path.join(FAST_CACHE, model)
        
        if os.path.exists(slow_path) and not os.path.exists(fast_path):
            print(f"Copying {model}...")
            try:
                subprocess.run(["rsync", "-ah", slow_path + "/", fast_path + "/"], 
                             check=True, stdout=subprocess.DEVNULL)
            except:
                shutil.copytree(slow_path, fast_path)
    
    # Set environment variables
    os.environ["HF_HOME"] = FAST_CACHE
    os.environ["TRANSFORMERS_CACHE"] = FAST_CACHE
    
    print(f"\n✓ Fast cache ready at: {FAST_CACHE}")
    print(f"✓ HF_HOME set to: {FAST_CACHE}")
    print("="*60 + "\n")


def cleanup_fast_cache():
    """
    Remove local cache after job completes.
    
    Frees up /tmp space by removing the fast local cache.
    Models remain in /cluster/scratch/tnanni/hub for future use.
    """
    from configuration.config import CONFIG
    
    if CONFIG.indexing.video.model_name != "llava-video":
        return
        
    FAST_CACHE = f"/tmp/{os.environ.get('USER', 'tnanni')}/hf_cache"
    
    if os.path.exists(FAST_CACHE):
        print("\n" + "="*60)
        print("CLEANING UP LOCAL CACHE")
        print("="*60)
        print(f"Removing: {FAST_CACHE}")
        shutil.rmtree(FAST_CACHE)
        print("✓ Local cache cleaned up")
        print("="*60 + "\n")
