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
    "sentence-transformers": "models--sentence-transformers--EmbeddingGemma-300M",
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
    # Create convenience bidirectional links (do not replace real model dirs)
    created_links: List[str] = []
    try:
        scratch_to_fast = os.path.join(SLOW_CACHE, "fast_hf_cache")
        if not os.path.exists(scratch_to_fast):
            os.symlink(FAST_CACHE, scratch_to_fast)
            created_links.append(scratch_to_fast)
        fast_to_scratch = os.path.join(FAST_CACHE, "scratch_hub")
        if not os.path.exists(fast_to_scratch):
            os.symlink(SLOW_CACHE, fast_to_scratch)
            created_links.append(fast_to_scratch)
        if verbose and created_links:
            print(f"🔗 Created convenience links: {created_links}")
    except OSError:
        # Not critical; continue
        created_links = []
    
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
    
    moved_models: List[str] = []

    # Copy/move large models to fast cache and create symlink from scratch -> fast
    for model_type, model_dir, size_gb in large_models:
        slow_path = os.path.join(SLOW_CACHE, model_dir)
        fast_path = os.path.join(FAST_CACHE, model_dir)

        if os.path.exists(fast_path):
            if verbose:
                print(f"✓ {model_type} already in fast cache")
            continue

        if not os.path.exists(slow_path):
            if verbose:
                print(f"⚠️  {model_type}: not found in scratch during move")
            continue

        if verbose:
            print(f"\n📥 Moving {model_type} ({size_gb:.1f}GB) to fast cache...")

        # Try fast atomic move first
        try:
            os.rename(slow_path, fast_path)
            moved_models.append(model_dir)
            if verbose:
                print(f"✓ {model_type} moved (rename) to fast cache")
        except OSError:
            # Different filesystems: fallback to rsync copy then remove
            try:
                cmd = ["rsync", "-a", slow_path + "/", fast_path + "/"]
                subprocess.run(cmd, check=True)
                shutil.rmtree(slow_path)
                moved_models.append(model_dir)
                if verbose:
                    print(f"✓ {model_type} copied (rsync) and removed from scratch")
            except subprocess.CalledProcessError:
                if verbose:
                    print(f"⚠️  rsync failed for {model_type}, trying shutil.copytree...")
                shutil.copytree(slow_path, fast_path)
                shutil.rmtree(slow_path)
                moved_models.append(model_dir)
                if verbose:
                    print(f"✓ {model_type} copied via shutil and removed from scratch")

        # Create symlink at the original scratch location pointing to fast cache
        try:
            if os.path.exists(slow_path):
                # should not exist, but if it does, remove to replace with symlink
                if os.path.islink(slow_path):
                    os.unlink(slow_path)
                else:
                    shutil.rmtree(slow_path)
            os.symlink(fast_path, slow_path)
            if verbose:
                print(f"🔗 Created symlink: {slow_path} -> {fast_path}")
        except OSError:
            if verbose:
                print(f"⚠️  Could not create symlink for {model_type}")
    
    # For small models, prefer to leave them on scratch but expose them via symlink
    for model_type, model_dir, _ in small_models:
        slow_path = os.path.join(SLOW_CACHE, model_dir)
        fast_path = os.path.join(FAST_CACHE, model_dir)

        if os.path.exists(fast_path):
            continue

        if os.path.exists(slow_path):
            # Try to move small model to fast cache and symlink scratch -> fast
            try:
                os.rename(slow_path, fast_path)
                moved_models.append(model_dir)
                if verbose:
                    print(f"✓ {model_type} moved (small) to fast cache")
            except OSError:
                try:
                    subprocess.run(["rsync", "-a", slow_path + "/", fast_path + "/"], check=True)
                    shutil.rmtree(slow_path)
                    moved_models.append(model_dir)
                    if verbose:
                        print(f"✓ {model_type} copied (rsync) and removed from scratch")
                except subprocess.CalledProcessError:
                    # leave on scratch and create symlink fast->slow as fallback
                    try:
                        os.symlink(slow_path, fast_path)
                        if verbose:
                            print(f"🔗 Linked {model_type} (fast -> scratch fallback)")
                    except OSError:
                        if verbose:
                            print(f"⚠️  Could not link {model_type}")
                    continue

            # create symlink at scratch pointing to fast
            try:
                if os.path.exists(slow_path):
                    if os.path.islink(slow_path):
                        os.unlink(slow_path)
                    else:
                        shutil.rmtree(slow_path)
                os.symlink(fast_path, slow_path)
                if verbose:
                    print(f"🔗 Linked {model_type} (scratch -> fast)")
            except OSError:
                if verbose:
                    print(f"⚠️  Could not create scratch->fast symlink for {model_type}")
    
    # Set environment to use fast cache
    os.environ["HF_HOME"] = FAST_CACHE
    os.environ["TRANSFORMERS_CACHE"] = FAST_CACHE
    
    if verbose:
        print(f"\n✓ Cache environment set to: {FAST_CACHE}")
        print("="*60 + "\n")
    
    return {
        "original_hf_home": original_hf_home,
        "original_transformers_cache": original_transformers_cache,
        "fast_cache_path": FAST_CACHE,
        "moved_models": moved_models,
        "slow_cache_path": SLOW_CACHE,
        "created_links": created_links,
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
    
    if cache_info is None:
        cache_info = {}

    moved_models = cache_info.get("moved_models", [])
    slow_cache = cache_info.get("slow_cache_path", "/cluster/scratch/tnanni/hub")
    created_links = cache_info.get("created_links", [])

    if os.path.exists(FAST_CACHE):
        if verbose:
            print(f"Restoring moved models and removing: {FAST_CACHE}")

        # Restore moved models back to scratch by replacing symlinks
        for model_dir in moved_models:
            slow_path = os.path.join(slow_cache, model_dir)
            fast_path = os.path.join(FAST_CACHE, model_dir)

            try:
                # If slow_path is a symlink pointing to fast, remove it
                if os.path.islink(slow_path):
                    try:
                        os.unlink(slow_path)
                    except Exception:
                        pass

                # If fast_path exists and slow_path does not, move it back
                if os.path.exists(fast_path) and not os.path.exists(slow_path):
                    try:
                        os.rename(fast_path, slow_path)
                    except OSError:
                        # Different filesystems: rsync then remove
                        try:
                            subprocess.run(["rsync", "-a", fast_path + "/", slow_path + "/"], check=True)
                            shutil.rmtree(fast_path)
                        except subprocess.CalledProcessError:
                            if verbose:
                                print(f"⚠️  Could not rsync {model_dir} back to scratch")
            except Exception as e:
                if verbose:
                    print(f"⚠️  Error restoring {model_dir}: {e}")

        # Finally remove the fast cache directory
        # Remove any convenience links recorded
        for link in created_links:
            try:
                if os.path.islink(link):
                    os.unlink(link)
                    if verbose:
                        print(f"✓ Removed link: {link}")
            except Exception:
                pass

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
        "models--sentence-transformers--EmbeddingGemma-300M"
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
