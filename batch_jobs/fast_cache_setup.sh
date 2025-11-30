#!/bin/bash
# Script per copiare la cache HuggingFace su /tmp locale (molto più veloce)

echo "================================================"
echo "Setting up fast local cache for HuggingFace"
echo "================================================"

# Directory cache originale (lenta, su NFS)
SLOW_CACHE="/cluster/scratch/tnanni/hub"

# Directory cache locale veloce (su SSD del nodo)
FAST_CACHE="/tmp/$USER/hf_cache"

# Crea directory locale
mkdir -p "$FAST_CACHE"

# Copia solo LLaVA (14GB) se non è già presente
if [ ! -d "$FAST_CACHE/models--llava-hf--LLaVA-NeXT-Video-7B-hf" ]; then
    echo "Copying LLaVA model to local cache (this will take 2-3 minutes)..."
    time rsync -ah --info=progress2 \
        "$SLOW_CACHE/models--llava-hf--LLaVA-NeXT-Video-7B-hf" \
        "$FAST_CACHE/"
    echo "✓ LLaVA copied to local cache"
else
    echo "✓ LLaVA already in local cache"
fi

# Copia altri modelli necessari (molto più piccoli)
for model in \
    "models--openai--clip-vit-base-patch32" \
    "models--openai--whisper-base" \
    "models--laion--clap-htsat-unfused" \
    "models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K" \
    "models--sentence-transformers--all-MiniLM-L6-v2"
do
    if [ -d "$SLOW_CACHE/$model" ] && [ ! -d "$FAST_CACHE/$model" ]; then
        echo "Copying $model..."
        rsync -ah "$SLOW_CACHE/$model" "$FAST_CACHE/"
    fi
done

# Imposta le variabili d'ambiente per usare la cache veloce
export HF_HOME="$FAST_CACHE"
export TRANSFORMERS_CACHE="$FAST_CACHE"

echo "================================================"
echo "Fast cache ready at: $FAST_CACHE"
echo "================================================"
echo ""
echo "Now LLaVA will load in 30-90 seconds instead of 27 minutes!"
echo ""
echo "Add these to your job script:"
echo "  export HF_HOME=\"$FAST_CACHE\""
echo "  export TRANSFORMERS_CACHE=\"$FAST_CACHE\""
