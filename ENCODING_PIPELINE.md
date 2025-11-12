# 🎬 EgoRAG Encoding Pipeline

## Panoramica

Sistema di encoding multimodale gerarchico per video lunghi. I video vengono divisi in scene e ogni scena viene encodata separatamente attraverso 4 modalità: video, caption, audio e testo.

---

## 📊 STAGE 1: VIDEO ENCODING

### Opzione A: XCLIP

**Configurazione**: `model_name: "xclip"`

**Pipeline**:

1. **Estrazione frame**: Estrae TUTTI i frame della scena usando `VideoReader`
2. **Feature extraction**: CLIP Vision embedda tutti i frame
3. **Clustering**: K-means clustering per identificare 8 cluster
4. **Selezione**: Seleziona il frame più vicino al centroide di ogni cluster
5. **Encoding**: XCLIP processa gli 8 frame selezionati
6. **Output**:
   - Embedding: **768-dim** tensor
   - Keyframes: Array numpy dei primi 8 frame (per captioner1)

**Caratteristiche**:

- ✅ Veloce ed efficiente
- ✅ Seleziona frame rappresentativi tramite clustering
- ❌ Non vede la continuità temporale completa

---

### Opzione B: Qwen2-VL (ATTUALE)

**Configurazione**: `model_name: "qwen2-vl"`

**Pipeline**:

1. **Creazione clip**: `ffmpeg` crea clip temporanea della scena (sempre re-encoded con libx264)
   ```bash
   ffmpeg -ss START -to END -i VIDEO -c:v libx264 -preset fast -crf 23 -an OUTPUT.mp4
   ```
2. **Passaggio video diretto**: Il video clip viene passato direttamente a Qwen2-VL tramite path
   ```python
   {"type": "video", "video": clip_path, "max_pixels": 360*420, "fps": 1.0}
   ```
3. **Processing Qwen2-VL**:
   - Qwen2-VL gestisce internamente il sampling dei frame
   - Applica chat template con prompt "Describe this video."
   - Processa vision info con `process_vision_info()`
   - Il processor carica e processa automaticamente il video
4. **Embedding extraction**:
   - Accede alla vision tower: `model.visual()`
   - Estrae il **primo token**: `vision_outputs[0, 0, :]` (NO pooling)
5. **Output**:
   - Embedding: **1536-dim** tensor (bfloat16 → float32)
   - Keyframes: 8 frame campionati uniformemente (per captioner1 se necessario)
6. **Cleanup**: Elimina la clip temporanea e libera memoria

**Caratteristiche**:

- ✅ Vede l'INTERA scena come sequenza continua
- ✅ Embedding più ricco e contestuale (1536-dim vs 768-dim)
- ✅ Usa il primo token senza pooling (preserva informazioni)
- ✅ Qwen2-VL può gestire nativamente video >20 minuti
- ✅ NO frame extraction necessaria - gestione interna ottimizzata
- ⚠️ Re-encoding con libx264 richiesto per evitare frame corrotti

**Configurazione modello**:

```python
torch_dtype=torch.bfloat16
device_map="cuda"
min_pixels=256*28*28
max_pixels=1280*28*28
```

---

## 💬 STAGE 2: CAPTION GENERATION

### Opzione A: Captioner1 (BLIP)

**Configurazione**: `use_captioner: "captioner1"`

**Pipeline**:

1. **Input**: Usa i keyframes estratti durante il video encoding
2. **Clustering**: Seleziona max 5 frame rappresentativi
3. **Captioning**: BLIP genera caption per ogni frame
4. **Aggregazione**: Combina le caption in una descrizione unica

**Caratteristiche**:

- ✅ Veloce - usa frame già estratti
- ✅ Leggero in memoria
- ❌ Caption meno contestuale (frame singoli)

---

### Opzione B: Captioner2 (Qwen2-VL) (ATTUALE)

**Configurazione**: `use_captioner: "captioner2"`

**Pipeline**:

1. **Creazione clip**: `ffmpeg` crea clip temporanea della scena (sempre re-encoded con libx264)
2. **Passaggio video diretto**: Il video clip viene passato direttamente a Qwen2-VL tramite path
   ```python
   {"type": "video", "video": clip_path, "max_pixels": 360*420, "fps": 1.0}
   ```
3. **Generation Qwen2-VL**:
   - Qwen2-VL gestisce internamente il sampling dei frame
   - Crea messaggio con video + prompt: "Describe this scene briefly."
   - Applica chat template
   - Processa vision info
   - Genera caption con `model.generate(max_new_tokens=64)`
4. **Post-processing**: Decodifica e pulisce il testo generato
5. **Cleanup**: Elimina clip temporanea e libera memoria

**Caratteristiche**:

- ✅ Caption contestuale basata sull'INTERA scena
- ✅ Descrizioni più accurate e dettagliate
- ✅ Può descrivere azioni temporali e transizioni
- ✅ Qwen2-VL gestisce video nativamente - NO frame extraction necessaria
- ❌ Più lento
- 🔄 **Condivisione modello**: Se video e caption usano entrambi Qwen2-VL, il modello viene caricato UNA sola volta e condiviso

---

## 🔊 STAGE 3: AUDIO ENCODING

**Pipeline**:

1. **Estrazione audio**: Estrae segmento audio dalla scena
2. **Transcription**:
   - Whisper (large-v3) transcrive l'audio
   - Opzionale: faster-whisper per velocità
3. **Audio embedding**: CLAP genera embedding audio
4. **Event detection** (opzionale): AST identifica eventi sonori
5. **Diarization** (opzionale): Identifica speaker diversi

**Output**:

- Audio embedding (CLAP)
- Transcript testuale
- Eventi audio (se abilitato)
- Speaker labels (se abilitato)

**Gestione video senza audio**:

- Rileva automaticamente video senza traccia audio
- Salta l'encoding audio per tutte le scene
- Imposta `audio_embedding = None`, `transcript = ""`

---

## 📝 STAGE 4: TEXT ENCODING

**Pipeline**:

1. **Combinazione testi**:
   ```
   full_text = f"Transcript: {transcript}. Visuals: {caption}"
   ```
2. **Encoding**:
   - Sentence Transformers (all-MiniLM-L6-v2)
   - Genera embedding per il testo completo
   - Genera embedding separato per la sola caption
3. **Output**:
   - Text embedding: combinazione transcript + caption
   - Caption embedding: solo caption visiva

---

## 🎯 AGGREGAZIONE EMBEDDINGS

Dopo l'encoding di tutte le scene:

**Per ogni scena** (`scene_embeddings`):

```python
{
    "video": Tensor[1536],      # o 768 per XCLIP
    "audio": Tensor[512],        # o None se no audio
    "text": Tensor[384],         # transcript + caption
    "caption": Tensor[384],      # solo caption
    "transcript": str,           # trascrizione audio
    "caption_text": str          # descrizione visiva
}
```

**Livello video globale** (`global_embeddings`):

- Media di tutti gli embedding delle scene
- Rappresentazione dell'intero video

---

## 🔄 OTTIMIZZAZIONI

### Condivisione Modelli

Quando `video_model_name == "qwen2-vl"` E `use_captioner == "captioner2"`:

1. Carica Qwen2-VL una sola volta per il video encoding
2. Condivide il modello con il captioner (solo carica il processor)
3. Unload dopo caption generation
4. **Risparmio memoria**: ~14GB VRAM (evita doppio caricamento)

### Gestione Memoria

- **Stage-by-stage loading**: Un solo modello caricato alla volta
- **Immediate cleanup**: `del model`, `torch.cuda.empty_cache()`, `gc.collect()`
- **Temporary files**: Automaticamente eliminati dopo l'uso
- **Keyframes deletion**: Eliminati dopo caption generation per ridurre dimensione pickle

### Parallelizzazione

- `ThreadPoolExecutor` con `max_workers=2` di default
- Encoding di scene multiple in parallelo
- Lock per `VideoReader` quando necessario

---

## 📦 OUTPUT FINALE

**VideoDataset** salvato come `.pkl`:

```python
{
    "video_datapoints": [
        {
            "video_name": str,
            "video_path": str,
            "scenes": Dict[str, Scene],
            "scene_embeddings": Dict[str, Dict],
            "global_embeddings": Dict[str, Tensor],
            "has_audio": bool
        },
        ...
    ],
    "encoded": True
}
```

---

## ⚙️ CONFIGURAZIONE

**File**: `configuration/config.yaml`

```yaml
indexing:
  video:
    model_name: "qwen2-vl" # o "xclip"
    qwen2_vl_id: "Qwen/Qwen2-VL-7B-Instruct"
    xclip_id: "microsoft/xclip-base-patch16"
    clip_id: "openai/clip-vit-base-patch32"

  caption:
    use_captioner: "captioner2" # o "captioner1"
    caption2_model_id: "Qwen/Qwen2-VL-7B-Instruct"
    caption_model_id: "Salesforce/blip-image-captioning-base"

  audio:
    asr_model_id: "openai/whisper-large-v3"
    audio_model_id: "laion/clap-htsat-unfused"
    use_faster_whisper: true
    use_audio_events: true

  text:
    text_model_id: "all-MiniLM-L6-v2"
```

---

## 🚀 UTILIZZO

```python
from data.video_dataset import VideoDataset
from indexing.multimodal_encoder import MultiModalEncoder

# Carica dataset
dataset = VideoDataset.from_json("path/to/videos.json")

# Inizializza encoder
encoder = MultiModalEncoder(
    video_dataset=dataset,
    device="cuda",
    max_workers=2
)

# Encoding completo
encoded_dataset = encoder.encode_videos()

# Salva
encoded_dataset.save("encoded_videos.pkl")
```

---

## 📊 DIMENSIONI EMBEDDINGS

| Modalità         | Modello                      | Dimensione |
| ---------------- | ---------------------------- | ---------- |
| Video (XCLIP)    | microsoft/xclip-base-patch16 | 768        |
| Video (Qwen2-VL) | Qwen/Qwen2-VL-7B-Instruct    | 1536       |
| Audio            | laion/clap-htsat-unfused     | 512        |
| Text             | all-MiniLM-L6-v2             | 384        |
| Caption          | all-MiniLM-L6-v2             | 384        |

---

## 🔧 DIPENDENZE

```bash
# Core
pip install torch transformers
pip install decord pillow numpy

# Qwen2-VL specifico
pip install qwen-vl-utils
pip install flash-attn --no-build-isolation

# Audio
pip install faster-whisper  # opzionale, per velocità

# Altri
pip install sentence-transformers
pip install scikit-learn  # per clustering XCLIP
```

---

## 📝 NOTE

1. **Qwen2-VL richiede**: Flash Attention 2, ~14GB VRAM, compatibilità GPU Ampere/Ada
2. **Clip temporanee**: Create in `/tmp/` con prefisso `qwen2vl_scene_*` o `qwen2vl_caption_*`
3. **Formato video**: MP4 consigliato per compatibilità `ffmpeg -c copy`
4. **Scene detection**: Usa PySceneDetect di default, può usare scene pre-esistenti
