import sys
import logging
import os
from video_dataset import VideoDataset
from video_encoder import MultiModalEncoder
import sys
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import numpy as np
from utils.logging_formatter import LevelAwareFormatter
from transformers import logging as hf_logging

handler = logging.StreamHandler()
handler.setFormatter(LevelAwareFormatter())
logging.basicConfig(
    level=logging.INFO,
    handlers=[handler],
)
hf_logging.set_verbosity_error()
hf_logging.disable_progress_bar()


def launch_experiment(num: int = 1):
    if num == 1:
        experiment_1()
    elif num == 2:
        experiment_2()


def experiment_1():
    """
    Launches a check with 2 videos to see if everything works end-to-end
    """
    logging.info("Starting MultiModalEncoder test run...")
    
    # Assicurati che le directory di supporto esistano
    if not os.path.exists("video_dataset.py") or not os.path.exists("utils/logging_formatter.py"):
         print("Assicurati che i file 'video_dataset.py' e 'utils/logging_formatter.py' siano presenti.")
         sys.exit(1)

    data_dir = "../../../data" # FIXME: aggiorna il percorso alla directory dei dati
    if not os.path.exists(data_dir):
        logging.error(f"Directory {os.path.abspath(data_dir)} non trovata.")
        sys.exit(1)

    video_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        and "animal" not in f.lower()  # Esempio di filtro per escludere certi video
        and "ai" not in f.lower()
        and "prova" not in f.lower() 
    ]

    if not video_files:
        logging.error(f"Nessun video trovato in {data_dir}.")
        sys.exit(1)

    logging.info(f"Trovati {len(video_files)} video:\n" + "\n".join(f" - {f}" for f in video_files))

    dataset = VideoDataset(video_files)
    logging.info(f"Creato VideoDataset con {len(dataset)} elementi.")

    encoder = MultiModalEncoder(video_dataset=dataset, device="cuda")
    encoder.load_models()

    encoded_dataset = encoder.encode_videos()

    if not encoded_dataset.video_datapoints:
        logging.warning("Nessun video è stato processato.")
        sys.exit(0)
        
    first_dp = encoded_dataset.video_datapoints[0]
    if first_dp:
        logging.info("\n=== RISULTATI PRIMO VIDEO ===")
        logging.info(f"Video path: {first_dp.video_path}")
        logging.info(f"Numero scene: {len(first_dp.scenes)}")

        if first_dp.global_embeddings:
            logging.info(f"Chiavi global embeddings: {list(first_dp.global_embeddings.keys())}")
        
        if first_dp.scene_embeddings:
            scene_keys = list(first_dp.scene_embeddings.keys())
            logging.info(f"Chiavi scene embeddings: {scene_keys[:3]} ...")
            
            if scene_keys:
                first_scene_key = scene_keys[0]
                first_scene = first_dp.scene_embeddings[first_scene_key]
                logging.info(f"--- Esempio scena: {first_scene_key} ---")
                
                # --- INIZIO MODIFICA ---
                # Controlla se l'embedding esiste prima di stamparne la forma
                video_shape = first_scene['video'].shape if first_scene['video'] is not None else "None"
                audio_shape = first_scene['audio'].shape if first_scene['audio'] is not None else "None"
                text_shape = first_scene['text'].shape if first_scene['text'] is not None else "None"
                
                logging.info(f"Video emb shape: {video_shape}")
                logging.info(f"Audio emb shape: {audio_shape}")
                logging.info(f"Text emb shape:  {text_shape}")
                # --- FINE MODIFICA ---
                
                logging.info(f"Transcript: '{first_scene['transcript'][:80]}...'")

    logging.info("Encoding completato con successo!")


def experiment_2():
    """
    Creates embeddings of some sentences and compares the similarity with video embeddings
    """

    logging.info("🚀 Starting Multimodal Retrieval Test Run (FULL)...")

    # === 1️⃣ Setup dataset e encoder ===
    data_dir = "../../../data"
    if not os.path.exists(data_dir):
        logging.error(f"❌ Directory {os.path.abspath(data_dir)} non trovata.")
        sys.exit(1)

    video_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".mp4", ".mov", ".mkv", ".avi"))
        and "animal" not in f.lower()  # Esempio di filtro per escludere certi video
        and "ai" not in f.lower()
    ]

    dataset = VideoDataset(video_files)
    encoder = MultiModalEncoder(video_dataset=dataset, device="cuda")
    encoder.load_models()
    encoded_dataset = encoder.encode_videos()

    if not encoded_dataset.video_datapoints:
        logging.error("❌ Nessun video processato.")
        sys.exit(1)

    logging.info(f"✅ Encoded {len(encoded_dataset)} videos.")

    # === 2️⃣ Definizione delle queries ===
    queries = [
        "Who scored the winning goal in Bologna’s match?",
        "Who won between Sinner and Alcaraz?",
        "Can you hear the crowd cheering?",
        "Is the commentator describing an amazing point?",
        "Did Sinner serve an ace during the match?",
        "Did Pisa make a comeback?",
        "Are there scenes of celebration or applause?",
        "Can you hear the sound of the racket?",
        "Is the atmosphere tense during the final rallies?",
        "Does the commentary rhythm suggest an important moment?",
        "Are there changes in pace or intensity in the commentary?",
        "Does it look like a break or preparation moment in the game?",
        "Does the video show strong emotions from the players or crowd?",
    ]


    # === 3️⃣ Funzioni helper ===
    def cosine(a, b):
        # Aggiunto epsilon per evitare divisione per zero
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        if a_norm == 0 or b_norm == 0:
            return 0.0
        a = a / a_norm
        b = b / b_norm
        return np.dot(a, b)

    def match_dim(x, target_dim):
        if len(x) > target_dim:
            return x[:target_dim]
        elif len(x) < target_dim:
            return np.pad(x, (0, target_dim - len(x)))
        return x

    # === 4️⃣ Embeddiamo tutte le queries ===
    text_embedder = encoder.text_embedder
    query_embs = text_embedder.encode(queries, convert_to_numpy=True)
    logging.info(f"[INFO] Query embeddings shape: {query_embs.shape}") # (13, 384)

    # === 5️⃣ Otteniamo tutti gli embeddings video globali ===
    all_embeddings = []
    for dp in encoded_dataset.video_datapoints:
        
        # --- INIZIO MODIFICA ---
        # Gestiamo il caso in cui i global_embeddings siano None
        video_emb = dp.global_embeddings["video"].cpu().numpy() if dp.global_embeddings["video"] is not None else None
        audio_emb = dp.global_embeddings["audio"].cpu().numpy() if dp.global_embeddings["audio"] is not None else None
        text_emb = dp.global_embeddings["text"].cpu().numpy() if dp.global_embeddings["text"] is not None else None
        
        all_embeddings.append({
            "video_name": dp.video_name,
            "video": video_emb,
            "audio": audio_emb,
            "text": text_emb,
        })
        # --- FINE MODIFICA ---

    # === 6️⃣ Definiamo tutte le fusioni possibili ===
    modalities = ["video", "audio", "text"]
    fusion_modes = []
    for r in range(1, len(modalities) + 1):
        for combo in itertools.combinations(modalities, r):
            fusion_modes.append(combo)

    logging.info(f"[INFO] Fusion modes: {fusion_modes}")

    # === 7️⃣ Calcoliamo similarità per ogni query e video ===
    for qi, qtext in enumerate(queries):
        logging.info("\n" + "=" * 90)
        logging.info(f"🔎 QUERY {qi+1}/{len(queries)}: '{qtext}'")
        qemb = query_embs[qi] # Dimensione (384,)

        results = []

        for dp in all_embeddings:
            video_name = dp["video_name"]

            # prepariamo query in dimensioni compatibili
            qv = match_dim(qemb, 768) # Pad 384 -> 768
            qa = match_dim(qemb, 512) # Pad 384 -> 512
            qt = match_dim(qemb, 384) # No change

            sims = {}

            for fusion in fusion_modes:
                fused_target = []
                fused_query = []
                
                # --- INIZIO MODIFICA ---
                skip_fusion = False # Flag per saltare la fusione se mancano dati
                
                for mod in fusion:
                    target_emb = dp[mod]
                    
                    # Se l'embedding per questa modalità è None, saltiamo
                    if target_emb is None:
                        skip_fusion = True
                        break 
                        
                    qtmp = {
                        "video": qv,
                        "audio": qa,
                        "text": qt
                    }[mod]

                    # Aggiunto epsilon per evitare divisione per zero
                    norm_target = np.linalg.norm(target_emb)
                    norm_q = np.linalg.norm(qtmp)

                    if norm_target == 0 or norm_q == 0:
                        # Se un vettore è nullo, la sim è 0, ma per sicurezza saltiamo
                        skip_fusion = True 
                        break

                    target_emb = target_emb / norm_target
                    qtmp = qtmp / norm_q

                    fused_target.append(target_emb)
                    fused_query.append(qtmp)

                fusion_name = "+".join(fusion)
                if skip_fusion:
                    sims[fusion_name] = -1.0 # Assegna un punteggio basso per ranking
                else:
                    fused_target = np.concatenate(fused_target)
                    fused_query = np.concatenate(fused_query)
                    sims[fusion_name] = cosine(fused_target, fused_query)
                # --- FINE MODIFICA ---

            results.append({
                "video_name": video_name,
                **sims
            })

        # === 8️⃣ Ordiniamo i risultati per ogni tipo di fusione ===
        logging.info("\n📊 RISULTATI per query:")
        for fusion in fusion_modes:
            fusion_name = "+".join(fusion)
            
            # Filtra i risultati che non sono stati saltati (punteggio > -1.0)
            valid_results = [r for r in results if r[fusion_name] > -1.0]
            
            if not valid_results:
                logging.info(f"  [{fusion_name.upper():>13}] → Nessun risultato valido (embedding mancanti).")
                continue

            sorted_res = sorted(valid_results, key=lambda r: r[fusion_name], reverse=True)
            best = sorted_res[0]
            logging.info(f"  [{fusion_name.upper():>13}] → top: {best['video_name']}  (sim={best[fusion_name]:.3f})")

        logging.info("=" * 90)

if __name__ == "__main__":
    launch_experiment(num=1)