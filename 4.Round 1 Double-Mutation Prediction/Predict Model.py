import os
import csv
import random
import time
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.ensemble import GradientBoostingRegressor
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations, product
from tqdm import tqdm


# ===== Configuration =====
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATHS = ["xgb_final_all_data_model.pkl"]  # Paths to trained models
OUTPUT_CSV = "filtered_mutants.csv"
PRED_THRESHOLD = 0.55

# ===== Hardware optimization (optimized for RTX 4060 with 8 GB VRAM) =====
PRED_BATCH_SIZE = 64          # Reduced batch size to fit 8 GB VRAM
CPU_WORKERS = 16              # Use 16 CPU cores on Windows
USE_FP16 = True               # Enable half-precision inference
PROGRESS_INTERVAL = 1000      # Show progress every 1000 sequences

# Original sequence and mutation positions
original_seq = (
    "MTSMQKVFAGYAARQAVLEASNDPFAKGIAWIEGEYVPLAEARIPLLDQGFMRSDLTYDVPSV"
    "WDGRFFRLDDHLTRLEVSCDKLRLKVPLPRDEVKRILVDMVAKSGIRDAHVCLIVTRGLKGVRGT"
    "KPEDIVNRLYMFIQPYVWVMEPEMQHTGGSAIVARTVRRVPPGAIDPTIKNLQWGDLVRGMFEAS"
    "DRGATYPFLTDGDAHLTEGSGFNIVLVKDGVLYTPDRGVLQGITRKSVFDAARACGIEVRLEFVP"
    "VELAYNADEIFMSTTAGGIMPITSLDGKPVNGGQVGPVTKAIWDTYWAMHYDPVYSFQIDYDGVQ"
    "KSGVNGLGKL"
)
positions_1b = [72, 113, 115, 126, 145, 147, 199, 272, 312, 325]
positions = [p - 1 for p in positions_1b]
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")


# ===== Sequence validation =====
def validate_double_mutation(original_seq, pos1, aa1, pos2, aa2):
    orig_aa1 = original_seq[pos1]
    orig_aa2 = original_seq[pos2]

    mutations = []
    mutation_count = 0

    if aa1 != orig_aa1:
        mutations.append(f"{orig_aa1}{pos1 + 1}{aa1}")
        mutation_count += 1

    if aa2 != orig_aa2:
        mutations.append(f"{orig_aa2}{pos2 + 1}{aa2}")
        mutation_count += 1

    is_valid = (mutation_count == 2)
    return is_valid, mutations, mutation_count


def generate_mutant_sequence(original_seq, pos1, aa1, pos2, aa2):
    """Generate a mutant sequence"""
    seq_list = list(original_seq)
    seq_list[pos1] = aa1
    seq_list[pos2] = aa2
    return "".join(seq_list)


# ===== Generate valid pairwise mutants =====
def generate_valid_pairwise_mutants():
    """Generate all valid double mutants (with caching support)"""
    seq_cache = "valid_mutant_sequences.npy"
    combo_cache = "valid_mutant_combos.npy"
    info_cache = "valid_mutant_info.npy"

    if os.path.exists(seq_cache) and os.path.exists(combo_cache) and os.path.exists(info_cache):
        print("Loading cached mutant sequences...")
        seqs = np.load(seq_cache, allow_pickle=True).tolist()
        combos = np.load(combo_cache, allow_pickle=True).tolist()
        infos = np.load(info_cache, allow_pickle=True).tolist()
    else:
        print("Generating valid double-mutant combinations...")
        seqs, combos, infos = [], [], []

        total_possible = len(list(combinations(positions, 2))) * (len(amino_acids) ** 2)
        valid_count = 0
        invalid_count = 0

        with tqdm(total=total_possible, desc="Generating mutants", unit="comb") as pbar:
            for pos1, pos2 in combinations(positions, 2):
                for aa1, aa2 in product(amino_acids, repeat=2):
                    is_valid, mutations, mutation_count = validate_double_mutation(
                        original_seq, pos1, aa1, pos2, aa2
                    )

                    if is_valid:
                        mutant_seq = generate_mutant_sequence(original_seq, pos1, aa1, pos2, aa2)
                        seqs.append(mutant_seq)
                        combos.append((pos1 + 1, aa1, pos2 + 1, aa2))
                        infos.append({
                            'mutations': mutations,
                            'mutation_count': mutation_count,
                            'original_aa1': original_seq[pos1],
                            'original_aa2': original_seq[pos2]
                        })
                        valid_count += 1
                    else:
                        invalid_count += 1

                    pbar.update(1)
                    pbar.set_postfix({
                        'valid': valid_count,
                        'invalid': invalid_count,
                        'valid_rate': f"{(valid_count / (valid_count + invalid_count)) * 100:.1f}%"
                    })

        np.save(seq_cache, np.array(seqs, dtype=object))
        np.save(combo_cache, np.array(combos, dtype=object))
        np.save(info_cache, np.array(infos, dtype=object))

        print(f"Generation finished! Valid: {valid_count}, Invalid: {invalid_count}")
        print(f"Validity rate: {(valid_count / (valid_count + invalid_count)) * 100:.2f}%")

    return seqs, combos, infos


# ===== Load models =====
def load_models():
    """Load ESM2 model and trained regression models"""
    print("Loading models...")

    missing_models = [path for path in MODEL_PATHS if not os.path.exists(path)]
    if missing_models:
        raise FileNotFoundError(f"Missing model files: {missing_models}")

    XGBoost_models = []
    for model_path in tqdm(MODEL_PATHS, desc="Loading models", unit="model"):
        with open(model_path, "rb") as f:
            XGBoost_models.append(pickle.load(f))

    print(f"Successfully loaded {len(XGBoost_models)} models")

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL, do_lower_case=False)
    model = AutoModel.from_pretrained(
        ESM2_MODEL,
        output_hidden_states=True,
        torch_dtype=torch.float16 if USE_FP16 else torch.float32
    ).to(DEVICE)

    model.eval()
    return tokenizer, model, XGBoost_models


# ===== Extract ESM embeddings =====
def extract_esm_embeddings(seqs, tokenizer, model):
    """Extract ESM2 embeddings with caching"""
    emb_cache = "valid_mutant_embeddings.npy"

    if os.path.exists(emb_cache):
        print("Loading cached embeddings...")
        return np.load(emb_cache)

    print("Extracting ESM2 embeddings...")
    embs = []
    total_batches = (len(seqs) + PRED_BATCH_SIZE - 1) // PRED_BATCH_SIZE

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_FP16):
        for i in tqdm(range(0, len(seqs), PRED_BATCH_SIZE), total=total_batches, desc="Embedding"):
            batch = seqs[i: i + PRED_BATCH_SIZE]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

            ids = enc["input_ids"].to(DEVICE)
            mask = enc["attention_mask"].to(DEVICE).unsqueeze(-1)
            if USE_FP16:
                mask = mask.half()

            out = model(ids, attention_mask=mask.squeeze(-1))
            last = out.hidden_states[-1]

            summed = (last * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-9)
            batch_embs = (summed / lengths).float().cpu().numpy()

            embs.append(batch_embs)

            del enc, ids, mask, out, last, summed, lengths
            torch.cuda.empty_cache()

    embeddings = np.vstack(embs)
    np.save(emb_cache, embeddings)
    print(f"Embeddings cached to {emb_cache}")

    return embeddings


# ===== Parallel prediction =====
def parallel_predict_multiple_models(embeddings, XGBoost_models):
    """Predict using multiple models and compute mean/std"""
    print(f"Running predictions with {len(XGBoost_models)} models...")

    all_preds = np.zeros((len(XGBoost_models), len(embeddings)), dtype=np.float32)

    for model_idx, XGBoost_model in enumerate(tqdm(XGBoost_models, desc="Model prediction", unit="model")):
        print(f"  Predicting with model {model_idx + 1}/{len(XGBoost_models)}")

        preds = np.empty(len(embeddings), dtype=np.float32)
        chunk_size = max(1, len(embeddings) // CPU_WORKERS)
        chunks = [embeddings[i: i + chunk_size] for i in range(0, len(embeddings), chunk_size)]

        with ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
            futures = {executor.submit(XGBoost_model.predict, chunk): idx for idx, chunk in enumerate(chunks)}
            for future in tqdm(as_completed(futures), total=len(chunks), leave=False):
                idx = futures[future]
                res = future.result()
                start = idx * chunk_size
                preds[start:start + len(res)] = res

        all_preds[model_idx] = preds

    return np.mean(all_preds, axis=0), np.std(all_preds, axis=0, ddof=1), all_preds


# ===== Memory cleanup =====
def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ===== Main =====
def main():
    print("=" * 60)
    print("Protein mutant screening system - multi-model ensemble prediction")
    print(f"Device: {DEVICE.upper()}, FP16: {'Enabled' if USE_FP16 else 'Disabled'}")
    print("=" * 60)

    total_start = time.time()

    try:
        print("\n[1/5] Generating valid double mutants...")
        seqs, combos, infos = generate_valid_pairwise_mutants()

        print("\n[2/5] Loading models...")
        tokenizer, esm2_model, XGBoost_models = load_models()

        print("\n[3/5] Extracting ESM2 embeddings...")
        embeddings = extract_esm_embeddings(seqs, tokenizer, esm2_model)
        del esm2_model
        clear_memory()

        print("\n[4/5] Predicting...")
        pred_means, pred_stds, _ = parallel_predict_multiple_models(embeddings, XGBoost_models)

        print("\n[5/5] Filtering and saving results...")
        idxs = np.where(pred_means > PRED_THRESHOLD)[0]

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Pos1", "AA1", "Pos2", "AA2",
                "Original_AA1", "Original_AA2",
                "Mutation1", "Mutation2",
                "Mutated_Sequence", "Pred_Mean"
            ])
            for i in idxs:
                pos1, aa1, pos2, aa2 = combos[i]
                info = infos[i]
                writer.writerow([
                    pos1, aa1, pos2, aa2,
                    info['original_aa1'], info['original_aa2'],
                    info['mutations'][0], info['mutations'][1],
                    seqs[i], f"{pred_means[i]:.4f}"
                ])

        print(f"Finished! Saved {len(idxs)} mutants to {OUTPUT_CSV}")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        clear_memory()
        print(f"Total time: {time.time() - total_start:.2f} s")


if __name__ == "__main__":
    main()
