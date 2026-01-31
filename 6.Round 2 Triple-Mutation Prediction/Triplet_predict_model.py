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

'''
- Effective sequences are 823,000 sequences (i.e., 823k sequences).
'''

# ===== Configuration =====
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATHS = ["xgb_final_all_data_model.pkl"]  # If multiple models, list them here
OUTPUT_CSV = "filtered_triplet_mutants.csv"
PRED_THRESHOLD = 0.55

# ===== Hardware Optimization Parameters (for RTX 4060 8GB VRAM) =====
PRED_BATCH_SIZE = 64  # Reduce batch size to fit in 8GB VRAM
CPU_WORKERS = 24      # On Windows, use 16 cores to avoid resource contention
USE_FP16 = True       # Enable half-precision for speed
PROGRESS_INTERVAL = 1000  # Show progress every 1000 sequences

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


# ===== Triple Mutation Validation =====
def validate_triple_mutation(original_seq, pos1, aa1, pos2, aa2, pos3, aa3):
    """
    Validate whether it is a true triple mutant (all three sites are mutated)
    Returns: (is_valid_triple_mutation, actual_mutations, mutation_count)
    actual_mutations is a list like ["A72G", "V113L", ...] (1-based positions)
    """
    orig_aa1 = original_seq[pos1]
    orig_aa2 = original_seq[pos2]
    orig_aa3 = original_seq[pos3]

    mutations = []
    mutation_count = 0

    if aa1 != orig_aa1:
        mutations.append(f"{orig_aa1}{pos1 + 1}{aa1}")
        mutation_count += 1
    if aa2 != orig_aa2:
        mutations.append(f"{orig_aa2}{pos2 + 1}{aa2}")
        mutation_count += 1
    if aa3 != orig_aa3:
        mutations.append(f"{orig_aa3}{pos3 + 1}{aa3}")
        mutation_count += 1

    is_valid = (mutation_count == 3)  # Only consider true triple mutants

    return is_valid, mutations, mutation_count


def generate_mutant_sequence_triple(original_seq, pos1, aa1, pos2, aa2, pos3, aa3):
    """Generate a mutant sequence containing three mutation sites"""
    seq_list = list(original_seq)
    seq_list[pos1] = aa1
    seq_list[pos2] = aa2
    seq_list[pos3] = aa3
    return "".join(seq_list)


# ===== Generate Valid Triple Mutants =====
def generate_valid_triplet_mutants():
    """Generate all valid triple mutant combinations (with caching). Returns seqs, combos, infos"""
    seq_cache = "valid_triplet_sequences.npy"
    combo_cache = "valid_triplet_combos.npy"
    info_cache = "valid_triplet_info.npy"

    if os.path.exists(seq_cache) and os.path.exists(combo_cache) and os.path.exists(info_cache):
        print("Loading cached triple mutant sequences...")
        seqs = np.load(seq_cache, allow_pickle=True).tolist()
        combos = np.load(combo_cache, allow_pickle=True).tolist()
        infos = np.load(info_cache, allow_pickle=True).tolist()
    else:
        print("Generating valid triple mutant combinations (only true triples)...")
        seqs, combos, infos = [], [], []

        total_combos = len(list(combinations(positions, 3))) * (len(amino_acids) ** 3)
        valid_count = 0
        invalid_count = 0

        with tqdm(total=total_combos, desc="Generating triple mutants", unit="comb") as pbar:
            for pos1, pos2, pos3 in combinations(positions, 3):
                for aa1, aa2, aa3 in product(amino_acids, repeat=3):
                    is_valid, mutations, mutation_count = validate_triple_mutation(
                        original_seq, pos1, aa1, pos2, aa2, pos3, aa3
                    )

                    if is_valid:
                        mutant_seq = generate_mutant_sequence_triple(original_seq, pos1, aa1, pos2, aa2, pos3, aa3)
                        seqs.append(mutant_seq)
                        combos.append((pos1 + 1, aa1, pos2 + 1, aa2, pos3 + 1, aa3))
                        infos.append({
                            'mutations': mutations,
                            'mutation_count': mutation_count,
                            'original_aa1': original_seq[pos1],
                            'original_aa2': original_seq[pos2],
                            'original_aa3': original_seq[pos3]
                        })
                        valid_count += 1
                    else:
                        invalid_count += 1

                    pbar.update(1)
                    pbar.set_postfix({
                        'Valid': valid_count,
                        'Invalid': invalid_count,
                        'Rate': f"{(valid_count / (valid_count + invalid_count)) * 100:.4f}%"
                    })

        # Save cache
        np.save(seq_cache, np.array(seqs, dtype=object))
        np.save(combo_cache, np.array(combos, dtype=object))
        np.save(info_cache, np.array(infos, dtype=object))

        print(f"\nGeneration completed! Valid triples: {valid_count}, Invalid combos: {invalid_count}")
        if valid_count + invalid_count > 0:
            print(f"Validity rate: {(valid_count / (valid_count + invalid_count)) * 100:.6f}%")

    return seqs, combos, infos


# ===== Load Models =====
def load_models():
    """Load ESM2 model and multiple  XGB model"""
    print("Loading models...")

    missing_models = [path for path in MODEL_PATHS if not os.path.exists(path)]
    if missing_models:
        raise FileNotFoundError(f"Missing model files: {missing_models}")

    XGBoost_models = []
    for model_path in tqdm(MODEL_PATHS, desc="Loading XGBoost models", unit="model"):
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


# ===== Extract ESM Embeddings =====
def extract_esm_embeddings(seqs, tokenizer, model):
    """Extract ESM2 embeddings with caching"""
    emb_cache = "valid_triplet_embeddings.npy"

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

    embeddings = np.vstack(embs) if len(embs) > 0 else np.zeros((0, 0))
    np.save(emb_cache, embeddings)
    print(f"Embeddings cached to {emb_cache}")

    return embeddings


# ===== Parallel Multi-Model Prediction =====
def parallel_predict_multiple_models(embeddings, XGBoost_models):
    """Predict in parallel using multiple models and compute mean"""
    print(f"Predicting with {len(XGBoost_models)} models...")

    if len(XGBoost_models) == 0:
        raise ValueError("No models loaded!")

    all_preds = np.zeros((len(XGBoost_models), len(embeddings)), dtype=np.float32)

    for model_idx, XGBoost_model in enumerate(tqdm(XGBoost_models, desc="Model Prediction", unit="model")):
        print(f"  Predicting with model {model_idx + 1}/{len(XGBoost_models)}...")

        preds = np.empty(len(embeddings), dtype=np.float32)
        chunk_size = max(1, len(embeddings) // CPU_WORKERS)
        chunks = [
            embeddings[i: i + chunk_size]
            for i in range(0, len(embeddings), chunk_size)
        ]

        completed = 0
        total_chunks = len(chunks)

        with ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
            futures = {
                executor.submit(XGBoost_model.predict, chunk): idx
                for idx, chunk in enumerate(chunks)
            }

            for future in tqdm(as_completed(futures), total=total_chunks, desc=f"Model {model_idx + 1}", leave=False):
                idx = futures[future]
                res = future.result()
                start = idx * chunk_size
                end = start + len(res)
                preds[start:end] = res

                completed += 1
                if completed % 5 == 0 or completed == total_chunks:
                    print(f"    Completed: {completed}/{total_chunks} chunks")

        all_preds[model_idx] = preds

    pred_means = np.mean(all_preds, axis=0)
    pred_stds = np.std(all_preds, axis=0, ddof=1) if all_preds.shape[0] > 1 else np.zeros(len(embeddings), dtype=np.float32)

    return pred_means, pred_stds, all_preds


# ===== Memory Optimization =====
def clear_memory():
    """Clear GPU and CPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ===== Main Program =====
def main():
    print("=" * 60)
    print("Protein Mutant Screening System - Triple Mutant Prediction (true triples only)")
    print(f"Device: {DEVICE.upper()}, FP16: {'Enabled' if USE_FP16 else 'Disabled'}")
    print(f"GPU Batch Size: {PRED_BATCH_SIZE}")
    print(f"CPU Workers: {CPU_WORKERS}")
    print(f"Number of Models: {len(MODEL_PATHS)}")
    print("=" * 60)

    total_start = time.time()

    try:
        # Stage 1: Generate triple mutants
        stage_start = time.time()
        print("\n[1/5] Generating valid triple mutants...")
        seqs, combos, infos = generate_valid_triplet_mutants()
        stage_time = time.time() - stage_start
        print(f"  Generation completed! Number of sequences: {len(seqs):,}, Time: {stage_time:.2f}s")

        # Show some example mutations
        print("\n  Example mutations:")
        for i in range(min(3, len(infos))):
            print(f"    Example {i + 1}: {infos[i]['mutations']}")

        # Stage 2: Load models
        stage_start = time.time()
        print("\n[2/5] Loading models...")
        tokenizer, esm2_model, XGBoost_models = load_models()
        stage_time = time.time() - stage_start
        print(f"  Models loaded! Time: {stage_time:.2f}s")

        # Stage 3: Extract embeddings
        stage_start = time.time()
        print("\n[3/5] Extracting ESM2 embeddings...")
        embeddings = extract_esm_embeddings(seqs, tokenizer, esm2_model)
        stage_time = time.time() - stage_start
        print(f"  Embeddings extracted! Shape: {embeddings.shape}, Time: {stage_time:.2f}s")

        del esm2_model
        clear_memory()

        # Stage 4: Multi-model prediction
        stage_start = time.time()
        print("\n[4/5] Multi-model prediction...")
        pred_means, pred_stds, all_preds = parallel_predict_multiple_models(embeddings, XGBoost_models)
        stage_time = time.time() - stage_start
        print(f"  Prediction completed! Time: {stage_time:.2f}s")

        if len(pred_means) > 0:
            print(f"  Prediction mean range: {pred_means.min():.4f} - {pred_means.max():.4f}")
            print(f"  Prediction std range: {pred_stds.min():.4f} - {pred_stds.max():.4f}")
        else:
            print("  No predictions generated (embeddings are empty)")

        # Stage 5: Filter and save
        stage_start = time.time()
        print("\n[5/5] Filtering and saving results...")
        idxs = np.where(pred_means > PRED_THRESHOLD)[0] if len(pred_means) > 0 else np.array([], dtype=int)

        with open(OUTPUT_CSV, "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            header = [
                "Pos1", "AA1", "Pos2", "AA2", "Pos3", "AA3",
                "Original_AA1", "Original_AA2", "Original_AA3",
                "Mutation1", "Mutation2", "Mutation3",
                "Mutated_Sequence",
                "Pred_Mean"
            ]
            writer.writerow(header)

            for i in tqdm(idxs, desc="Saving results", unit="rec"):
                pos1, aa1, pos2, aa2, pos3, aa3 = combos[i]
                info = infos[i]

                muts = info['mutations']
                muts = muts + [""] * (3 - len(muts))

                row = [
                    pos1, aa1, pos2, aa2, pos3, aa3,
                    info['original_aa1'], info['original_aa2'], info['original_aa3'],
                    muts[0], muts[1], muts[2],
                    seqs[i],
                    f"{pred_means[i]:.4f}"
                ]

                writer.writerow(row)

        stage_time = time.time() - stage_start
        print(f"  Saved results! Time: {stage_time:.2f}s")
        print(f"  Filtered {len(idxs)} triple mutants (threshold: {PRED_THRESHOLD})")

        total_time = time.time() - total_start
        print("\n" + "=" * 60)
        print("Task Completed!")
        print(f"Total time: {total_time:.2f}s")
        print(f"Processing speed: {len(seqs) / total_time:.2f} sequences/s" if total_time > 0 else "Processing speed: N/A")
        print(f"Filtered results: {len(idxs)}/{len(seqs)} valid triple mutants")
        if len(pred_means) > 0:
            print(f"Average prediction: {pred_means.mean():.4f} ± {pred_means.std():.4f}")
        print(f"Output file: {OUTPUT_CSV}")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        clear_memory()


if __name__ == "__main__":
    main()
