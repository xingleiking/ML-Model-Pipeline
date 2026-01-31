import os
import gc
import pickle
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ============================================================
# Configuration
# ============================================================
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Batch size for ESM feature extraction (adjust based on GPU memory)
FEATURE_CACHE = "cached_esm2_features.pt"
LABEL_CACHE = "cached_labels.npy"
RANDOM_STATE = 42

# ------------------------------------------------------------
# Best hyperparameters (fixed)
# ------------------------------------------------------------
BEST_PARAMS = {
    "colsample_bytree": 0.6,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 3,
    "n_estimators": 100,
    "reg_alpha": 0.2,
    "reg_lambda": 5,
    "subsample": 0.5
}

# ============================================================
# Matplotlib style settings (publication-ready)
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["figure.titlesize"] = 12


# ============================================================
# Utility functions
# ============================================================
def set_global_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_torch_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# ESM model loading and feature extraction
# ============================================================
def load_esm_model(model_name):
    """Load ESM2 tokenizer and model with reduced memory footprint."""
    print(f"Loading ESM2 model `{model_name}` to {DEVICE} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    model = model.to(DEVICE).eval()
    clear_torch_cuda()
    return tokenizer, model


def extract_esm_embeddings(sequences, tokenizer, model, batch_size=4, max_length=1024):
    """
    Extract ESM embeddings in batches (mean pooling of last hidden state).

    Returns
    -------
    np.ndarray
        Shape: (n_sequences, embedding_dim)
    """
    print("Extracting ESM embeddings ...")
    all_embs = []

    with torch.no_grad():
        total_batches = (len(sequences) + batch_size - 1) // batch_size
        for idx, start in enumerate(range(0, len(sequences), batch_size), 1):
            batch = sequences[start : start + batch_size]
            print(f"  Batch {idx}/{total_batches}", end="\r")

            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

            out = model(input_ids, attention_mask=attention_mask)
            last_hidden = out.hidden_states[-1]  # (batch, seq_len, hidden)

            mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            summed = (last_hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / lengths).float().cpu().numpy()
            all_embs.append(mean_pooled)

            # Release memory
            del enc, input_ids, attention_mask, out, last_hidden, mask, summed, lengths, mean_pooled
            clear_torch_cuda()

    embeddings = np.vstack(all_embs) if all_embs else np.zeros((0, 0))
    print(f"\n  Extracted embeddings shape: {embeddings.shape}")
    return embeddings


# ============================================================
# Cross-validation (return best-fold model for reproducibility)
# ============================================================
def perform_cross_validation_and_save_best(X, y, n_splits=5, random_state=42):
    """
    Perform K-fold cross-validation.

    Returns
    -------
    cv_results : dict
        Aggregated CV statistics
    best_model : XGBRegressor
        Model from the best-performing fold (by validation R²)
    best_score : float
        Best validation R²
    best_spearman : float
        Spearman correlation of the best fold
    best_rmse : float
        RMSE of the best fold
    best_mae : float
        MAE of the best fold
    fold_infos : list of dict
        Detailed metrics for each fold
    """
    print(f"\n=== Performing {n_splits}-fold Cross Validation ===")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_r2_scores = []
    cv_spearman_scores = []
    cv_rmse_scores = []
    cv_mae_scores = []

    best_score = -np.inf
    best_model = None
    best_spearman = -np.inf
    best_rmse = np.inf
    best_mae = np.inf

    fold_infos = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"Training fold {fold_idx}/{n_splits} ...")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
            **BEST_PARAMS,
        )

        model.fit(X_train, y_train)

        # Training evaluation
        preds_train = model.predict(X_train)
        train_r2 = r2_score(y_train, preds_train)
        train_spearman = spearmanr(y_train, preds_train).correlation
        train_rmse = np.sqrt(mean_squared_error(y_train, preds_train))
        train_mae = mean_absolute_error(y_train, preds_train)

        # Validation evaluation
        preds_val = model.predict(X_val)
        val_r2 = r2_score(y_val, preds_val)
        val_spearman = spearmanr(y_val, preds_val).correlation
        val_rmse = np.sqrt(mean_squared_error(y_val, preds_val))
        val_mae = mean_absolute_error(y_val, preds_val)

        cv_r2_scores.append(val_r2)
        cv_spearman_scores.append(val_spearman)
        cv_rmse_scores.append(val_rmse)
        cv_mae_scores.append(val_mae)

        fold_infos.append(
            {
                "fold": fold_idx,
                "train_r2": float(train_r2),
                "val_r2": float(val_r2),
                "train_spearman": float(train_spearman),
                "val_spearman": float(val_spearman),
                "train_rmse": float(train_rmse),
                "val_rmse": float(val_rmse),
                "train_mae": float(train_mae),
                "val_mae": float(val_mae),
                "val_size": len(val_idx),
                "train_size": len(train_idx),
            }
        )

        print(
            f"  Fold {fold_idx} Train R²: {train_r2:.4f}, "
            f"Spearman: {train_spearman:.4f}, "
            f"RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}"
        )
        print(
            f"  Fold {fold_idx} Val   R²: {val_r2:.4f}, "
            f"Spearman: {val_spearman:.4f}, "
            f"RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}"
        )

        # Track best fold by validation R²
        if val_r2 > best_score:
            best_score = val_r2
            best_spearman = val_spearman
            best_rmse = val_rmse
            best_mae = val_mae
            best_model = pickle.loads(pickle.dumps(model))

        del model, X_train, X_val, y_train, y_val, preds_val, preds_train
        clear_torch_cuda()
        gc.collect()

    cv_r2_scores = np.array(cv_r2_scores)
    cv_spearman_scores = np.array(cv_spearman_scores)
    cv_rmse_scores = np.array(cv_rmse_scores)
    cv_mae_scores = np.array(cv_mae_scores)

    # Save detailed CV results
    cv_df = pd.DataFrame(fold_infos)
    cv_df.to_csv("cross_validation_results.csv", index=False, encoding="utf-8")
    print("\nCross-validation results saved to cross_validation_results.csv")

    # Print summary statistics
    print("\nCross-validation R² statistics:")
    print(f"  Mean R² = {cv_r2_scores.mean():.4f} ± {cv_r2_scores.std():.4f}")
    print(f"  Best R² = {cv_r2_scores.max():.4f}")

    print("\nCross-validation Spearman statistics:")
    print(
        f"  Mean Spearman = {cv_spearman_scores.mean():.4f} "
        f"± {cv_spearman_scores.std():.4f}"
    )
    print(f"  Best Spearman = {cv_spearman_scores.max():.4f}")

    print("\nCross-validation RMSE statistics:")
    print(
        f"  Mean RMSE = {cv_rmse_scores.mean():.4f} "
        f"± {cv_rmse_scores.std():.4f}"
    )
    print(f"  Best RMSE = {cv_rmse_scores.min():.4f}")

    print("\nCross-validation MAE statistics:")
    print(
        f"  Mean MAE = {cv_mae_scores.mean():.4f} "
        f"± {cv_mae_scores.std():.4f}"
    )
    print(f"  Best MAE = {cv_mae_scores.min():.4f}")

    cv_results = {
        "r2_scores": cv_r2_scores,
        "spearman_scores": cv_spearman_scores,
        "rmse_scores": cv_rmse_scores,
        "mae_scores": cv_mae_scores,
        "mean_r2": cv_r2_scores.mean(),
        "std_r2": cv_r2_scores.std(),
        "mean_spearman": cv_spearman_scores.mean(),
        "std_spearman": cv_spearman_scores.std(),
        "mean_rmse": cv_rmse_scores.mean(),
        "std_rmse": cv_rmse_scores.std(),
        "mean_mae": cv_mae_scores.mean(),
        "std_mae": cv_mae_scores.std(),
        "best_r2": cv_r2_scores.max(),
        "best_spearman": cv_spearman_scores.max(),
        "best_rmse": cv_rmse_scores.min(),
        "best_mae": cv_mae_scores.min(),
    }

    return (
        cv_results,
        best_model,
        best_score,
        best_spearman,
        best_rmse,
        best_mae,
        fold_infos,
    )


# ============================================================
# Main workflow
# ============================================================
def main(csv_path="normalized.csv"):
    start_time = time.time()
    set_global_seed(RANDOM_STATE)

    print("=== ESM + XGBoost pipeline (for publication) ===")
    print(f"Device: {DEVICE}, Random seed: {RANDOM_STATE}")

    print("Loading data ...")
    df = pd.read_csv(csv_path)
    sequences = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].astype(float).values
    print(f"Loaded {len(sequences)} samples")

    # --------------------------------------------------------
    # Feature extraction or cache loading
    # --------------------------------------------------------
    if os.path.exists(FEATURE_CACHE) and os.path.exists(LABEL_CACHE):
        print("Loading cached features and labels ...")
        X = torch.load(FEATURE_CACHE).numpy()
        labels = np.load(LABEL_CACHE)
    else:
        tokenizer, esm_model = load_esm_model(ESM2_MODEL)
        X = extract_esm_embeddings(
            sequences, tokenizer, esm_model, batch_size=BATCH_SIZE
        )
        torch.save(torch.tensor(X), FEATURE_CACHE)
        np.save(LABEL_CACHE, labels)

        del tokenizer, esm_model
        clear_torch_cuda()
        gc.collect()

    print(f"Feature matrix shape: {X.shape}")

    # --------------------------------------------------------
    # Cross-validation (save best-fold model)
    # --------------------------------------------------------
    (
        cv_results,
        best_cv_model,
        best_cv_r2,
        best_cv_spearman,
        best_cv_rmse,
        best_cv_mae,
        fold_infos,
    ) = perform_cross_validation_and_save_best(
        X, labels, n_splits=5, random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # Standard train/test split evaluation
    # --------------------------------------------------------
    print("\n=== Standard Train/Test Split (for reporting) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    xgb_on_split = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
        **BEST_PARAMS,
    )

    print("Training model on train split ...")
    xgb_on_split.fit(X_train, y_train)

    # Training evaluation
    y_train_pred = xgb_on_split.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_spearman = spearmanr(y_train, y_train_pred).correlation
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)

    # Test evaluation
    y_test_pred = xgb_on_split.predict(X_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_spearman = spearmanr(y_test, y_test_pred).correlation
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(
        f"Train R²: {train_r2:.4f}, Spearman: {train_spearman:.4f}, "
        f"RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}"
    )
    print(
        f"Test  R²: {test_r2:.4f}, Spearman: {test_spearman:.4f}, "
        f"RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}"
    )

    # --------------------------------------------------------
    # Save plotting data
    # --------------------------------------------------------
    plot_rows = []

    for info in fold_infos:
        plot_rows.append(
            {
                "figure": "CV_bar",
                "fold": info["fold"],
                "subset": "train",
                "r2": info["train_r2"],
                "spearman": info["train_spearman"],
                "rmse": info["train_rmse"],
                "mae": info["train_mae"],
                "size": info["train_size"],
            }
        )
        plot_rows.append(
            {
                "figure": "CV_bar",
                "fold": info["fold"],
                "subset": "val",
                "r2": info["val_r2"],
                "spearman": info["val_spearman"],
                "rmse": info["val_rmse"],
                "mae": info["val_mae"],
                "size": info["val_size"],
            }
        )

    for y_t, y_p in zip(y_train, y_train_pred):
        plot_rows.append(
            {
                "figure": "scatter",
                "subset": "train",
                "true_value": float(y_t),
                "pred_value": float(y_p),
            }
        )

    for y_t, y_p in zip(y_test, y_test_pred):
        plot_rows.append(
            {
                "figure": "scatter",
                "subset": "test",
                "true_value": float(y_t),
                "pred_value": float(y_p),
            }
        )

    pd.DataFrame(plot_rows).to_csv("cvx_plot_data.csv", index=False, encoding="utf-8")
    print("Saved plotting data -> cvx_plot_data.csv")

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------
    with open("xgb_model_on_train_split.pkl", "wb") as f:
        pickle.dump(xgb_on_split, f)
    print("Saved model trained on train split -> xgb_model_on_train_split.pkl")

    print("\nTraining final model on ALL available data ...")
    final_xgb = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
        **BEST_PARAMS,
    )
    final_xgb.fit(X, labels)

    with open("xgb_final_all_data_model.pkl", "wb") as f:
        pickle.dump(final_xgb, f)
    print("Saved final model -> xgb_final_all_data_model.pkl")

    if best_cv_model is not None:
        with open("xgb_best_fold_model.pkl", "wb") as f:
            pickle.dump(best_cv_model, f)
        print(
            f"Saved best CV-fold model "
            f"(R²={best_cv_r2:.4f}, Spearman={best_cv_spearman:.4f}, "
            f"RMSE={best_cv_rmse:.4f}, MAE={best_cv_mae:.4f})"
        )

    # --------------------------------------------------------
    # Visualization (1 row × 4 columns: all metrics)
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=300)

    ax = axes[0]
    bars = ax.bar(range(1, len(cv_results["r2_scores"]) + 1), cv_results["r2_scores"])
    ax.axhline(
        y=cv_results["mean_r2"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f'Mean R² = {cv_results["mean_r2"]:.4f}',
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("R²")
    ax.set_title("Cross-validation R²")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bars = ax.bar(
        range(1, len(cv_results["spearman_scores"]) + 1),
        cv_results["spearman_scores"],
        color="orange",
    )
    ax.axhline(
        y=cv_results["mean_spearman"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f'Mean Spearman = {cv_results["mean_spearman"]:.4f}',
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("Spearman")
    ax.set_title("Cross-validation Spearman")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    bars = ax.bar(
        range(1, len(cv_results["rmse_scores"]) + 1),
        cv_results["rmse_scores"],
        color="green",
    )
    ax.axhline(
        y=cv_results["mean_rmse"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f'Mean RMSE = {cv_results["mean_rmse"]:.4f}',
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("RMSE")
    ax.set_title("Cross-validation RMSE")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    bars = ax.bar(
        range(1, len(cv_results["mae_scores"]) + 1),
        cv_results["mae_scores"],
        color="purple",
    )
    ax.axhline(
        y=cv_results["mean_mae"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f'Mean MAE = {cv_results["mean_mae"]:.4f}',
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("MAE")
    ax.set_title("Cross-validation MAE")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cross_validation_results.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved figure -> cross_validation_results.png")

    # --------------------------------------------------------
    # Save textual summary
    # --------------------------------------------------------
    summary_text = []
    summary_text.append("=== Model training summary ===")
    summary_text.append(f"Date: {time.asctime()}")
    summary_text.append(f"Random seed: {RANDOM_STATE}")
    summary_text.append(f"Dataset: {csv_path}")
    summary_text.append(f"Number of samples: {len(labels)}")
    summary_text.append(f"Feature shape: {X.shape}")
    summary_text.append("")
    summary_text.append("Cross-validation (5-fold):")
    summary_text.append(f"  Mean R² = {cv_results['mean_r2']:.4f} ± {cv_results['std_r2']:.4f}")
    summary_text.append(f"  Best R²  = {cv_results['best_r2']:.4f}")
    summary_text.append(
        f"  Mean Spearman = {cv_results['mean_spearman']:.4f} "
        f"± {cv_results['std_spearman']:.4f}"
    )
    summary_text.append(f"  Best Spearman  = {cv_results['best_spearman']:.4f}")
    summary_text.append(f"  Mean RMSE = {cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
    summary_text.append(f"  Best RMSE  = {cv_results['best_rmse']:.4f}")
    summary_text.append(f"  Mean MAE = {cv_results['mean_mae']:.4f} ± {cv_results['std_mae']:.4f}")
    summary_text.append(f"  Best MAE  = {cv_results['best_mae']:.4f}")
    summary_text.append("")
    summary_text.append("Train/Test split (80%/20%):")
    summary_text.append(
        f"  Train R² = {train_r2:.4f}, Spearman = {train_spearman:.4f}, "
        f"RMSE = {train_rmse:.6f}, MAE = {train_mae:.6f}"
    )
    summary_text.append(
        f"  Test  R² = {test_r2:.4f}, Spearman = {test_spearman:.4f}, "
        f"RMSE = {test_rmse:.6f}, MAE = {test_mae:.6f}"
    )
    summary_text.append("")
    summary_text.append("Saved models:")
    summary_text.append("  xgb_model_on_train_split.pkl")
    summary_text.append("  xgb_best_fold_model.pkl")
    summary_text.append("  xgb_final_all_data_model.pkl")
    summary_text.append("")
    summary_text.append("Best hyperparameters:")
    for k, v in BEST_PARAMS.items():
        summary_text.append(f"  {k}: {v}")

    with open("model_performance_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_text))
    print("Saved model performance summary -> model_performance_summary.txt")

    detailed_metrics = pd.DataFrame(
        {
            "Metric": ["R²", "Spearman", "RMSE", "MAE"],
            "CV_Mean": [
                cv_results["mean_r2"],
                cv_results["mean_spearman"],
                cv_results["mean_rmse"],
                cv_results["mean_mae"],
            ],
            "CV_Std": [
                cv_results["std_r2"],
                cv_results["std_spearman"],
                cv_results["std_rmse"],
                cv_results["std_mae"],
            ],
            "CV_Best": [
                cv_results["best_r2"],
                cv_results["best_spearman"],
                cv_results["best_rmse"],
                cv_results["best_mae"],
            ],
            "Train": [train_r2, train_spearman, train_rmse, train_mae],
            "Test": [test_r2, test_spearman, test_rmse, test_mae],
        }
    )
    detailed_metrics.to_csv("detailed_metrics.csv", index=False, encoding="utf-8")
    print("Saved detailed metrics -> detailed_metrics.csv")

    total_time = time.time() - start_time
    print("\nAll tasks completed successfully!")
    print(f"Total elapsed time: {total_time:.2f} seconds")
    print("All models and result files have been saved to the current directory.")


if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "normalized.csv"
    main(csv_path)
