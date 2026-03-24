import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    print("[Warning] XGBoost not found. Install via: pip install xgboost")
    HAS_XGBOOST = False


# ─────────────────────────── Configuration ───────────────────────────

DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LENGTH = 100
BATCH_SIZE     = 2
N_SPLITS       = 5   # 5-fold CV ≈ 80/20 train-test split

DATA_FILE   = "9_1_normalized.csv"
RESULT_FILE = "cv_results_5fold.csv"

ENCODER_MODELS = {
    "AAindex":   None,
    "ESM2_8M":   "facebook/esm2_t6_8M_UR50D",
    "ESM2_35M":  "facebook/esm2_t12_35M_UR50D",
    "ESM2_150M": "facebook/esm2_t30_150M_UR50D",
    "ESM2_650M": "facebook/esm2_t33_650M_UR50D",
    "ProtBert":  "Rostlab/prot_bert",
}


# ─────────────────────────── Deep Kernel Learning ────────────────────

class DeepKernelLearning:
    """
    Lightweight DKL: an MLP projects inputs into a low-dimensional
    representation, which is then passed to a Gaussian Process with an
    RBF kernel.  L2 regularisation and early stopping are used to
    reduce overfitting.
    """

    def __init__(self):
        self.mlp = MLPRegressor(
            hidden_layer_sizes=(128, 32),
            activation="relu",
            solver="adam",
            max_iter=300,
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
        kernel = C(1.0) * RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-6,
            normalize_y=True,
        )
        self.scaler = StandardScaler()

    def _get_hidden(self, X):
        X_scaled = self.scaler.transform(X)
        hidden = X_scaled
        for W, b in zip(self.mlp.coefs_[:-1], self.mlp.intercepts_[:-1]):
            hidden = np.maximum(0, hidden @ W + b)
        return hidden

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.mlp.fit(X_scaled, y)
        self.gp.fit(self._get_hidden(X), y)
        return self

    def predict(self, X):
        return self.gp.predict(self._get_hidden(X))


# ─────────────────────────── Regressor Factory ───────────────────────

def make_regressor(name: str):
    """Return a fresh regressor instance to prevent state leakage across folds."""
    registry = {
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
        "GBRT": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42,
        ),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "DKL": DeepKernelLearning(),
    }
    if HAS_XGBOOST:
        registry["XGBoost"] = XGBRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.1,
            random_state=42,
            verbosity=0,
        )
    return registry[name]


REGRESSORS = list(make_regressor.__code__.co_consts)   # placeholder; keys resolved at runtime


# ─────────────────────────── Feature Extraction ──────────────────────

def load_hf_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, do_lower_case=False, max_length=MAX_SEQ_LENGTH
    )
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    return tokenizer, model


def extract_esm_embeddings(sequences: list, model_name: str) -> np.ndarray:
    cache_file = f"cached_{model_name.replace('/', '_')}.npy"
    if os.path.exists(cache_file):
        print(f"  Loading cached features: {cache_file}")
        return np.load(cache_file)

    tokenizer, model = load_hf_model(model_name)
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(sequences), BATCH_SIZE):
            batch = sequences[i : i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
            )
            input_ids      = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

            out = model(input_ids, attention_mask=attention_mask)

            layer_embs = []
            for idx in [-1, -2, -3]:
                hidden = out.hidden_states[idx]
                mask   = attention_mask.unsqueeze(-1)
                mean_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                layer_embs.append(mean_pooled.cpu())

            all_embs.append(torch.cat(layer_embs, dim=1).numpy())

    X = np.vstack(all_embs)
    np.save(cache_file, X)
    print(f"  Features cached to: {cache_file}")
    return X


def extract_aaindex_features(sequences: list) -> np.ndarray:
    cache_file = "cached_AAindex.npy"
    if os.path.exists(cache_file):
        print(f"  Loading cached features: {cache_file}")
        return np.load(cache_file)

    aa_list = "ACDEFGHIKLMNPQRSTVWY"
    X = np.array(
        [[seq.count(aa) / len(seq) for aa in aa_list] for seq in sequences],
        dtype=float,
    )
    np.save(cache_file, X)
    print(f"  AAindex features cached to: {cache_file}")
    return X


# ─────────────────────────── Evaluation & Plotting ───────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "R2":               r2_score(y_true, y_pred),
        "ExplainedVariance": explained_variance_score(y_true, y_pred),
        "MAE":              mean_absolute_error(y_true, y_pred),
        "MSE":              mean_squared_error(y_true, y_pred),
    }


def plot_cv_fit(y_true, y_preds, encoder: str, reg_name: str):
    fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
    ax.scatter(y_true, y_preds, alpha=0.6, edgecolors="k", linewidths=0.3)
    lo, hi = min(y_true), max(y_true)
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Ideal")
    ax.set_xlabel("True Value")
    ax.set_ylabel("CV Predicted Value")
    ax.set_title(f"{encoder} + {reg_name}\nCV R²={r2_score(y_true, y_preds):.4f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"plot_{encoder}_{reg_name}_cv.png", dpi=150)
    plt.close(fig)


def plot_encoder_comparison(results_df: pd.DataFrame):
    summary = (
        results_df.groupby("Encoder")["Test_R2_Mean"]
        .mean()
        .sort_values(ascending=False)
    )
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(summary)))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(summary.index, summary.values, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_ylabel("Avg Test R² (across regressors)")
    ax.set_title("Encoder Comparison — Average Test R² (5-Fold CV)")
    ax.set_xticklabels(summary.index, rotation=20, ha="right")
    for bar, val in zip(bars, summary.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center", va="bottom", fontsize=8,
        )
    fig.tight_layout()
    fig.savefig("encoder_comparison_5fold.png", dpi=150)
    plt.close(fig)

    print("\n[Encoder ranking by average test R²]")
    print(summary.to_string())
    return summary


# ─────────────────────────── Main Pipeline ───────────────────────────

def main():
    df        = pd.read_csv(DATA_FILE)
    sequences = df.iloc[:, 0].astype(str).tolist()
    labels    = df.iloc[:, 1].values

    reg_names = ["RandomForest", "GBRT", "KNN", "DKL"] + (["XGBoost"] if HAS_XGBOOST else [])
    kf        = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    results   = []

    for encoder_name, model_name in ENCODER_MODELS.items():
        print(f"\n{'=' * 55}")
        print(f"Extracting features with encoder: {encoder_name}")

        X = (
            extract_aaindex_features(sequences)
            if encoder_name == "AAindex"
            else extract_esm_embeddings(sequences, model_name)
        )

        for reg_name in reg_names:
            print(f"\n  >> {encoder_name} + {reg_name}  ({N_SPLITS}-fold CV, ~80/20 split)")

            train_metrics_list, test_metrics_list = [], []
            y_preds = np.zeros_like(labels, dtype=float)

            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                model = make_regressor(reg_name)
                model.fit(X_train, y_train)

                train_m = evaluate(y_train, model.predict(X_train))
                test_m  = evaluate(y_test,  model.predict(X_test))

                train_metrics_list.append(train_m)
                test_metrics_list.append(test_m)
                y_preds[test_idx] = model.predict(X_test)

                print(f"     Fold {fold+1} | Train R²: {train_m['R2']:.4f} | Test R²: {test_m['R2']:.4f}")

            def mean_std(lst, key):
                vals = [m[key] for m in lst]
                return np.mean(vals), np.std(vals)

            keys = ["R2", "ExplainedVariance", "MAE", "MSE"]
            train_row = {f"Train_{k}_Mean": round(mean_std(train_metrics_list, k)[0], 4) for k in keys}
            train_row.update({f"Train_{k}_Std":  round(mean_std(train_metrics_list, k)[1], 4) for k in keys})
            test_row  = {f"Test_{k}_Mean":  round(mean_std(test_metrics_list,  k)[0], 4) for k in keys}
            test_row.update({f"Test_{k}_Std":   round(mean_std(test_metrics_list,  k)[1], 4) for k in keys})

            overfit_gap = train_row["Train_R2_Mean"] - test_row["Test_R2_Mean"]
            print(f"     Overfitting gap (Train - Test R²): {overfit_gap:.4f}")

            results.append({
                "Encoder":     encoder_name,
                "Regressor":   reg_name,
                "Overfit_Gap": round(overfit_gap, 4),
                **train_row,
                **test_row,
            })

            plot_cv_fit(labels, y_preds, encoder_name, reg_name)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULT_FILE, index=False)
    print(f"\nAll results saved to: {RESULT_FILE}")
    print(results_df.to_string())

    plot_encoder_comparison(results_df)


if __name__ == "__main__":
    main()