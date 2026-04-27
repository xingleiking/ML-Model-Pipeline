import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from scipy.stats import spearmanr

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
N_SPLITS       = 5   # 5-fold cross-validation (~80/20 train-test split)

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
    Lightweight DKL:

    An MLP projects inputs into a low-dimensional representation,
    which is then passed to a Gaussian Process with an RBF kernel.

    L2 regularisation and early stopping are used
    to reduce overfitting.
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
        """Extract hidden-layer representation."""
        X_scaled = self.scaler.transform(X)

        hidden = X_scaled
        for W, b in zip(self.mlp.coefs_[:-1], self.mlp.intercepts_[:-1]):
            hidden = np.maximum(0, hidden @ W + b)

        return hidden

    def fit(self, X, y):
        """Train MLP and GP."""
        X_scaled = self.scaler.fit_transform(X)

        self.mlp.fit(X_scaled, y)

        self.gp.fit(self._get_hidden(X), y)

        return self

    def predict(self, X):
        """Predict using trained model."""
        return self.gp.predict(self._get_hidden(X))


# ─────────────────────────── Regressor Factory ───────────────────────

def make_regressor(name: str):
    """Return a fresh regressor instance to prevent state leakage across folds."""

    registry = {

        "RandomForest":
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
            ),

        "GBRT":
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=10,
                min_samples_leaf=4,
                subsample=0.8,
                random_state=42,
            ),

        "KNN":
            KNeighborsRegressor(n_neighbors=5),

        "DKL":
            DeepKernelLearning(),
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


# ─────────────────────────── Feature Extraction ──────────────────────

def load_hf_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        do_lower_case=False,
        max_length=MAX_SEQ_LENGTH,
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

    # ProtBert requires amino acids separated by spaces.
    # Otherwise tokenization fails and embeddings become identical,
    # leading to NaN Spearman correlation.

    is_protbert = "prot_bert" in model_name.lower()

    if is_protbert:
        print("  [ProtBert] Reformatting sequences with space-separated amino acids...")
        sequences = [" ".join(list(seq)) for seq in sequences]

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

            out = model(
                input_ids,
                attention_mask=attention_mask,
            )

            layer_embs = []

            for idx in [-1, -2, -3]:

                hidden = out.hidden_states[idx]

                mask = attention_mask.unsqueeze(-1)

                mean_pooled = (
                    (hidden * mask).sum(dim=1)
                    / mask.sum(dim=1).clamp(min=1e-9)
                )

                layer_embs.append(mean_pooled.cpu())

            all_embs.append(
                torch.cat(layer_embs, dim=1).numpy()
            )

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
        [
            [seq.count(aa) / len(seq) for aa in aa_list]
            for seq in sequences
        ],
        dtype=float,
    )

    np.save(cache_file, X)

    print(f"  AAindex features cached to: {cache_file}")

    return X


# ─────────────────────────── Evaluation & Plotting ───────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Evaluation metrics:

    - R²              : goodness-of-fit
    - Spearman ρ      : ranking correlation (critical for screening tasks)
    - MAE             : mean absolute error
    - RMSE            : root mean squared error

    Note:
    If predictions are constant, Spearman returns NaN.
    In that case it will be reset to 0.0.
    """

    try:

        spearman_rho, _ = spearmanr(y_true, y_pred)

        if np.isnan(spearman_rho):

            print("  [Warning] Spearman ρ is NaN (predictions may be constant). Set to 0.0.")

            spearman_rho = 0.0

    except Exception as e:

        print(f"  [Warning] Spearman calculation failed: {e}. Set to 0.0.")

        spearman_rho = 0.0

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {

        "R2":
            r2_score(y_true, y_pred),

        "Spearman":
            round(float(spearman_rho), 4),

        "MAE":
            mean_absolute_error(y_true, y_pred),

        "RMSE":
            rmse,
    }


# Remaining main() function stays exactly the same
# (no structural changes required)

if __name__ == "__main__":
    main()
