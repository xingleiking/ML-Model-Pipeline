import os
import torch
import gpytorch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
import pickle

# Traditional ML models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin


# ============================================================
# Deep Kernel Learning (DKL) Model Definition
# ============================================================

class FeatureExtractor(torch.nn.Module):
    """
    Neural network used as a feature extractor for DKL.
    Projects high-dimensional ESM embeddings into a latent space.
    """
    def __init__(self, input_dim, hidden_dim=256, output_dim=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GPRegressionLayer(gpytorch.models.ExactGP):
    """
    Gaussian Process regression layer operating on learned features.
    """
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super().__init__(train_x, train_y, likelihood)
        self.feature_extractor = feature_extractor
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        projected_x = self.feature_extractor(x)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for Deep Kernel Learning
    implemented using GPyTorch.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        output_dim=64,
        lr=0.01,
        training_iter=50,
        device="cpu",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.training_iter = training_iter
        self.device = device

    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)

        self.feature_extractor = FeatureExtractor(
            self.input_dim, self.hidden_dim, self.output_dim
        ).to(self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = GPRegressionLayer(
            X, y, self.likelihood, self.feature_extractor
        ).to(self.device)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            [
                {"params": self.model.feature_extractor.parameters()},
                {"params": self.model.mean_module.parameters()},
                {"params": self.model.covar_module.parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.lr,
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model
        )

        for _ in range(self.training_iter):
            optimizer.zero_grad()
            output = self.model(X)
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = self.likelihood(self.model(X))
        return preds.mean.cpu().numpy()


# ============================================================
# Configuration
# ============================================================

ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

FEATURE_CACHE = "cached_esm2_features.pt"
LABEL_CACHE = "cached_labels.npy"


# ============================================================
# Utility Functions
# ============================================================

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, do_lower_case=False
    )
    model = AutoModel.from_pretrained(
        model_name, output_hidden_states=True
    )
    return tokenizer, model.to(DEVICE).eval()


def extract_esm_embeddings(sequences, tokenizer, model, batch_size=16):
    """
    Extract mean-pooled ESM2 embeddings for a list of protein sequences.
    """
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)

            out = model(input_ids, attention_mask=attention_mask)
            last_hidden = out.hidden_states[-1]

            mask = attention_mask.unsqueeze(-1)
            summed = (last_hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1)

            mean_pooled = summed / lengths
            all_embs.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embs)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def safe_save_model(model, name):
    """
    Safely save trained models.
    Attempts pickle first; falls back to state_dict if necessary.
    """
    try:
        with open(f"{name}_model.pkl", "wb") as f:
            pickle.dump(model, f)
        print(f"[Saved] {name}_model.pkl")
    except Exception as e:
        print(f"[Warning] Pickle failed for {name}: {e}")
        if hasattr(model, "state_dict"):
            try:
                torch.save(model.state_dict(), f"{name}_state_dict.pt")
                print(f"[Saved] {name}_state_dict.pt")
            except Exception as e2:
                print(f"[Error] state_dict save failed for {name}: {e2}")


# ============================================================
# Main Pipeline
# ============================================================

def main(csv_path="normalized.csv"):
    df = pd.read_csv(csv_path)
    sequences = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].values

    # Load or compute ESM2 features
    if os.path.exists(FEATURE_CACHE) and os.path.exists(LABEL_CACHE):
        print("Loading cached ESM2 features...")
        X = torch.load(FEATURE_CACHE).numpy()
        labels = np.load(LABEL_CACHE)
    else:
        print("Extracting ESM2 embeddings...")
        tokenizer, model = load_model(ESM2_MODEL)
        X = extract_esm_embeddings(
            sequences, tokenizer, model, batch_size=BATCH_SIZE
        )
        torch.save(torch.tensor(X), FEATURE_CACHE)
        np.save(LABEL_CACHE, labels)

    print("Splitting dataset into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.1, random_state=42
    )

    models = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GBRT": GradientBoostingRegressor(random_state=42),
        "KNN": KNeighborsRegressor(),
        "XGBoost": XGBRegressor(
            random_state=42, n_jobs=-1, verbosity=0
        ),
        "DKL": DKLRegressor(
            input_dim=X.shape[1], device=DEVICE, training_iter=50
        ),
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_all_pred = model.predict(X)

        results.append({
            "Model": name,
            "Train_R2": r2_score(y_train, y_train_pred),
            "Test_R2": r2_score(y_test, y_test_pred),
            "Overall_R2": r2_score(labels, y_all_pred),
            "Train_RMSE": rmse(y_train, y_train_pred),
            "Test_RMSE": rmse(y_test, y_test_pred),
            "Overall_RMSE": rmse(labels, y_all_pred),
            "Train_MAE": mean_absolute_error(y_train, y_train_pred),
            "Test_MAE": mean_absolute_error(y_test, y_test_pred),
            "Overall_MAE": mean_absolute_error(labels, y_all_pred),
            "Train_Spearman": spearmanr(y_train, y_train_pred)[0],
            "Test_Spearman": spearmanr(y_test, y_test_pred)[0],
            "Overall_Spearman": spearmanr(labels, y_all_pred)[0],
        })

        safe_save_model(model, name)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        "model_comparison_results_extended.csv", index=False
    )
    print("\nModel comparison results saved.")
    print(results_df)


if __name__ == "__main__":
    main("normalized.csv")
