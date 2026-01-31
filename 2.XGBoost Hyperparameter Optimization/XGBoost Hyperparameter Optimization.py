# esmXGBoost_fixed.py
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import pickle
import csv

# ============================================================
# Configuration
# ============================================================
ESM2_MODEL = "facebook/esm2_t33_650M_UR50D"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16  # Keep consistent with esmGBRT2
FEATURE_CACHE = "esm2_features.pt"  # Shared cache with GBRT
LABEL_CACHE = "labels.npy"
RANDOM_STATE = 42


# ============================================================
# Utility functions
# ============================================================
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    return tokenizer, model.to(DEVICE).eval()


def extract_esm_embeddings(sequences, tokenizer, model, batch_size=16):
    all_embs = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]
            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
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


# ============================================================
# Custom logger for GridSearchCV (XGBoost-style logging)
# ============================================================
class GridSearchLogger:
    def __init__(self, log_file="xgb_grid_search_logged.csv"):
        self.log_file = log_file
        self.results = []

    def __call__(self, candidate_params, cv_score, estimator=None):
        result = candidate_params.copy()
        result["cv_score"] = cv_score
        self.results.append(result)

        # Write results to CSV in real time
        with open(self.log_file, "w", newline="") as f:
            fieldnames = list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in self.results:
                writer.writerow(res)


# ============================================================
# Main pipeline
# ============================================================
def main(csv_path="normalized.csv"):
    # Fix random seeds for reproducibility
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    df = pd.read_csv(csv_path)
    sequences = df.iloc[:, 0].astype(str).tolist()
    labels = df.iloc[:, 1].values

    # Load cached features if available (shared with GBRT)
    if os.path.exists(FEATURE_CACHE) and os.path.exists(LABEL_CACHE):
        print("Loading cached features...")
        X = torch.load(FEATURE_CACHE).numpy()
        labels = np.load(LABEL_CACHE)
    else:
        print("Extracting features with ESM2...")
        tokenizer, model = load_model(ESM2_MODEL)
        X = extract_esm_embeddings(
            sequences,
            tokenizer,
            model,
            batch_size=BATCH_SIZE,
        )
        torch.save(torch.tensor(X), FEATURE_CACHE)
        np.save(LABEL_CACHE, labels)

    print(f"Feature dimension: {X.shape}")

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Hyperparameter search space
    param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.05, 0.1],
        "max_depth": [2, 3, 4],
        "min_child_weight": [1, 3],
        "subsample": [0.5, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.1, 0.2],
        "reg_lambda": [0.5, 1, 5],
    }

    # Optional logger (kept for consistency with GBRT workflow)
    logger = GridSearchLogger("xgb_grid_search_logged.csv")

    # Base XGBoost model
    # To avoid nested parallelism:
    # - XGBRegressor uses n_jobs=1
    # - GridSearchCV uses n_jobs=-1
    xgb_base = XGBRegressor(
        tree_method="hist",
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )

    grid = GridSearchCV(
        xgb_base,
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    print(
        "Starting GridSearchCV "
        "(this may take some time depending on data size and parameter grid)..."
    )
    grid.fit(X_train, y_train)

    # Save full GridSearch results
    results_df = pd.DataFrame(grid.cv_results_)
    results_df.to_csv(
        "xgb_grid_search_full_results.csv",
        index=False,
    )
    print(
        "Full GridSearchCV results saved to "
        "xgb_grid_search_full_results.csv"
    )

    model = grid.best_estimator_
    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)

    # Evaluation
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"Train R2: {train_r2:.4f}")
    print(f"Test R2:  {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Save trained model
    with open("xgb_optimized_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to xgb_optimized_model.pkl")

    # Feature importance
    feature_importance = model.feature_importances_
    top_features = (
        np.argsort(feature_importance)[-20:]
        if feature_importance is not None
        else []
    )

    if len(top_features) > 0:
        plt.figure(figsize=(10, 6))
        plt.barh(
            range(len(top_features)),
            feature_importance[top_features],
        )
        plt.yticks(
            range(len(top_features)),
            [f"Feature_{i}" for i in top_features],
        )
        plt.xlabel("Feature Importance")
        plt.title("XGBoost Feature Importance (Top 20)")
        plt.tight_layout()
        plt.savefig(
            "xgb_feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Performance visualization (GBRT-style)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    plt.figure(figsize=(8, 6), dpi=300)
    plt.scatter(
        y_train,
        y_train_pred,
        alpha=0.6,
        label=f"Train (R² = {train_r2:.4f})",
    )
    plt.scatter(
        y_test,
        y_test_pred,
        alpha=0.6,
        label=f"Test (R² = {test_r2:.4f})",
    )
    plt.plot(
        [min(labels), max(labels)],
        [min(labels), max(labels)],
        "k--",
        lw=1.5,
    )

    plt.xlabel("True Value", fontname="Times New Roman", fontsize=12)
    plt.ylabel("Predicted Value", fontname="Times New Roman", fontsize=12)
    plt.title(
        "XGBoost Regression Performance",
        fontname="Times New Roman",
        fontsize=12,
    )
    plt.legend(
        frameon=False,
        prop={"family": "Times New Roman", "size": 10},
    )
    plt.grid(False)
    plt.xticks(fontname="Times New Roman")
    plt.yticks(fontname="Times New Roman")
    plt.tight_layout()
    plt.savefig(
        "xgb_optimized_fit_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # Save prediction results
    df_plot = pd.DataFrame(
        {
            "set": ["train"] * len(y_train)
            + ["test"] * len(y_test),
            "true": np.concatenate([y_train, y_test]),
            "predicted": np.concatenate(
                [y_train_pred, y_test_pred]
            ),
        }
    )
    df_plot.to_csv(
        "xgb_optimized_fit_result.csv",
        index=False,
    )
    print(
        "Prediction data saved to xgb_optimized_fit_result.csv"
    )


if __name__ == "__main__":
    main("normalized.csv")
