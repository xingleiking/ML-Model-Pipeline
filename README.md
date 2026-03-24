<img width="827" height="337" alt="1" src="https://github.com/user-attachments/assets/abb61b6e-78b8-473d-aa0f-c3155d6e3eb9" />A complete ML workflow container integrating model training, evaluation, and prediction
## Workflow
<img width="827" height="337" alt="1" src="https://github.com/user-attachments/assets/75a1e6e8-a8fb-4b9b-84ad-a0b80ad0cf14" />


### 1. Regression Model Selection
- Normalize the dataset using `Normalization.py`
- Evaluate regression models with `Model_Evaluation.py`

### 2. XGBoost Hyperparameter Optimization
- Perform grid search for optimal parameters using `XGBoost_Hyperparameter_Optimization.py`

### 3. Round 1 Model Training
- Train the model with optimized parameters using `Train_Model.py`

### 4. Round 1 Double-Mutation Prediction
- Predict double mutations using the trained model via `Predict_Model.py`

### 5. Round 2 Model Training
- Add validated double mutations to the dataset
- Retrain the model using `Train_Model.py`

### 6. Round 2 Triple-Mutation Prediction
- Perform triple-mutation prediction using the newly trained model `Triplet_predict_model.py`
