A complete ML workflow container integrating model training, evaluation, and prediction
## Workflow
<img width="1521" height="1558" alt="Frame" src="https://github.com/user-attachments/assets/5b674dd6-55f3-4bb4-840e-f598c1acac2d" />

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
