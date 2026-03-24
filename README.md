A complete ML workflow container integrating model training, evaluation, and prediction
## Workflow
<img width="3618" height="1473" alt="1" src="https://github.com/user-attachments/assets/d76b934c-2098-4573-b491-fa0ea217f22e" />



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
