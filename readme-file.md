# Housing Price Prediction Model

## Overview
This project implements and compares various machine learning algorithms to predict housing prices based on the Ames Housing dataset. The model development process includes exploratory data analysis, feature engineering, model selection, hyperparameter optimization, and evaluation. The final optimized model can be used to predict house prices based on various property characteristics.

## Table of Contents
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Feature Engineering](#feature-engineering)
- [Models Implemented](#models-implemented)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [License](#license)

## Project Structure
```
housing-price-prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebooks/
│   └── model_development.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
│
├── results/
│   └── results.json
│
├── README.md
└── requirements.txt
```

## Dataset
The project uses the Ames Housing dataset, which contains comprehensive information about house sales in Ames, Iowa, with 79 explanatory variables. The dataset includes:

- Categorical features (e.g., neighborhood, house style, zoning)
- Numerical features (e.g., square footage, lot area, year built)
- Ordinal features (e.g., overall quality, condition ratings)

Key features include:
- Property types and styles
- Size and area measurements
- Quality and condition ratings
- Location details
- Building materials
- Amenities and special features

## Methodology
The project follows a comprehensive machine learning workflow:

1. **Data Exploration and Analysis**: Analyzing distributions, correlations, and mutual information scores to understand feature importance
2. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features
3. **Feature Engineering**: Creating new informative features to improve model performance
4. **Model Selection**: Training and comparing multiple machine learning algorithms
5. **Hyperparameter Optimization**: Using Optuna for automated hyperparameter tuning
6. **Model Evaluation**: Using RMSE as the primary metric for model comparison

## Feature Engineering
Several engineered features were created to improve model performance:

- **log_lotArea**: Log transformation of lot area to handle skewed distribution
- **CarSpace**: Interaction feature between garage area and capacity
- **LivLotRatio**: Ratio of living area to lot area
- **Spaciousness**: Average room size
- **TotalOutsideSF**: Combined outdoor space features
- **GrLivArea_Neigh**: Interaction between living area and neighborhood
- **OverallQual_Neigh**: Interaction between quality and neighborhood
- **CombinedCondition**: Composite feature from different condition variables

## Models Implemented
The project compares several machine learning models:

- **Linear Models**:
  - Logistic Regression
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)

- **Tree-Based Models**:
  - Decision Trees
  - Random Forest
  - Gradient Boosting
  - TensorFlow Decision Forests
  - CatBoost

Each model type is evaluated on both the original dataset and the feature-engineered dataset to determine the best combination.

## Hyperparameter Optimization
After identifying TensorFlow Decision Forest as the best-performing model, an extensive hyperparameter optimization was performed using Optuna with a three-phase approach:

1. **Broad Exploration**: Random sampling was used for initial exploration of the hyperparameter space with 50 trials
2. **Focused Search**: Tree-structured Parzen Estimator (TPE) was applied for more efficient optimization with 100 trials
3. **Final Fine-Tuning**: CMA-ES sampler was used for further refinement with 50 trials

The optimization process yielded these optimal parameters:
- Number of trees: 400 (reduced from initial 550 in earlier optimization)
- Maximum depth: 6 
- Minimum examples per leaf: 3
- Shrinkage rate (learning rate): 0.103
- Honest trees setting: False

This optimization improved model performance from 21,212 RMSE to 20,590 RMSE, representing a 2.9% improvement through hyperparameter tuning alone.

## Results
The comprehensive model evaluation revealed several key insights:

1. **Model Performance**: TensorFlow Decision Forest significantly outperformed all other models with an RMSE of 21,212 on the original dataset, followed by CatBoost (25,354 RMSE).

2. **Feature Engineering Impact**: Custom engineered features like `GrLivArea_Neigh` (interaction between living area and neighborhood) and `OverallQual_Neigh` (quality rating by neighborhood) showed strong predictive power, as evidenced in the mutual information scores.

3. **Model Comparison**: 
   - Tree-based models dramatically outperformed linear models
   - The original TensorFlow Decision Forest RMSE: 21,212
   - The optimized TensorFlow Decision Forest RMSE: 20,590
   - SVM and Gradient Boosting performed worst with RMSE > 65,000

4. **Key Predictors**: Based on mutual information analysis, the top predictors were:
   - Overall Quality (MI score: 0.58)
   - Neighborhood (MI score: 0.52)
   - Ground Living Area (MI score: 0.48)
   - OverallQual_Neigh interaction feature (MI score: 0.46)
   - CarSpace custom feature (MI score: 0.40)

5. **Hyperparameter Optimization**: The optimized TFDF model used:
   - 400 trees
   - Maximum depth of 6
   - Minimum 3 examples per leaf
   - Learning rate (shrinkage) of 0.103
   - Non-honest trees

## Installation
To set up this project, follow these steps:

```bash
# Clone the repository
git clone https://github.com/yourusername/housing-price-prediction.git
cd housing-price-prediction

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
To train and evaluate the models:

```python
# Example code to run the pipeline
from src.model_training import train_and_evaluate_models
from src.preprocessing import load_data

# Load data
X_train, X_test, y_train, y_test = load_data()

# Train and evaluate models
results = train_and_evaluate_models(models, X_train, X_test, y_train, y_test)
```

### Model Loading and Prediction
The project uses TensorFlow Decision Forests (TFDF) models saved in the SavedModel format, which requires specific handling when loading for prediction:

```python
import tensorflow as tf

# Load the TFDF model
loaded_model = tf.saved_model.load("final_tfdf_model")
predict_fn = loaded_model.signatures["serving_default"]

# Preprocess your test data
preprocessor = fullPreprocessor("TensorFlow Decision Forest", categorical_columns, numerical_columns, transformer2=True)
X_new_transformed = preprocessor.fit_transform(df_test)

# Create a DataFrame with the correct feature names
X_new_transformed = pd.DataFrame(
    X_new_transformed, 
    columns=[f'feat_{i}' for i in range(X_new_transformed.shape[1])]
)

# Convert each column to a tensor with the right name
input_dict = {}
for col in X_new_transformed.columns:
    input_dict[col] = tf.convert_to_tensor(X_new_transformed[col].values, dtype=tf.float32)

# Make predictions using the input dictionary
predictions = predict_fn(**input_dict)

# Extract the prediction values
output_key = list(predictions.keys())[0]  # Usually 'output_1'
prediction_values = predictions[output_key].numpy()

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame(prediction_values, columns=["SalePrice"])
```

Note: When working with TensorFlow models across different versions of Keras, you may need to adjust your approach as TensorFlow SavedModel compatibility can change between major versions. The above code is designed to work with Keras 3.x and TensorFlow 2.x.

## Future Improvements
Potential areas for enhancement:

- Implement more advanced ensemble techniques
- Explore deep learning approaches for feature extraction
- Add geospatial analysis using neighborhood data
- Create an interactive web application for real-time predictions
- Incorporate time-series analysis for market trend predictions

## License
This project is licensed under the MIT License - see the LICENSE file for details.
