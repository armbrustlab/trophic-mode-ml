# Trophic mode ml
Scripts for feature extraction, training, and model application.

## Contents
- param_feature_selection:
  - Hyperparameter selection via gridsearch for the 3 classifiers compared in the manuscript.
  - Feature selection using Mean Decrease in Accuracy.
- training_evaluation:
  - Script to perform cross-validation on resulting models.
- model
  - CLI to make predictions using either XGBoost or Random Forest. 

## Dependencies
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Keras==2.3.1
- Tensorflow==2.0.0
