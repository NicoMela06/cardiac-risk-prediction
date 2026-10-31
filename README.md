# Post-Myocardial Infarction Complication Prediction

## Executive Summary

Predicting complications following a myocardial infarction (MI) is crucial for proactive patient care and hospital resource optimization. However, clinical datasets often present a unique machine learning challenge: they are moderately sized, highly dimensional tabular data, making complex algorithms highly prone to severe overfitting.

This project evaluates the tension between algorithmic complexity and constrained data volume. To establish a rigorous mathematical baseline, a Logistic Regression model was derived and implemented entirely from scratch using NumPy. Its performance was then systematically compared against hyperparameter-optimized Multi-Layer Perceptron (MLP) and Random Forest architectures.

Full Research Paper: [Read the ACM-formatted Final Report (PDF)](Machine%20Learning%20Project%20(1).pdf)

## Technical Highlights

* From-Scratch Implementation: Manually coded a Logistic Regression algorithm using batch gradient descent to minimize binary cross-entropy (log-loss), completely independent of high-level ML predictive libraries.

* Overfitting Mitigation: Designed a generalized Neural Network pipeline utilizing early stopping mechanisms and heavy L2 regularization penalties to combat the extreme overfitting propensity of deep learning on small tabular datasets.

* Hyperparameter Tuning: Conducted exhaustive Grid Search optimization using stratified 5-fold cross-validation to ensure equitable and stable model comparisons.

## Dataset Overview

* Volume: 1,700 patient records.

* Features: 105 clinical and biometric predictors (after removing features with >30% missing values and applying statistically robust median imputation).

* Target: A unified binary classification target (1 = Presence of post-MI complications, 0 = Total absence).

## Methodology & Models

### 1. Custom Logistic Regression (Baseline)

* Stack: NumPy.

* Mechanics: Mathematical dot product of the feature vector and weight matrix, mapped through a sigmoid activation function. Converged over 2,000 iterations tracking manual log-loss.

### 2. Multi-Layer Perceptron (Neural Network)

* Stack: Scikit-Learn (MLPClassifier).

* Mechanics: Hidden layers with ReLU activations, optimized via Adam solver. Heavily regularized to force the network to learn generalized representations rather than memorizing the limited cohort.

### 3. Random Forest Classifier (Ensemble)

* Stack: Scikit-Learn (RandomForestClassifier).

* Mechanics: Bootstrap aggregating (bagging) and stochastic feature selection to natively model complex, non-linear biological interactions without explicit feature engineering.

## Results & Visualizations

The Random Forest ensemble proved to be the optimal architecture, successfully capturing complex clinical interactions without overfitting, achieving an Accuracy of 82.0% and an AUC of 0.902. Notably, the regularized MLP essentially collapsed into a generalized linear approximation, performing virtually identically to the custom-built mathematical baseline.

### Custom Logistic Regression: Convergence

![Convergence](Gradient_descent.png)

Textbook mathematical convergence of the custom-built gradient descent optimizer, minimizing binary cross-entropy.

### Model Performance Comparison

![Performance](ROC.png)

Stratified 5-fold CV F1-Scores demonstrating Random Forest
