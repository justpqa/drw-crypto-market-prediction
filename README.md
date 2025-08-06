# 🧠 DRW Crypto Market Prediction – Kaggle Competition 🪙📈

This is my repository for the **[DRW Crypto Market Prediction](https://www.kaggle.com/competitions/drw-crypto-market-prediction/overview)** competition on Kaggle.  
🎯 **Currently ranked 13th out of 1,092 teams** on the **private leaderboard**.

---

## 🧮 Problem Statement

Participants must **develop models capable of predicting future crypto market price movements**, using only known market metrics & anonymous features, and without time information upon prediction

---

## 🔍 EDA & Feature Engineering

- 🔢 **Feature Selection**
  - Used the **top 100–200 features most correlated with the label**, plus common market metrics created later on:
    - `bid`, `ask`, `buy/sell qty`, `volume`, `order flow imbalance`, etc.
  - Later removed some “popular” features (like `order flow imbalance`, `volume-weighted buy`) due to **normalization** in newer data → these features **lost meaning** and became **less predictive**.

- 🧬 **Feature Clustering**
  - Clustering features showed that **highly correlated features often appear in the same cluster** → strong predictors.
  - Clustering **data points** based on popular features separated scenarios with **extreme labels** (very high/low) from the rest.
    → Suggests potential for **scenario-specific modeling**.

- 🔁 **Interaction Features**
  - Built interactions using top features selected by **GBDT models**.
  - Some new features showed **correlation with the label up to 0.09–0.1**, compared to ~0.05 for single features → **significant gain**.

- ❌ **What Didn't Work**
  - **Incremental PCA** to analyze feature importance → ~100 features explain ~80% of variance, but didn't help modeling.
  - **Lag analysis** on popular and engineered features to reverse-engineer anonymized ones → no consistent insight.

---

## 🧠 Modeling Process

- 🌲 **Gradient Boosted Decision Trees (GBDT)**
  - Started with **XGBoost > LightGBM
  - Compared performance across:
    - **Only anonymized features**
    - **Anonymized + popular features** (popular features is not as useful, but they might unravel information regarding market regime)
    - **Anonymized + popular features + engineered market features** (which later show that since the data is normalized before, these features lose its meaning and predictive power)
    - Result: **Only anonymized performed better**.
  - Used **SHAP** to identify top 30 most important features → best performance came from limiting to this set.
  - Identify interaction of these 30 features => new features to model => +0.02 to +0.04 CV Boost => use **anonymized with interaction features** model

- 🔥 **MLP (Multi-Layer Perceptron) with MLX**
  - Trained on the **same top features** as GBDT.
  - **Outperformed XGBoost and LightGBM**, even with **minimal tuning**.
  - **Batching strategy**:
    - Grouped data into batches by time (e.g., 30 min / 1 hr / 2 hr / 12 hr).
    - Only **shuffle after batching**, preserving **time-forward progression**.
  - Integrated interaction features into MLP with +0.05 to +0.1 in CV score but more unstable in LB => revert back to **only anonymized features** model
  - ⚠️ MLP's fold correlation was **less stable**, likely due to sensitivity in training process.

- 🧬 **Final Combination**
  - Ensemble of 2 models with equal weight
      - ✅ **GBDT** with interaction features  
      - ✅ **MLP** without interaction features
  - Each set of models trained on differen timeframes trained using 3 different random seeds
 
- ❌ **What Didn’t Work (Modeling)**
  - **Combining anonymized + popular features** often led to worse performance than using anonymized alone.
  - **Using too many features** (instead of selecting top ones) degraded performance due to noise and overfitting.
  - **Using the Supervised Autoencoder-MLP** to find non-linear latent features that could be powerful predictor

---

## ⚙️ Training & CV Strategy

- 🎯 **Objective**:  
  All models were trained using **Mean Squared Error (MSE)** directly as the loss function.

- 📦 **Hyperparameter Optimization**:  
  Tuning was done with **Optuna**, using **TPE (Tree-structured Parzen Estimator) sampler** for Bayesian Optimization.

- 🧪 **Cross-Validation Strategy**:  
  Employed a **rolling window CV** with:
  - `4 months` for training  
  - `1 month` gap (to prevent data leakage)  
  - `4 months` for testing  
  - Repeated for **4 folds**

  This setup was designed to **respect temporal structure** in the crypto market and improve generalization to unseen time periods.

---

## 📦 Prediction Process

- 🤝 **Final Ensemble = GBDT + MLP**
- 🕒 Trained models at **different time slices** using **different chunk sizes** in the training data:
  - Helps model **different market regimes** and capture **shifting crypto trends**
  - 🎯 Result: Boosted Pearson Correlation by **~0.015–0.02**

---

Thanks for checking out the repo! ⭐️ Feel free to follow or reach out with any questions.
