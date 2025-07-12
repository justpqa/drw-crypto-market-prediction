This is my repo for the DRW Crypto Market Prediction competition in [Kaggle](https://www.kaggle.com/competitions/drw-crypto-market-prediction/overview). Details on my approach will be updated later after the end of the competition.

Current note on approaches:
- EDA & Feature Engineering: 
    + Only used top 100-200 feature correlated with label the most, along with some popular features (bid, ask, buy, sell qty, volume, order flow imbalance, etc.), later have to remove extra popular features like order flow imbalance or volume weighted buy due to the old popular features were normalized in new data => these features lose meaning and less important
    + Clustering features to find out that top features correlated with label often come together in the same cluster => strong in predicting
    + Clustering data points based on popular features => might build specific models for each scenario since clustering did separate between data points with very high/very low label and other
    + What did not work: incremental PCA to identify what feature contribute to variability (about 100 features takes up 80% variance), analyze the lagging of popular features (and newly created features) to find out what are the anonymized features are about
- Modeling process:
    + Starting with GBDT model (XGBoost > LightGBM > CatBoost, not really use the last one comparing to first 2) on (only anonymized, anonymized + popular => only anonymized perform better)
    + Utilize SHAP to identify top features + experimenting to found that working with top 30 features chosen by GBDT work the best
    + Train XGBoost and LightGBM on new features, all part of of process were optimized with Optuna (Bayesian Optimization wuth TPE Sampling)
    + Train MLP model using MLX on the top features => work much better than XGBoost and LightGBM without training a lot (batch size: fill each 30 min/1 hour/2 hours/12 hours/etc., only shuffle after batching), so that the model update based on data in the time-forward manner
    + (Currently working on) Try the supervised AE-MLP approach
- Prediction process:
    + Ensemble of 2 GBDT + MLP
    + Train at different time slices of different size in the train data in hoping that the model can learn different trends overtime in crypto market since crypto market changes rapidly and can have different regime (boost CV by 0.015-0.02)
