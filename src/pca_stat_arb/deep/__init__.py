"""Deep-learning statistical arbitrage (after Guijarro-Ordonez, Pelger & Zanotti, 2021).

PCA residuals -> CNN + Transformer signal extractor -> portfolio weights trained to maximise
the Sharpe ratio after trading costs, evaluated with walk-forward retraining.
"""
