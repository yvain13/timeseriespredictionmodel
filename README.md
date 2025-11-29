# Time Series Prediction Model

Overview

This repository demonstrates using Chronos-2 for time series forecasting. The included notebook (Time_Series_chronos_2_forecasting.ipynb) shows how to use a pretrained Chronos-2 pipeline for inference and fine-tuning across several example tasks: univariate forecasting, forecasting with covariates (energy prices, retail sales, SIR/cyber-security incidents), cross-learning (joint prediction), and fine-tuning.

Features

- Examples for univariate and multivariate forecasting
- Support for past and known-future covariates
- Joint (cross-learning) prediction across series
- Fine-tuning API examples
- Visualization helpers to compare forecasts

Requirements

- Python 3.8+
- chronos-forecasting>=2.1
- pandas[pyarrow]
- matplotlib
- numpy

Installation

Install required packages:

pip install 'chronos-forecasting>=2.1' 'pandas[pyarrow]' 'matplotlib'

Quickstart

1. Load the Chronos-2 pipeline:

from chronos import BaseChronosPipeline, Chronos2Pipeline
pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")

2. Predict on a long-format pandas DataFrame:

pred_df = pipeline.predict_df(context_df, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])

predict_df (high-level arguments)

- df: long-format DataFrame with id, timestamp, target and optional past covariates
- future_df: optional DataFrame with known future covariates (columns present in both df and future_df are treated as known future covariates)
- id_column: default "item_id" (or use "id")
- timestamp_column: default "timestamp"
- target: target column name (default "target")
- prediction_length: horizon to forecast
- quantile_levels: list of quantiles to compute

Included examples

1) Univariate forecasting
- Example uses the M4 hourly dataset loaded from: https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv

2) Energy price forecasting (with covariates)
- Historical data: https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/train.parquet
- Future covariates: https://autogluon.s3.amazonaws.com/datasets/timeseries/electricity_price/test.parquet

3) Retail demand forecasting (with covariates)
- Training data: https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/train.parquet
- Test data: https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/test.parquet

4) SIR / Cyber-security incident prediction (domain adaptation example)
- The notebook references local CSV files for the SIR example. Add these files to the environment where you run the notebook:
  - "Train SIR DataSet - Sheet1.csv"
  - "Future Covariates - Sheet1-2.csv"

Cross-learning (joint prediction)

Enable cross-learning by passing predict_batches_jointly=True to predict_df. This allows the model to share information across series in a batch. Validate results for your task — cross-learning can help or hurt depending on data homogeneity and batch configuration.

Advanced API

- predict_quantiles: lower-level numpy/torch API. Accepts either a 3D array (batch, num_variates, history_length) for univariate/multivariate inputs without covariates, or a list of dicts for inputs with past/future covariates.
- fit: fine-tuning API that accepts inputs in the same dict/list format, with arguments such as prediction_length, num_steps, learning_rate, and batch_size. Returns a fine-tuned pipeline.

Notebook

- Time_Series_chronos_2_forecasting.ipynb — main notebook with runnable examples and visualizations. It includes links to open in Colab / SageMaker Studio Lab.

Data

- Public datasets are loaded from remote URLs (autogluon S3) in the notebook. For the SIR example, provide the local CSV files in the runtime environment.

License

This repository contains a LICENSE file at the repository root. See LICENSE for details.

Contributing

Issues, suggestions, and pull requests are welcome. When opening PRs, include reproducible examples and reference the relevant notebook cell or dataset.

Contact

For questions about the notebook or examples, open an issue in the repository.