Implementation of the SWHFormer-based baseline used in our study for significant wave height estimation from radar data.

# SWHFormer Baseline Implementation

This repository provides the TensorFlow/Keras implementation of the SWHFormer-based comparison model used in our study.

## Model Description

The implementation follows the architectural specifications described in:

Yang and Huang (2024), *SWHFormer: A Vision Transformer for Significant Wave Height Estimation From Nautical Radar Images*.

The model includes:

- Patch embedding
- Class token
- Positional encoding
- 12-layer Transformer encoder
- Regression head for SWH prediction

## Input Adaptation

Since our dataset consists of radar image sequences, the input radar sequence is converted into a 3-channel temporal feature map:

- mean
- standard deviation
- frame difference

This representation allows the transformer architecture to capture temporal characteristics while maintaining compatibility with the original model design.

## Note

The raw radar dataset cannot be shared due to data privacy and institutional policy.

## Code


The repository contains the TensorFlow/Keras implementation of the SWHFormer-based baseline model used for comparison in our study.
