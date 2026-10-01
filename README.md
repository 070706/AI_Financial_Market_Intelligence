# AI Financial Market Intelligence

An AI-powered financial market intelligence system that combines Apache Spark, Neo4j, Graph Neural Networks (GNN), PyTorch Geometric, and Power BI for stock-market analysis and direction prediction.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PySpark](https://img.shields.io/badge/PySpark-4.2.0-orange)
![Neo4j](https://img.shields.io/badge/Neo4j-GraphDatabase-green)
![PyTorch](https://img.shields.io/badge/PyTorch-GNN-red)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)

## Project Highlights

- Large-scale stock-market data processing using Apache Spark
- Financial correlation graph built using Neo4j
- Graph Neural Network using PyTorch Geometric
- Enhanced financial features for stock-direction prediction
- Chronological train, validation, and test split
- Power BI dashboard for prediction analysis
- Automated Python syntax checking using GitHub Actions

---
# AI-Powered Financial Market Intelligence Using Graph Neural Networks and Big Data

## Project Overview

This project builds an AI-powered financial market intelligence system using Big Data processing, financial correlation graphs, Graph Neural Networks, Neo4j, Apache Spark, PyTorch Geometric, and Power BI.

The system processes historical S&P 500 stock data, calculates stock returns, identifies relationships between companies, constructs a financial graph, trains a Graph Neural Network, predicts next-day stock direction, and visualizes the results using Power BI.

## Technologies Used

- Python
- Apache Spark / PySpark
- Neo4j
- PyTorch
- PyTorch Geometric
- Pandas
- NumPy
- Power BI
- VS Code

## Dataset

Dataset:

`SP500_Historical_Data.csv`

Main columns:

- Ticker
- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

`Adj Close` is used for return calculation.

## Project Pipeline

```text
S&P 500 Historical Data
          ↓
Apache Spark
          ↓
Data Preprocessing
          ↓
Daily Stock Returns
          ↓
Training-Period Correlations
          ↓
Neo4j Financial Graph
          ↓
Enhanced Feature Engineering
          ↓
PyTorch Geometric
          ↓
GraphSAGE GNN
          ↓
Next-Day Direction Prediction
          ↓
Evaluation
          ↓
Prediction CSV
          ↓
Power BI Dashboard 
## GNN Results

### Enhanced GNN Test Performance

| Metric | Result |
|---|---:|
| Test Accuracy | 50.86% |
| Precision | 0.5165 |
| Recall | 0.7700 |
| F1 Score | 0.6183 |

### Confusion Matrix

| | Predicted DOWN | Predicted UP |
|---|---:|---:|
| Actual DOWN | 2,836 | 9,550 |
| Actual UP | 3,048 | 10,202 |

### Validation Performance

Best validation accuracy:

**52.73%**

The model uses a GraphSAGE architecture with enhanced financial features including:

- Volume
- Current Return
- Volatility
- Momentum
- Moving Average
- Volume Change

> **Disclaimer:** Model predictions are experimental and are not guaranteed financial outcomes or investment advice.