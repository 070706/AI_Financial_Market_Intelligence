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
## Dashboard Preview

![AI Financial Market Intelligence Dashboard](dashboard/dashboard.png)

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
```

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

## GNN Architecture

The project uses a GraphSAGE-based Graph Neural Network.

```text
Input Features
      ↓
GraphSAGE Layer
      ↓
ReLU
      ↓
Dropout
      ↓
GraphSAGE Layer
      ↓
ReLU
      ↓
Dropout
      ↓
Output Layer
      ↓
DOWN / UP Prediction
```

### Input Features

The enhanced model uses 6 features:

1. Volume
2. Current Return
3. Volatility
4. Momentum
5. Moving Average
6. Volume Change

### Financial Graph

- Nodes represent companies.
- Edges represent correlations between companies.
- Correlations are calculated using the training period.
- 26 companies are included in the training graph.
- 46 correlation relationships are used.
- Edges are represented in both directions for graph message passing.

## Target Variable

The model predicts the next trading day's stock price direction.

```text
Next-Day Return > 0
        ↓
       UP

Next-Day Return ≤ 0
        ↓
      DOWN
```

## Data Splitting

The dataset is divided chronologically to avoid using future information during training.

```text
Historical Data
      ↓
Training Set
      ↓
Validation Set
      ↓
Test Set
```

The model uses:

- Training period: before 17-Apr-2018
- Validation period: 17-Apr-2018 to 15-Mar-2022
- Test period: from 16-Mar-2022

## Financial Graph

The financial graph is created using stock return correlations.

- Each company is represented as a graph node.
- A correlation edge connects companies with sufficiently strong relationships.
- Correlations are calculated using the training period.
- The training graph contains 26 companies.
- 46 correlation relationships are used.
- Each correlation relationship is represented in both directions for GNN message passing.

## Feature Engineering

Apache Spark is used for large-scale feature engineering.

The enhanced feature set contains:

| Feature | Description |
|---|---|
| Volume | Daily trading volume |
| Current Return | Daily stock return |
| Volatility | Rolling volatility of returns |
| Momentum | Return-based momentum feature |
| Moving Average | Rolling average return |
| Volume Change | Change in trading volume |

## Model Training

The project uses GraphSAGE layers to learn information from both company-level financial features and relationships between companies.

Training uses:

- GraphSAGE
- ReLU activation
- Dropout
- Weighted Cross-Entropy Loss
- Adam optimizer
- Learning rate: 0.001
- 30 training epochs

## Model Evaluation

The enhanced GNN achieved:

- Test Accuracy: **50.86%**
- Precision: **0.5165**
- Recall: **0.7700**
- F1 Score: **0.6183**
- Best Validation Accuracy: **52.73%**

The results indicate that predicting short-term stock direction from the selected historical features and correlation graph is a challenging task.

## Prediction Output

The model generates predictions containing:

- Date
- Ticker
- Prediction Label
- Down Probability
- Up Probability
- Confidence
- Signal Strength

Signal strength is categorized as:

```text
Confidence ≥ 60%
        ↓
     Strong

Confidence ≥ 55%
        ↓
    Moderate

Confidence < 55%
        ↓
      Weak
```

## Power BI Dashboard

The prediction results are exported to CSV and visualized using Power BI.

The dashboard contains:

- Total Predictions
- UP Predictions
- DOWN Predictions
- Average Confidence
- Prediction Distribution
- Average Prediction Confidence Over Time
- Average Confidence by Company
- Prediction Details Table
- Ticker Slicer
- Prediction Label Slicer
- Date Slicer

## Project Structure

```text
AI_Financial_Market_Intelligence/
│
├── README.md
├── LICENSE
├── .gitignore
├── .gitattributes
├── requirements.txt
│
├── .github/
│   └── workflows/
│       └── python-check.yml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── graph/
│
├── spark/
│   ├── preprocess.py
│   ├── calculate_returns.py
│   ├── calculate_pairwise_correlations.py
│   ├── calculate_training_correlations.py
│   ├── prepare_graph_data.py
│   ├── prepare_training_graph.py
│   ├── create_gnn_dataset.py
│   ├── create_gnn_features.py
│   ├── create_enhanced_gnn_features.py
│   └── create_training_node_features.py
│
├── neo4j/
│   ├── test_connection.py
│   └── load_graph.py
│
├── gnn/
│   ├── neo4j_to_pyg.py
│   ├── build_training_graph.py
│   ├── build_final_training_graph.py
│   ├── create_training_daily_graph.py
│   ├── create_enhanced_training_daily_graph.py
│   ├── normalize_training_graph.py
│   ├── normalize_enhanced_training_graph.py
│   ├── train_training_graph_gnn.py
│   ├── train_enhanced_training_graph_gnn.py
│   ├── evaluate_enhanced_training_graph_gnn.py
│   ├── generate_enhanced_predictions.py
│   ├── export_predictions_csv.py
│   ├── create_final_market_report.py
│   └── create_powerbi_dataset.py
│
├── models/
│
├── results/
│
├── notebooks/
│
└── dashboard/
    └── AI_Financial_Market_Intelligence.pbix
```

## Outputs

The project generates:

- Cleaned stock-market data
- Daily stock returns
- Stock correlation data
- Financial graph data
- PyTorch Geometric graph datasets
- Trained GNN model
- GNN evaluation results
- Stock direction predictions
- Final market intelligence report
- Power BI dataset
- Power BI dashboard

## Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Programming and ML development |
| Apache Spark | Large-scale data processing |
| Neo4j | Financial graph storage |
| PyTorch | Deep learning |
| PyTorch Geometric | Graph Neural Networks |
| Pandas | Data analysis |
| NumPy | Numerical processing |
| Power BI | Data visualization |
| GitHub Actions | Automated Python syntax checking |
| Git | Version control |

## Limitations

- Stock-market prediction is highly uncertain.
- The model is trained using historical market data.
- Historical relationships may change over time.
- The current model provides directional predictions rather than exact future prices.
- Test accuracy is close to random-level directional classification.
- Model predictions should not be considered financial advice.

## Future Improvements

Possible improvements include:

- LSTM or Transformer-based temporal modeling
- Graph Attention Networks
- Additional financial indicators
- Sector-level relationships
- News and sentiment analysis
- Market index features
- Hyperparameter optimization
- Longer historical training periods
- Ensemble models
- Real-time market data integration

## Conclusion

This project demonstrates an end-to-end financial market intelligence pipeline combining Big Data processing, financial graphs, Graph Neural Networks, and business intelligence.

Apache Spark handles large-scale financial data processing, Neo4j represents relationships between companies, PyTorch Geometric enables graph-based deep learning, and Power BI provides an interactive interface for analyzing model predictions.

> **Disclaimer:** This project is developed for educational and research purposes. The predictions are experimental and should not be used as financial or investment advice.