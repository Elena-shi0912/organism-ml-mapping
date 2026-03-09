# Organism Human-in-the-Loop Mapping

This project studies whether machine learning models can learn CLSI clinical mappings from microbiology identifiers and improve robustness to noisy inputs, while supporting a human-in-the-loop feedback workflow.

## Tasks
- Predict Reported Name
- Predict CLSI Breakpoint Group
- Predict Bacteria Gram Stain

## Inputs
- Species Code
- Species Full Name

## Methods
- Exact lookup baseline
- Normalized lookup baseline
- Character n-gram + Logistic Regression
- Character n-gram + XGBoost / Random Forest
- Streamlit feedback UI

## Structure
- `src/`: training, preprocessing, evaluation
- `app/`: Streamlit app
- `data/`: raw, processed, feedback
- `notebooks/`: exploratory work
