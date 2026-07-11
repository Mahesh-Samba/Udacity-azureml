# Optimizing a Machine Learning Model using HyperDrive and Automated Machine Learning (Azure ML SDK v1)

> **Important Note**
>
> The cloud lab environment provided for this project experienced persistent issues during execution (including notebook instability and AutoML SDK submission issues). To ensure successful completion of the project, the entire workflow was reproduced using my own Microsoft Azure subscription and Azure Machine Learning Workspace.
>
> All required Azure resources (Workspace, Compute Cluster, HyperDrive experiment, AutoML experiment, Registered Dataset, and Registered Models) were created in my own Azure subscription. After successful completion of the project, the Azure compute resources were deleted to avoid unnecessary cloud charges. Screenshots of the created resources and the compute cleanup process are included in the **Screenshots** section.

---

# Project Overview

This project demonstrates how to optimize a machine learning classification model using two different optimization approaches provided by Azure Machine Learning.

1. HyperDrive (Hyperparameter Tuning)
2. Automated Machine Learning (AutoML)

Both approaches were trained using the same Bank Marketing dataset and their performances were compared using the **Accuracy** metric.

The objective of this project is to identify which approach produces the best performing classification model.

---

# Pipeline Architecture

```
                    Local Machine
                          │
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
   config.json                        bank.csv
        │                                   │
        └──────────────┬────────────────────┘
                       │
             Azure ML Workspace
                       │
         ┌─────────────┼─────────────┐
         │             │             │
     Compute      Experiment     Dataset
      Cluster                     Registration
         │
         ├──────────────┐
         │              │
    HyperDrive       AutoML
         │              │
         └──────┬───────┘
                │
         Best Performing Model
                │
        Model Registration
```

The pipeline consists of the following stages:

- Dataset loading using pandas
- Dataset registration in Azure Machine Learning
- Baseline Logistic Regression model
- HyperDrive hyperparameter optimization
- AutoML model selection
- Model comparison
- Model registration

---

# Dataset

The original project references Microsoft's sample Bank Marketing dataset.

During implementation, the original dataset could not be used directly within the execution environment. Therefore, a compatible **Bank Marketing** dataset obtained from Kaggle was used.

The dataset was loaded locally using **pandas** and then registered as an Azure Machine Learning Tabular Dataset.

Dataset Characteristics:

- Number of Records: 11,162
- Number of Features: 16
- Target Variable: `deposit`
- Task: Binary Classification

The dataset was stored as:

```
bank.csv
```

and loaded using:

```python
df = pd.read_csv("bank.csv")
```

---

# Compute Cluster

Azure Machine Learning Compute Cluster was used to execute all experiments.

Configuration:

- Azure ML Compute Cluster
- CPU-based compute
- Shared between HyperDrive and AutoML
- Automatically scaled during execution

Both HyperDrive and AutoML experiments were executed on the same compute infrastructure to ensure a fair comparison.

---

# HyperDrive Configuration

## Classification Algorithm

The baseline model and HyperDrive experiment both used:

- Logistic Regression

## Hyperparameters Tuned

Two hyperparameters were optimized:

- Regularization Strength (`C`)
- Maximum Iterations (`max_iter`)

Search Space

| Parameter | Values |
|------------|-------------------------|
| C | 0.001, 0.01, 0.1, 1, 10 |
| max_iter | 100, 150, 200 |

---

## Parameter Sampling Strategy

Random Parameter Sampling was selected.

### Why Random Sampling?

Random sampling explores the hyperparameter space efficiently without evaluating every possible combination.

Advantages:

- Faster than Grid Search
- Better exploration for limited computational resources
- Recommended by Azure Machine Learning for most tuning scenarios

---

## Early Stopping Policy

Bandit Policy was selected.

Configuration:

- Slack Factor = 0.1
- Evaluation Interval = 1
- Delay Evaluation = 5

### Why Bandit Policy?

Bandit Policy terminates poorly performing runs early while allowing promising runs to continue.

Benefits:

- Reduces unnecessary compute time
- Lowers Azure compute cost
- Speeds up hyperparameter optimization

---

# AutoML Configuration

AutoML was configured as follows:

| Setting | Value |
|---------|--------|
| Task | Classification |
| Target Column | deposit |
| Primary Metric | Accuracy |
| Validation | Automatic |
| Compute Target | Azure ML Compute Cluster |
| Maximum Trials | 20 |
| Maximum Concurrent Trials | 4 |
| Experiment Timeout | 30 Minutes |
| Iteration Timeout | 10 Minutes |

## AutoML Generated Model

AutoML evaluated multiple machine learning algorithms and preprocessing pipelines.

Best Pipeline:

- MaxAbsScaler
- LightGBM

Primary Metric:

- Accuracy

Final Accuracy:

- **0.85630**

---

# Results

| Model | Algorithm | Accuracy |
|--------|-----------|---------:|
| Baseline | Logistic Regression | 0.8021 |
| HyperDrive | Logistic Regression (C=0.01, max_iter=150) | 0.8052 |
| AutoML | MaxAbsScaler + LightGBM | **0.85630** |

---

# Model Comparison

The baseline Logistic Regression model achieved an accuracy of **0.8021**. HyperDrive improved the baseline model by tuning the regularization strength and maximum iterations, increasing the accuracy to **0.8052**.

AutoML explored multiple algorithms and preprocessing pipelines instead of restricting the search to Logistic Regression. The best pipeline selected by AutoML combined **MaxAbsScaler** with **LightGBM**, producing the highest accuracy of **0.85630**.

This experiment demonstrates that while HyperDrive can improve the performance of a chosen algorithm through hyperparameter tuning, AutoML can further improve results by automatically selecting a more suitable algorithm and preprocessing pipeline for the dataset.

---

# NOTE regarding the RUNDETAILS widget
Due to a compatibility issue between the legacy Azure ML SDK v1 widget and the current Microsoft Azure ML managed notebook environment, the widget did not render, although the HyperDrive experiment completed successfully and all results were verified in Azure ML Studio.

Although the Azure ML SDK v1 RunDetails widget did not render in the current managed notebook environment, the HyperDrive experiment executed successfully. The HyperDrive parent run, all child runs, best child run, evaluation metrics, and selected best model were verified through Azure Machine Learning Studio. To provide equivalent evidence of the HyperDrive execution, screenshots of the HyperDrive parent run, child runs, best child run, and corresponding metrics have been included in the Screenshots section of this README.

# Future Improvements

Several improvements can further enhance the model performance.

- Explore a larger HyperDrive search space by evaluating additional values for the Logistic Regression hyperparameters. This increases the likelihood of identifying a better-performing configuration.

- Perform additional feature engineering, including creating new features and removing redundant ones. Better feature representation can improve model performance.

- Evaluate more advanced classification algorithms such as XGBoost, CatBoost, and ensemble techniques using HyperDrive.

- Increase the AutoML experiment duration and maximum number of trials. Allowing AutoML to evaluate more candidate models may lead to improved predictive performance.

- Deploy the best model as a real-time Azure ML endpoint and continuously monitor its performance using production data.

---

# Cleanup

To avoid unnecessary Azure cloud charges, all Azure Machine Learning compute resources created for this project were deleted after completing the experiments.

The cleanup process included:

- Deleting the Azure ML Compute Cluster
- Stopping active compute resources
- Verifying that no running jobs remained

Screenshots showing the compute resources before deletion and the cleanup process are included in the **Screenshots** section.

---

# Screenshots

The repository contains screenshots demonstrating the complete workflow.

- Azure ML Workspace
- Compute Cluster
- Workspace Connection
- Baseline Training
- HyperDrive Configuration
- HyperDrive Completed
- HyperDrive Best Run
- Registered Dataset
- AutoML Configuration
- AutoML Job
- AutoML Best Model
- Results Comparison
- Compute Cleanup
- HyperDrive Child Runs
- HyperDrive Best Run
- HyperDrive Metrics
- HyperDrive Experiment Runs

---

# Repository Structure

```
AzureML-HyperDrive-AutoML/
│
├── README.md
├── train.py
├── udacity-project.ipynb
├── bank.csv
├── config.json.example
│
├── screenshots/
│
└── architecture/
```