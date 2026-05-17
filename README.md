# Part 1: Neural Network Fundamentals and Training Behavior Analysis

This repository contains the neural network implementation and core analytical experiments designed to evaluate customer churn trends using structured corporate records.

## Dataset Profile & Exploratory Insights
* **Observations (Rows):** 2,000  
* **Features (Columns):** 17 total (1 ID tracking attribute, 4 distinct categorical dimensions, 11 continuous/discrete numerical features, and 1 binary class indicator).  
* **Missing Target Densities:** Verification confirmed 0 null values across all parameters; data integrity was completely maintained.  
* **Target (Churn Distribution):**
  * `0` (Retained Customers): 1,969 instances (98.45%)
  * `1` (Churned Customers): 31 instances (1.55%)

**Critical Structural Challenge:** The target class exhibits severe operational imbalance. Standard loss formulations naturally bias models toward classifying every instance as a non-churning customer to hit a superficial 98.45% accuracy score. This problem requires specialized stratification across validation subsets to avoid empty sample spaces for positive labels.

---

## Preprocessing & Data Pipeline Steps
1. **Feature Exclusion:** Removed `customer_id` from modeling arrays to eliminate non-predictive identifier bias.
2. **Categorical Processing:** One-hot encoded `region`, `plan_type`, `contract_type`, and `payment_method` dimensions, leveraging a drop-first strategy to minimize multicollinearity.
3. **Data Splitting:** Applied a 20% validation split, utilizing a stratified sequence on the target to preserve structural representation across training and testing boundaries.
4. **Feature Uniformity:** Fitted a `StandardScaler` strictly across training distributions and carried projections forward over validation matrices to eliminate structural leakage.

---

## Hyperparameter Experimentation & Metrics Table

| Experiment Run Config | Train Loss | Train Acc | Test Loss | Test Acc | False Positives | True Positives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp 1: Baseline** (16 Nodes, ReLU, LR=0.001, Batch=32, Epochs=20) | 0.0654 | 0.9844 | 0.0678 | 0.9850 | 0 | 0 |
| **Exp 2: Deeper** (32->16 Nodes, ReLU, LR=0.005, Batch=64, Epochs=30) | 0.0012 | 1.0000 | 0.1573 | 0.9825 | 1 | 0 |
| **Exp 3: Alternative** (16 Nodes, Tanh, LR=0.0005, Batch=16, Epochs=15) | 0.0831 | 0.9844 | 0.0811 | 0.9850 | 0 | 0 |

---

## Technical Reflection & Architectural Observations

### What specific roles do weights and biases serve inside this modeling architecture?
Weights act as directional tuning dials that scale incoming signals to rank features by predictive value. Biases function as shift thresholds, allowing activation barriers to slide independently of raw multiplier values. Together, they shift and rotate coordinate space, letting the network map cross-parameter relationships.

### Why is an explicit activation function mandatory in hidden layer steps?
Without activation layers, stacking layers collapses into a single matrix calculation. The network would be limited to solving basic, linear equations. Nonlinear functions (like ReLU or Tanh) allow the network to handle complex real-world data patterns, such as sharp thresholds or non-linear customer churn habits.

### What behavior manifests if the learning rate parameter is set too high vs. too low?
When set too low (seen in Exp 3), loss tracking stalls, requiring excessive computational time to reach optimal states. When set too high (seen in Exp 2), the model over-adjusts, skipping past ideal weights and causing the validation loss to spike (0.1573).

### Did the structural models show evidence of underfitting or severe overfitting?
* **Exp 1 and Exp 3** remain locked at baseline majority-class limits. They underfit the minority churn class because the loss function finds no mathematical incentive to look past the 98.45% baseline accuracy without explicit class-weighting.
* **Exp 2** displays severe, classic overfitting. The network collapsed training loss down to 0.0012 with 100% accuracy by memorizing specific data profiles. However, it completely failed on validation sets, spiking validation loss to 0.1573 and predicting false alarms while missing every actual churn instance.