# Experiment Comparison Report

## Side-by-Side Experiment Comparison

### 1. Overview

| Attribute                  | Dummy Experiment 3            | Test Efficient Det        |
|----------------------------|-------------------------------|---------------------------|
| **Experiment Name**        | Dummy Experiment 3            | Test Efficient Det        |
| **Model Version**          | YoloX                         | EfficientDet              |
| **Datasets Used**          | Train, Val, Test              | Train, Val, Test          |
| **Created By**             | thibaut                       | thibaut                   |

---

### 2. Key Training Parameters

| Parameter                  | Dummy Experiment 3            | Test Efficient Det        |
|----------------------------|-------------------------------|---------------------------|
| **Epochs**                 | 300                           | 150                       |
| **Batch Size**             | 8                             | 16                        |
| **Learning Rate**          | 0.0001                        | 0.001                     |
| **Image Size**             | 640                           | 512                       |
| **Data Augmentations**     | None (weather transform off)  | Enabled                   |

---

### 3. Training Metrics

#### Loss Trends (Train)

| Metric        | Dummy Experiment 3     | Test Efficient Det |
|---------------|------------------------|---------------------|
| Classification| Initial: 2.43, Final: 1.39 | Initial: 2.10, Final: 1.25 |
| Confidence    | Initial: 4.04, Final: 1.57 | Initial: 3.90, Final: 1.40 |
| IoU           | Initial: 2.03, Final: 0.87 | Initial: 1.95, Final: 0.72 |
| Total Loss    | Initial: 8.50, Final: 4.48 | Initial: 8.00, Final: 3.37 |


#### Validation Metrics (AP/Recall Trends)

| Epoch   | Dummy Experiment 3 (AP50-95) | Test Efficient Det (AP50-95) |
|---------|-------------------------------|-------------------------------|
| 1       | 0.14                          | 0.18                          |
| 50      | 0.23                          | 0.35                          |
| 100     | 0.22                          | 0.37                          |
| Final   | 0.22                          | 0.40                          |


---

### 4. Performance Metrics (Final)

| Class   | AP (Exp. 3) | AP (Exp. 2) | AR (Exp. 3) | AR (Exp. 2) | TP (Exp. 3) | FP (Exp. 3) | FN (Exp. 3) | TP (Exp. 2) | FP (Exp. 2) | FN (Exp. 2) |
|---------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Car     | 0.21        | 0.32        | 1.0         | 0.98        | 7           | 26          | 0           | 10          | 20          | 0           |
| Person  | 0.20        | 0.29        | 1.0         | 0.95        | 15          | 59          | 0           | 20          | 48          | 0           |
| Tree    | 0.63        | 0.67        | 1.0         | 0.97        | 15          | 9           | 0           | 18          | 8           | 0           |

---

### Insights

- **Test Efficient Det shows better overall AP (Average Precision) across all classes**. Particularly, it improved for `Car` and `Person` classes.
- **Recall (AR)** remained high for both experiments, indicating effective detection coverage.
- **Loss Reduction**: Test Efficient Det reduced losses more effectively across all components, especially IoU.
- **False Positives**: Test Efficient Det slightly reduced false positives, particularly for the `Car` and `Person` classes.

---

### Recommendations

1. **Use EfficientDet as the baseline for future experiments**, given its superior performance metrics.
2. Optimize further by focusing on underperforming classes like `Person` and fine-tuning augmentations and data strategies.
3. Address **False Positive Rates** through stricter confidence thresholds.

---

### Visualizations

#### Loss Trends

![Loss Trends](https://via.placeholder.com/800x400?text=Loss+Trends+Chart)

#### Validation Metrics

![Validation Metrics](https://via.placeholder.com/800x400?text=Validation+Metrics+Chart)

#### Class-Wise AP and AR

![Class-Wise Performance](https://via.placeholder.com/800x400?text=Class-Wise+AP+and+AR+Chart)

---

Let me know if you want to explore specific metrics or include additional visualizations!
