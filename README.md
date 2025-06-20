# DeepCDN: Intelligent Cache Management for Content Delivery Networks

DeepCDN is a Python-based intelligent caching system that optimizes content delivery across distributed servers using machine learning techniques(DL). It aims to reduce latency, improve hit rates, and enhance bandwidth efficiency in modern CDNs (Content Delivery Networks).

By analyzing user request patterns and dynamically updating the cache using predictive models, DeepCDN intelligently decides which content to cache or evict — going beyond traditional caching strategies like LRU or LFU.
This project was developed as part of my M.Tech coursework in Deep Learning with a focus on real-world applications of Deep Learning techniques.

## Features

- **Intelligent Caching**: Uses predictive models to decide what content should be cached or replaced.
- **Machine Learning Integration**: Incorporates learning from past content request patterns to optimize future caching decisions.
- **Dynamic Cache Replacement**: Goes beyond static algorithms (e.g., LRU/LFU) using adaptive, data-driven strategies.
- **Cloud-Simulated Environment**: Stored and retrieved request data using Google Cloud Console for scalable data management(you can refer to youtube.py for script that automates the task of fetching the queries and storing it in Google cloud console).
- **Performance Metrics**: Tracks and logs cache hit ratio, latency reduction, and bandwidth savings over time.


## Technologies Used

###  Programming & Data Processing
- **Python** – Main programming language for data manipulation, modeling, and orchestration
- **pandas, NumPy** – For data cleaning, feature engineering, and numerical processing
- **re (Regex)** – Used for custom text cleaning operations
- **nltk** – Downloaded and used stopwords for title and tag preprocessing

### Data Collection & Storage
- **YouTube Data API v3** – Used to collect real-world metadata about video content (not shown in script, but referenced)
- **Google Cloud Console (GCC)** – Used to store and retrieve collected request metadata for scalable access

### Machine Learning & Deep Learning
- **scikit-learn** – Used for:
  - Train-test splitting with stratification
  - Cross-validation with `StratifiedKFold`
  - Feature scaling (`StandardScaler`)
  - Evaluation metrics (Accuracy, Precision, Recall, F1-score, AUC, Classification Report, Confusion Matrix)
  - Class weight balancing

- **TensorFlow & Keras (tf.keras)** – Used for building a deep learning model combining:
  - **Numerical Input Branch** with Dense layers and BatchNorm
  - **Text Input Branch** using `Embedding`, `Conv1D`, `MaxPooling1D`, and `GlobalMaxPooling1D`
  - Regularization (`Dropout`, `L2`)
  - Output through a sigmoid-activated Dense layer for binary classification

- **Callbacks**:
  - `EarlyStopping`
  - `ReduceLROnPlateau`
  - `ModelCheckpoint`

- **Tokenizer and pad_sequences** – For converting text features (title + tags) into fixed-length tokenized sequences

### Visualization & Reporting
- **matplotlib, seaborn** – Used to visualize:
  - Confusion matrix
  - Actual vs Predicted class labels
  - ROC Curve

## Running the Project

This project is written in **Python 3.8+** and uses common machine learning libraries like:

- `pandas`, `numpy`, `scikit-learn`
- `tensorflow` (via `tf.keras`)
- `nltk`, `matplotlib`, `seaborn`

To run the project:

---bash
python DeepCDN.py

## Results & Benchmarks

The DeepCDN model was trained using **5-fold cross-validation** and evaluated using an ensemble strategy. The goal was to predict whether a content request should be cached based on metadata, enabling intelligent cache decisions.

### Ensemble Model Performance (Test Set)

| Metric               | Value   |
|----------------------|---------|
| Accuracy             | 0.8500  |
| Precision            | 0.8352  |
| Recall               | 0.8720  |
| F1-Score             | 0.8532  |
| ROC AUC              | 0.9342  |
| MAE (Mean Abs Error) | 0.2654  |
| MSE (Mean Sq Error)  | 0.1122  |

---

### Classification Report

          precision    recall  f1-score   support

       0       0.87      0.83      0.85       250
       1       0.84      0.87      0.85       250

    accuracy   0.85       500



---

### Individual Model Evaluations (Cross-Validation Folds)

- Model 1 – Accuracy: 0.8200, AUC: 0.9123  
- Model 2 – Accuracy: 0.7860, AUC: 0.8997  
- Model 3 – Accuracy: 0.7760, AUC: 0.9208  
- Model 4 – Accuracy: 0.8280, AUC: 0.9214  
- Model 5 – Accuracy: 0.8380, AUC: 0.9196  

---

### Cache Effectiveness

| Metric             | Value  |
|--------------------|--------|
| Cache Hit Count    | 218    |
| Cache Miss Count   | 32     |
| Cache Hit Ratio    | 0.8720 |
| Cache Miss Ratio   | 0.1280 |

---

### Visual Outputs

- `Confusion_matrix.png` – Confusion matrix heatmap
- `ROC_curve.png` – Receiver Operating Characteristic (ROC) curve
- `Actual_VS_Predicted.png`-  Actual vs Predicted class scatter plot – Visual comparison of classification results
