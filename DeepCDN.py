import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tf.keras.layers import Input, Embedding, Bidirectional, Conv1D, MaxPooling1D # type: ignore
from tf.keras.layers import Dense, Concatenate, GlobalMaxPooling1D, Dropout, BatchNormalization # type: ignore
from tf.keras.preprocessing.text import Tokenizer# type: ignore
from tf.keras.preprocessing.sequence import pad_sequences # type: ignore
from tf.keras.optimizers import Adam # type: ignore
from tf.keras.regularizers import l2 # type: ignore
from tf.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler # type: ignore
import nltk
from nltk.corpus import stopwords
import re

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Read the dataset
file_path = 'DATASET.csv'
df = pd.read_csv(file_path, on_bad_lines='skip', engine='python')

print("Original shape:", df.shape)
print("Null values per column:", df.isnull().sum())

# Drop unnecessary columns and rows with nulls
df_clean = df.drop(columns=['description']).dropna().copy()
# print("Shape after cleaning:", df_clean.shape)

# Print shape and missing value summary
# print("Original Dataset Description (Before Cleaning)\n")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
# print("\n Null values per column:")
# print(df.isnull().sum())

# Print column types and memory usage
print("\nData Types and Memory Info:")
print(df.info())

# Print basic statistics for numeric columns
print("\n Statistical Summary (numeric columns):")
print(df.describe())

# Print categorical column summaries
print("\n Summary of Categorical Columns:")
print(df.describe(include=['object']))



# Feature Engineering
# ==================

# 1. Process numerical features - ENSURE ALL ARE NUMERIC TYPES
# Convert to appropriate numeric types
numerical_cols = ['categoryId', 'view_count', 'likes', 'dislikes', 'comment_count']
for col in numerical_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Drop any rows that have NaN after numeric conversion
df_clean = df_clean.dropna()
print(f"Shape after ensuring numeric types: {df_clean.shape}")

# Create new features
df_clean['like_ratio'] = df_clean['likes'] / (df_clean['likes'] + df_clean['dislikes'] + 1)
df_clean['engagement_ratio'] = (df_clean['likes'] + df_clean['dislikes'] + df_clean['comment_count']) / (df_clean['view_count'] + 1)
df_clean['comment_to_view_ratio'] = df_clean['comment_count'] / (df_clean['view_count'] + 1)
df_clean['title_length'] = df_clean['title'].str.len()
df_clean['tags_count'] = df_clean['tags'].str.count('\|') + 1
df_clean.info()

# 2. Process date features
df_clean['publishedAt'] = pd.to_datetime(df_clean['publishedAt'], errors='coerce', utc=True)
df_clean['trending_date'] = pd.to_datetime(df_clean['trending_date'], errors='coerce', utc=True)

# Drop rows with NaN date values
df_clean = df_clean.dropna(subset=['publishedAt', 'trending_date'])
print(f"Shape after date cleaning: {df_clean.shape}")

# Calculate days since published (relative to April 10, 2025)-
reference_date = pd.Timestamp('2025-04-10', tz='UTC')
df_clean['days_since_published'] = (reference_date - df_clean['publishedAt']).dt.days

# Calculate time to trend (days between published and trending)
df_clean['time_to_trend'] = (df_clean['trending_date'] - df_clean['publishedAt']).dt.days
# Fix any negative values (when trending date is before published date due to data issues)
df_clean['time_to_trend'] = df_clean['time_to_trend'].clip(lower=0)

# Check for and remove any NaN values that might have been introduced
df_clean = df_clean.dropna()
print(f"Final shape after all cleaning: {df_clean.shape}")

# 3. Categorical features - ONE HOT ENCODING
# Convert categoryId to string first to ensure proper categorical conversion
df_clean['categoryId'] = df_clean['categoryId'].astype(str)
category_dummies = pd.get_dummies(df_clean['categoryId'], prefix='cat')
df_clean = pd.concat([df_clean, category_dummies], axis=1)

# 4. Text preprocessing

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Check if text is string, if not return empty string
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove punctuation and convert to lowercase
    return ' '.join([word for word in text.split() if word not in stop_words])

df_clean['title_clean'] = df_clean['title'].apply(clean_text)
df_clean['tags_clean'] = df_clean['tags'].apply(clean_text)
df_clean['text_combined'] = df_clean['title_clean'] + ' ' + df_clean['tags_clean']

# Print shape and missing value summary
# print("Original Dataset Description (Before Cleaning)\n")
print(f"Shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
# print("\nNull values per column:")
# print(df.isnull().sum())

# Print column types and memory usage
print("\nData Types and Memory Info:")
print(df_clean.info())

# Print basic statistics for numeric columns
print("\n Statistical Summary (numeric columns):")
print(df_clean.describe())

# Print categorical column summaries
print("\n Summary of Categorical Columns:")
print(df_clean.describe(include=['object']))

# Feature selection and scaling
# Select features for the model
num_features = [
    'view_count', 'likes', 'dislikes', 'comment_count',
    'days_since_published', 'like_ratio', 'engagement_ratio',
    'comment_to_view_ratio', 'title_length', 'tags_count', 'time_to_trend'
]

# Add category dummy columns
cat_dummies = [col for col in df_clean.columns if col.startswith('cat_')]
selected_features = num_features + cat_dummies

# ENSURE ALL FEATURES ARE NUMERIC
# This step is crucial to prevent the ValueError
for col in selected_features:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Drop any rows with NaN after converting to numeric
df_clean = df_clean.dropna(subset=selected_features)
print(f"Shape after ensuring all features are numeric: {df_clean.shape}")

# Scale numerical features
scaler = StandardScaler()
df_clean[num_features] = scaler.fit_transform(df_clean[num_features])

#Text tokenization
max_words = 10000  # Vocabulary size
max_length = 100   # Sequence length

tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(df_clean['text_combined'])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(df_clean['text_combined'])
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Define target variable
# Ensure view_count is numeric when calculating median
df_clean['view_count'] = pd.to_numeric(df_clean['view_count'])
df_clean['is_popular'] = (df_clean['view_count'] > df_clean['view_count'].median()).astype(int)
target = df_clean['is_popular'].values

# Prepare data for model
X_num = df_clean[selected_features].values
X_text = padded_sequences
y = target.reshape(-1, 1)

# With this approach instead:
# First check the datatypes
print("Data types of selected features:")
for col in selected_features:
    print(f"{col}: {df_clean[col].dtype}")

# Force convert all columns to numeric
for col in selected_features:
    # Convert to float to handle all numeric cases
    df_clean[col] = df_clean[col].astype(float)

# Then create the arrays
X_num = df_clean[selected_features].values
# Now this should work
print(f"Any NaN in numerical features: {np.isnan(X_num).any()}")
# If there are NaNs, remove those rows
if np.isnan(X_num).any():
    print("Removing rows with NaN values...")
    valid_indices = ~np.isnan(X_num).any(axis=1)
    X_num = X_num[valid_indices]
    X_text = X_text[valid_indices]
    y = y[valid_indices]
    print(f"Shape after removing NaNs: X_num: {X_num.shape}, X_text: {X_text.shape}, y: {y.shape}")

# Print selected numerical feature names
print("Selected Numerical Features:")
print(selected_features)

# Print information about text features
print("\nText Feature Used for Tokenization:")
print("text_combined")

# Print target variable information
print("\nTarget Variable:")
print("is_popular")

print("\n=== Feature Summary ===")
print(f"Numerical Features Shape: X_num -> {X_num.shape}")
print(f"Text Features Shape: X_text -> {X_text.shape}")
print(f"Target Shape: y -> {y.shape}")
print(f"Numerical Features: {selected_features}")
print(f"Text Feature Used: 'text_combined'")
print(f"Target Variable: 'is_popular'")

# Split data with stratification(balanacing in both train and test dataset)
X_num_train, X_num_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_num, X_text, y, test_size=0.2, random_state=42, stratify=y
)

# Further split test into validation and test
X_num_val, X_num_test, X_text_val, X_text_test, y_val, y_test = train_test_split(
    X_num_test, X_text_test, y_test, test_size=0.5, random_state=42, stratify=y_test
)

print(f"Training set: {X_num_train.shape[0]}, Validation set: {X_num_val.shape[0]}, Test set: {X_num_test.shape[0]}")

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train.flatten()), y=y_train.flatten())
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("Class weights:", class_weight_dict)

# Model definition


def create_model(num_features, sequence_length, vocab_size):
    # Numerical features branch
    num_input = Input(shape=(num_features,), name='numerical_input')
    bn1 = BatchNormalization()(num_input)
    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(bn1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.3)(dense1)
    num_features_output = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(dense1)

    # Text features branch
    text_input = Input(shape=(sequence_length,), name='text_input')
    # Remove input_length from Embedding layer
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=100)(text_input)

    conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embedding_layer)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')(pool1)
    pool2 = GlobalMaxPooling1D()(conv2)

    text_features = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(pool2)
    text_features = Dropout(0.4)(text_features)

    # Combine branches
    combined = Concatenate()([num_features_output, text_features])
    combined = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(0.4)(combined)

    # Output layer
    output = Dense(1, activation='sigmoid')(combined)

    # Create model
    model = Model(inputs=[num_input, text_input], outputs=output)

    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    return model

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

checkpoint = ModelCheckpoint(
    'best_youtube_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Combine training and validation for cross-validation
X_num_train_val = np.vstack([X_num_train, X_num_val])
X_text_train_val = np.vstack([X_text_train, X_text_val])
y_train_val = np.vstack([y_train, y_val])

# Cross-validation
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
fold_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_num_train_val, y_train_val.flatten())):
    print(f"\nTraining fold {fold+1}/{k}")

    # Split data
    X_num_fold_train, X_num_fold_val = X_num_train_val[train_idx], X_num_train_val[val_idx]
    X_text_fold_train, X_text_fold_val = X_text_train_val[train_idx], X_text_train_val[val_idx]
    y_fold_train, y_fold_val = y_train_val[train_idx], y_train_val[val_idx]

    # Create model
    model = create_model(
        num_features=X_num_train.shape[1],
        sequence_length=max_length,
        vocab_size=max_words
    )

    # Train model
    history = model.fit(
        [X_num_fold_train, X_text_fold_train], y_fold_train,
        validation_data=([X_num_fold_val, X_text_fold_val], y_fold_val),
        epochs=20,
        batch_size=32,
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Save model for ensemble
    fold_models.append(model)

    # Evaluate
    _, accuracy, auc, precision, recall = model.evaluate([X_num_fold_val, X_text_fold_val], y_fold_val)
    print(f"Fold {fold+1} - Val Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")

# Final model training on all training data
final_model = create_model(
    num_features=X_num_train.shape[1],
    sequence_length=max_length,
    vocab_size=max_words
)

history = final_model.fit(
    [X_num_train_val, X_text_train_val], y_train_val,
    validation_data=([X_num_test, X_text_test], y_test),
    epochs=20,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Ensemble predictions
print("\nEnsemble model evaluation:")
ensemble_preds = np.zeros((X_num_test.shape[0], 1))
for model in fold_models:
    ensemble_preds += model.predict([X_num_test, X_text_test])
ensemble_preds /= len(fold_models)
ensemble_binary = (ensemble_preds > 0.5).astype(int)

# Calculate ensemble metrics
acc = accuracy_score(y_test, ensemble_binary)
auc = roc_auc_score(y_test, ensemble_preds)
prec = precision_score(y_test, ensemble_binary)
rec = recall_score(y_test, ensemble_binary)
f1 = f1_score(y_test, ensemble_binary)

print(f"Ensemble Test Accuracy: {acc:.4f}")
print(f"Ensemble Test AUC: {auc:.4f}")
print(f"Ensemble Test Precision: {prec:.4f}")
print(f"Ensemble Test Recall: {rec:.4f}")
print(f"Ensemble Test F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, ensemble_binary))

# Individual fold model evaluation
print("\nIndividual model evaluations:")
for i, model in enumerate(fold_models):
    _, acc, auc, prec, rec = model.evaluate([X_num_test, X_text_test], y_test, verbose=0)
    print(f"Model {i+1} - Test Accuracy: {acc:.4f}, AUC: {auc:.4f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error # Import necessary functions


# Assuming y_test and ensemble_preds are already defined
mae = mean_absolute_error(y_test, ensemble_preds)
mse = mean_squared_error(y_test, ensemble_preds)

print(f"Ensemble MAE: {mae:.4f}")
print(f"Ensemble MSE: {mse:.4f}")

from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix for the ensemble model
conf_matrix = confusion_matrix(y_test, ensemble_binary)

# Print the confusion matrix
print("\nEnsemble Confusion Matrix:")
print(conf_matrix)

# Optionally, you can also visualize the confusion matrix using a heatmap
import seaborn as sns
import matplotlib.pyplot as plt

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Ensemble Confusion Matrix')
# Save the figure
plt.savefig('ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate hits and misses
true_positives = np.sum((ensemble_binary == 1) & (y_test == 1))  # Cache Hit
false_negatives = np.sum((ensemble_binary == 0) & (y_test == 1))  # Cache Miss

# Total actual requests (i.e., ground truth = 1)
total_requests = np.sum(y_test == 1)

# Ratios
hit_ratio = true_positives / total_requests if total_requests != 0 else 0
miss_ratio = false_negatives / total_requests if total_requests != 0 else 0

# Print results
print(f"Cache Hit Count: {true_positives}")
print(f"Cache Miss Count: {false_negatives}")
print(f"Cache Hit Ratio: {hit_ratio:.4f}")
print(f"Cache Miss Ratio: {miss_ratio:.4f}")

import numpy as np
import matplotlib.pyplot as plt


# Flatten y_test to 1D array before plotting:
y_test_flat = y_test.flatten()  # <-- Add this line to define y_test_flat
ensemble_pred_class = ensemble_binary.flatten() # <-- Add this line to define ensemble_pred_class


plt.figure(figsize=(10, 6))
# Add jitter to make overlapping points more visible
jitter = np.random.normal(0, 0.02, size=len(y_test_flat))

plt.scatter(range(len(y_test_flat)), y_test_flat + jitter, color='blue', label='Actual', alpha=0.6)
plt.scatter(range(len(ensemble_pred_class)), ensemble_pred_class + jitter, color='red', label='Predicted', alpha=0.6)
plt.title("Actual vs Predicted Class Labels")
plt.xlabel("Sample Index")
plt.ylabel("Class Label")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, ensemble_preds)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig('ROC.png', dpi=300, bbox_inches='tight')
plt.show()