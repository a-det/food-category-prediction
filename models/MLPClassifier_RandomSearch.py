import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Set results directory path
RESULTS_DIR = "../docs/MLP_results_top5_RU_randomsearch_run1"
DATASET_PATH = "../data/nutrition_with_general_category_top5_RU.csv"

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(DATASET_PATH, sep=",")
print("\nDataset loaded. Displaying the first few rows:")
print(data.head())

# Preprocess the data
print("\nPreprocessing data...")

# Extract features and target variable
X = data.drop("General_category", axis=1)
y = data["General_category"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
print("\nTraining MLP classifier...")

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "classifier",
            MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        ),
    ]
)

# Define parameter distributions for RandomizedSearchCV
param_distributions = {
    "classifier__hidden_layer_sizes": [(50,), (100,), (128, 64), (100, 50), (256, 128)],
    "classifier__activation": ["relu", "tanh"],
    "classifier__solver": ["adam", "sgd"],
    "classifier__alpha": [1e-4, 1e-3, 1e-2],
    "classifier__learning_rate": ["constant", "adaptive"],
    "classifier__max_iter": [300, 500, 1000],
}

# Randomized search
print("\nPerforming randomized search for hyperparameter optimization...")
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=20,  # Number of parameter settings sampled
    cv=5,  # More robust cross-validation
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
    random_state=42,
)
random_search.fit(X_train, y_train)
print(f"Best parameters found: {random_search.best_params_}")
print(f"Best cross-validation accuracy: {random_search.best_score_:.4f}")

# Use the best estimator for further evaluation
best_pipeline = random_search.best_estimator_

# Cross-validation with best pipeline
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring="accuracy")
cv_mean = cv_scores.mean()
cv_std = cv_scores.std()
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_mean:.4f}")
print(f"Standard deviation of CV accuracy: {cv_std:.4f}")

# Evaluate on test data
y_pred = best_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
classification_rep = classification_report(y_test, y_pred)
print(classification_rep)

# Save metrics to a text file
metrics_filename = os.path.join(RESULTS_DIR, "metrics.txt")
with open(metrics_filename, "w") as f:
    f.write(
        "Cross-validation scores: "
        + ", ".join([f"{score:.4f}" for score in cv_scores])
        + "\n"
    )
    f.write(f"Mean CV accuracy: {cv_mean:.4f}\n")
    f.write(f"Standard deviation of CV accuracy: {cv_std:.4f}\n\n")
    f.write(f"Best parameters: {random_search.best_params_}\n")
    f.write(f"Best cross-validation accuracy: {random_search.best_score_:.4f}\n")
    f.write(f"Test accuracy: {accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_rep)
print(f"Metrics saved to '{metrics_filename}'")

# Save the model
model_filename = os.path.join(RESULTS_DIR, "food_category_model.pkl")
joblib.dump(best_pipeline, model_filename)
print(f"\nModel saved as '{model_filename}'")


# Create a prediction function
def predict_food_category(nutrition_data):
    """
    Predict food category based on nutrition facts.

    Parameters:
    nutrition_data (dict or pd.DataFrame): Nutrition values

    Returns:
    str: Predicted food category
    """
    # Convert to DataFrame if input is a dict
    if isinstance(nutrition_data, dict):
        nutrition_data = pd.DataFrame([nutrition_data])

    # Ensure all required columns are present
    missing_cols = set(X.columns) - set(nutrition_data.columns)
    for col in missing_cols:
        nutrition_data[col] = 0

    # Use only the columns from training
    nutrition_data = nutrition_data[X.columns]

    # Make prediction
    prediction = best_pipeline.predict(nutrition_data)[0]

    return prediction


# Example usage
print("\nExample prediction:")
sample_data = X.iloc[0].to_dict()
predicted_category = predict_food_category(sample_data)
print(f"Predicted category: {predicted_category}")
print(f"Actual category: {y.iloc[0]}")


# Function to load the model and make predictions
def load_model_and_predict(nutrition_data):
    """
    Load the saved model and make a prediction.

    Parameters:
    nutrition_data (dict): Dictionary containing nutrition values

    Returns:
    str: Predicted food category
    """
    loaded_model = joblib.load(os.path.join(RESULTS_DIR, "food_category_model.pkl"))

    # Convert to DataFrame
    nutrition_df = pd.DataFrame([nutrition_data])

    # Ensure all columns from training are present
    missing_cols = set(X.columns) - set(nutrition_df.columns)
    for col in missing_cols:
        nutrition_df[col] = 0

    # Use only the columns from training
    nutrition_df = nutrition_df[X.columns]

    # Make prediction
    prediction = loaded_model.predict(nutrition_df)[0]
    return prediction
