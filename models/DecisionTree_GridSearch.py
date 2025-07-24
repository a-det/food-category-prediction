import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Set results directory path
RESULTS_DIR = "../results/DT_results_top6_gridsearch"
DATASET_PATH = "../data/nutrition_with_general_category_top6.csv"

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(DATASET_PATH)
print("\nDataset loaded. Displaying the first few rows:")
print(data.head())  # Check the structure of the dataset

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
print("\nTraining Decision Tree classifier...")
pipeline = Pipeline([("classifier", DecisionTreeClassifier(random_state=42))])

# Define parameter grid for GridSearchCV
param_grid = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [None, 10, 20, 30],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": [None, "sqrt", "log2"],
}

# Grid search
print("\nPerforming grid search for hyperparameter optimization...")
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=2,
)
grid_search.fit(X_train, y_train)
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")

# Use the best estimator for further evaluation
best_pipeline = grid_search.best_estimator_

# Cross-validation
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
    f.write(f"Best parameters: {grid_search.best_params_}\n")
    f.write(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}\n")
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
