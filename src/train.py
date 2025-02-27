import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import os

from preprocess import load_data  # Import updated data processing

# Load & Preprocess Data
X_train, X_test, y_train, y_test = load_data("data/processed.cleveland.data")

# Start an MLflow experiment
mlflow.set_experiment("HeartDiseaseExperiment")

def train_random_forest():
    """ Train and log a Random Forest model """
    with mlflow.start_run(run_name="RandomForest"):
        # Hyperparameters
        n_estimators = 100
        max_depth = 5
        
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Train model
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        rf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = rf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc", auc)
        
        # Log model
        mlflow.sklearn.log_model(rf, "model")
        
        print(f"RandomForest -> Accuracy: {acc:.4f}, AUC: {auc:.4f}")

def train_xgboost():
    """ Train and log an XGBoost model """
    with mlflow.start_run(run_name="XGBoost"):
        # Hyperparameters
        n_estimators = 200
        max_depth = 4
        learning_rate = 0.1
        
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        
        # Train model
        xg_clf = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)
        xg_clf.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = xg_clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc", auc)
        
        # Log model
        mlflow.xgboost.log_model(xg_clf, "model")
        
        print(f"XGBoost -> Accuracy: {acc:.4f}, AUC: {auc:.4f}")

def train_neural_network():
    """ Train and log a Neural Network (MLP) """
    with mlflow.start_run(run_name="NeuralNetwork"):
        mlflow.log_param("model_type", "NeuralNetwork")
        
        # Define model architecture
        model = keras.Sequential([
            layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Train model
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=16, verbose=0)
        
        # Evaluate model
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc", auc)
        
        # Save the Keras model to MLflow
        mlflow.keras.log_model(model, "model")

        # Save the best model
        model_save_path = "models/best_model.keras"
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)

        print(f"NeuralNetwork -> Accuracy: {acc:.4f}, AUC: {auc:.4f}")

if __name__ == "__main__":
    print("Training Random Forest...")
    train_random_forest()
    
    print("Training XGBoost...")
    train_xgboost()
    
    print("Training Neural Network...")
    train_neural_network()
    