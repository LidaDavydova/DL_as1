# train_wine_model.py
import os
import joblib
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import mlflow.exceptions

# Load the Wine 
TARGET = "Wine"
FEATURES = ["Alcohol","Malic.acid","Color.int","Hue","Proline"]



BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE = os.path.join(BASE_DIR, "../../data/processed/train_data.csv")
TEST_FILE = os.path.join(BASE_DIR, "../../data/processed/test_data.csv")

train = pd.read_csv(TRAIN_FILE)
test  = pd.read_csv(TEST_FILE)

X_train = train[FEATURES]
y_train = train[TARGET]
X_test  = test[FEATURES]
y_test  = test[TARGET]

# Create a pipeline with scaling + Logistic Regression
params = {
    'max_iter': 20, 
    'solver': "lbfgs", 
    'multi_class': "auto"
}
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(**params)
)

# Train the model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="macro")
recall = recall_score(y_test, y_pred, average="macro")
f1 = f1_score(y_test, y_pred, average="macro")
print(acc, precision, recall, f1)

experiment_name = "MLflow experiment 01"
run_name = "run 01"
try:
    # Create a new MLflow Experiment
    experiment_id = mlflow.create_experiment(name=experiment_name)
except mlflow.exceptions.MlflowException as e:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    print(experiment_id)
    
with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
    # Log the hyperparameters
    mlflow.log_params(params=params)
    # Log the performance metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    mlflow.log_metrics({"accuracy": acc,
                        "f1": f1
                        })
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic LR model for iris data")
    # Infer the model signature
    signature = infer_signature(X_test, y_test)
    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="wine_model",
        signature=signature,
        input_example=X_test,
        registered_model_name="LR_model_01",
        pyfunc_predict_fn = "predict_proba"
    )

    sk_pyfunc = mlflow.sklearn.load_model(model_uri=model_info.model_uri)
    predictions = sk_pyfunc.predict(X_test)
    print(predictions)
    eval_data = pd.DataFrame(y_test)
    eval_data.columns = ["label"]
    eval_data["predictions"] = predictions
    results = mlflow.evaluate(
        data=eval_data,
        model_type="classifier",
        targets= "label",
        predictions="predictions",
        evaluators = ["default"]
    )
    print(f"metrics:\\n{results.metrics}")
    print(f"artifacts:\\n{results.artifacts}")
                            

# Save the model
model_path = "models/model.pkl"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"Model saved to {model_path}") 


