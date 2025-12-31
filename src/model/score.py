import json
import os
import logging
import mlflow.pyfunc
import numpy as np

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global model variable
model = None

def init():
    """Initialize the model - called once when the container starts"""
    global model
    try:
        # Get the model directory from Azure ML
        model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'model')
        logger.info(f"Loading model from: {model_path}")
        
        # Load MLflow model
        model = mlflow.pyfunc.load_model(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def run(raw_data):
    """Run predictions - called for each request"""
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Extract the data array
        if 'data' in data:
            X = np.array(data['data'])
        else:
            X = np.array(data)
        
        logger.info(f"Input shape: {X.shape}")
        
        # Make predictions
        predictions = model.predict(X)
        
        # Convert to list for JSON serialization
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        
        logger.info(f"Predictions: {predictions}")
        
        # Return predictions
        return predictions
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": str(e), "input_received": raw_data}