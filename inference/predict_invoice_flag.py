import os
import joblib
import pandas as pd

# Path to model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "predict_flag_invoice.pkl")


def load_model(model_path: str = MODEL_PATH):
    model = joblib.load(model_path)
    return model


def predict_invoice_flag(input_data):
    """
    input_data: dict or pandas DataFrame
    Example:
    {
        "amount": [1000, 5000],
        "country": [1, 0]
    }
    """

    model = load_model()

    # Convert to DataFrame
    if isinstance(input_data, dict):
        input_df = pd.DataFrame(input_data)
    else:
        input_df = input_data

    # Ensure correct feature order
    expected_features = model.feature_names_in_
    input_df = input_df[expected_features]

    # Predictions
    predictions = model.predict(input_df)

    # Optional: map 0/1 to labels
    output_df = input_df.copy()
    output_df["Predicted_Invoice_Flag"] = predictions

    return output_df


# Example run
if __name__ == "__main__":
    sample_data = {
        "invoice_quantity": [5, 10],
        "invoice_dollars": [1000, 5000],
        "Freight": [200, 400],
        "total_brands": [2, 3],
        "days_po_to_invoice": [10, 15],
        "total_item_quantity": [50, 100],
        "total_item_dollars": [1500, 6000]
    }

    result = predict_invoice_flag(sample_data)

    print("\nPrediction Output:\n")
    print(result.to_string(index=False))