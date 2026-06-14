import os
import joblib
import pandas as pd

# Path to model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "predict_freight_model.pkl")


def load_model(model_path: str = MODEL_PATH):
    model = joblib.load(model_path)
    return model


def predict_freight_cost(input_data):
    model = load_model()

    if isinstance(input_data, dict):
        input_df = pd.DataFrame(input_data)
    else:
        input_df = input_data

    # Reorder columns
    expected_features = model.feature_names_in_
    input_df = input_df[expected_features]

    # Predictions
    predictions = model.predict(input_df)

    # Create output table
    output_df = input_df.copy()
    output_df["Predicted_Freight_Cost"] = predictions

    return output_df


# Example run
if __name__ == "__main__":
    sample_data = {
        "Dollars": [18500, 9000, 3000, 200],
        "Quantity": [10, 5, 2, 1]
    }

    result = predict_freight_cost(sample_data)

    print("\nPrediction Output:\n")
    print(result)