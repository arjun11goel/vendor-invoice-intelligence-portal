import joblib
from pathlib import Path

from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)


def main():

    # Paths
    data_path = "../Inventory/vendor_invoice.csv"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Load Data
    df = load_vendor_invoice_data(data_path)

    # Prepare Features
    X, y = prepare_features(df)

    # Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train Models
    lr = train_linear_regression(X_train, y_train)
    dt = train_decision_tree(X_train, y_train)
    rf = train_random_forest(X_train, y_train)

    # Evaluate
    results = []
    results.append(evaluate_model(lr, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt, X_test, y_test, "Decision Tree"))
    results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))

    for r in results:
        print(r)

    # Select Best Model (Lowest MAE)
    best_model = min(results, key=lambda x: x['MAE'])

    print("\nBest Model:", best_model['Model'])

    if best_model['Model'] == "Linear Regression":
        final_model = lr
    elif best_model['Model'] == "Decision Tree":
        final_model = dt
    else:
        final_model = rf

    # Save Model
    joblib.dump(final_model, model_dir / "predict_freight_model.pkl")


if __name__ == "__main__":
    main()