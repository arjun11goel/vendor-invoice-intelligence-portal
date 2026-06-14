import joblib
from data_preprocessing import load_invoice_data, apply_labels, split_data
from model_evaluation import train_decision_tree, evaluate_model


FEATURES = [
    'invoice_quantity',
    'invoice_dollars',
    'Freight',
    'total_brands',
    'days_po_to_invoice',
    'total_item_quantity',
    'total_item_dollars'
]

TARGET = 'flag_invoice'


def main():
    print("Loading data...")
    df = load_invoice_data()

    print("Creating labels...")
    df = apply_labels(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, FEATURES, TARGET)

    print("Training model...")
    grid_search = train_decision_tree(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("\nBest Parameters:", grid_search.best_params_)

    print("Evaluating model...")
    evaluate_model(best_model, X_test, y_test)

    print("Saving model...")
    joblib.dump(best_model, "models/predict_flag_invoice.pkl")

    print("Model saved!")


if __name__ == "__main__":
    main()