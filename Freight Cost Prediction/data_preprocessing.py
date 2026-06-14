import pandas as pd
from sklearn.model_selection import train_test_split

# Load CSV
def load_vendor_invoice_data(file_path: str):
    df = pd.read_csv(file_path)
    return df


def prepare_features(df: pd.DataFrame):
    # Select features
    X = df[['Quantity', 'Dollars']]
    y = df['Freight']
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)