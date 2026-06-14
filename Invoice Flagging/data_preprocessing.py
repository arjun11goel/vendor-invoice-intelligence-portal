import pandas as pd
import os

def load_invoice_data():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    vendor_path = os.path.join(base_path, "Inventory", "vendor_invoice.csv")
    purchases_path = os.path.join(base_path, "Inventory", "purchases.csv")

    vendor = pd.read_csv(vendor_path)
    purchases = pd.read_csv(purchases_path)

    # ---------- Purchases processing ----------
    purchases['PODate'] = pd.to_datetime(purchases['PODate'])
    purchases['ReceivingDate'] = pd.to_datetime(purchases['ReceivingDate'])

    purchases['receiving_delay'] = (
        purchases['ReceivingDate'] - purchases['PODate']
    ).dt.days

    agg_data = purchases.groupby('PONumber').agg({
        'Brand': 'nunique',
        'Quantity': 'sum',
        'Dollars': 'sum',
        'receiving_delay': 'mean'
    }).reset_index()

    agg_data.columns = [
        'PONumber',
        'total_brands',
        'total_item_quantity',
        'total_item_dollars',
        'avg_receiving_delay'
    ]

    # ---------- Vendor processing ----------
    vendor['PODate'] = pd.to_datetime(vendor['PODate'])
    vendor['InvoiceDate'] = pd.to_datetime(vendor['InvoiceDate'])
    vendor['PayDate'] = pd.to_datetime(vendor['PayDate'])

    vendor['days_po_to_invoice'] = (
        vendor['InvoiceDate'] - vendor['PODate']
    ).dt.days

    vendor['days_to_pay'] = (
        vendor['PayDate'] - vendor['InvoiceDate']
    ).dt.days

    vendor['invoice_quantity'] = vendor['Quantity']
    vendor['invoice_dollars'] = vendor['Dollars']

    # ---------- Merge ----------
    df = pd.merge(
        vendor,
        agg_data,
        on='PONumber',
        how='left'
    )

    return df


# EXACT same label logic
def create_invoice_risk_label(row):
    if abs(row["invoice_dollars"] - row["total_item_dollars"]) > 5:
        return 1
    if row["avg_receiving_delay"] > 10:
        return 1
    return 0


def apply_labels(df):
    df["flag_invoice"] = df.apply(create_invoice_risk_label, axis=1)
    return df


from sklearn.model_selection import train_test_split

def split_data(df, features, target):
    X = df[features]
    y = df[target]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42  # VERY IMPORTANT
    )