import pandas as pd
import numpy as np

# configurable: how many historical days to require
HISTORY_DAYS = 60

def ensure_daily_index(df):
    """Ensure df has continuous daily Date index for its range."""
    df = df.sort_values("Date").set_index("Date")
    full = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full)
    # keep Product Category column
    if "Product Category" in df.columns:
        df["Product Category"] = df["Product Category"].ffill().bfill()
    # Quantity/Price may become NaN for inserted days
    if "Quantity" in df.columns:
        df["Quantity"] = df["Quantity"].fillna(0)
    if "Price per Unit" in df.columns:
        df["Price per Unit"] = df["Price per Unit"].ffill().bfill().fillna(0)
    df.index.name = "Date"
    return df.reset_index()

def build_features(history_df, product, target_date):
    """
    history_df: full daily.csv (Date parsed as datetime, contains Product Category, Quantity, Price per Unit)
    product: product category string
    target_date: pd.Timestamp or parseable date string (the day you want prediction for)
    Returns: dict of features in the exact names used by your model/features.txt
    """
    # parse target_date
    if not isinstance(target_date, pd.Timestamp):
        target_date = pd.to_datetime(target_date)

    # filter product history
    prod_hist = history_df[history_df["Product Category"] == product].copy()
    if prod_hist.empty:
        raise ValueError(f"No history for product '{product}'")

    # ensure continuous daily index for range covering target_date and last HISTORY_DAYS
    prod_hist = ensure_daily_index(prod_hist)
    # if target_date is after last history date, allow prediction using last available history
    last_date = prod_hist["Date"].max()
    if target_date > last_date:
        # extend index to target_date with Quantity=0 days (or keep last price)
        extra_idx = pd.date_range(last_date + pd.Timedelta(days=1), target_date, freq="D")
        if len(extra_idx) > 0:
            tail = pd.DataFrame({
                "Date": extra_idx,
                "Product Category": product,
                "Quantity": 0,
                "Price per Unit": prod_hist["Price per Unit"].iloc[-1] if not prod_hist["Price per Unit"].isna().all() else 0,
                "Total Amount": 0
            })
            prod_hist = pd.concat([prod_hist, tail], ignore_index=True)

    # pick recent window for feature computation
    window_end_idx = prod_hist[prod_hist["Date"] <= target_date].index.max()
    if pd.isna(window_end_idx):
        raise ValueError("target_date is earlier than available history")
    # get up to HISTORY_DAYS before target_date (including target_date row if exists)
    mask = prod_hist["Date"] <= target_date
    recent = prod_hist.loc[mask].copy().reset_index(drop=True)
    recent = recent.tail(HISTORY_DAYS + 1)  # +1 to allow shifts

    qty = recent["Quantity"].reset_index(drop=True)
    price = recent["Price per Unit"].reset_index(drop=True)

    # helper safe functions
    def safe_lag(series, n):
        if len(series) > n:
            return float(series.iloc[-n])
        return 0.0

    def safe_roll_mean(series, n):
        if len(series) >= 1:
            return float(series.shift(1).rolling(n, min_periods=1).mean().iloc[-1])
        return 0.0

    def safe_roll_std(series, n):
        if len(series) >= 1:
            val = series.shift(1).rolling(n, min_periods=1).std().iloc[-1]
            return float(0.0 if pd.isna(val) else val)
        return 0.0

    def safe_ewm(series, alpha=0.3):
        if len(series) >= 1:
            val = series.shift(1).ewm(alpha=alpha).mean().iloc[-1]
            return float(0.0 if pd.isna(val) else val)
        return 0.0

    # build features dictionary
    feats = {
        "Quantity_lag_1": safe_lag(qty, 1),
        "Quantity_lag_7": safe_lag(qty, 7),
        "Quantity_lag_28": safe_lag(qty, 28),
        "Quantity_roll_mean_7": safe_roll_mean(qty, 7),
        "Quantity_roll_mean_14": safe_roll_mean(qty, 14),
        "Quantity_roll_mean_28": safe_roll_mean(qty, 28),
        "Quantity_roll_std_7": safe_roll_std(qty, 7),
        "Quantity_roll_std_14": safe_roll_std(qty, 14),
        "Quantity_roll_std_28": safe_roll_std(qty, 28),
        "Quantity_ewm_0.3": safe_ewm(qty, alpha=0.3),
        "price pct change 7d": float(price.pct_change(7).iloc[-1]) if len(price) > 7 else 0.0,
    }

    # ratio to category 28d mean
    cat_mean_28d = qty.shift(1).rolling(28, min_periods=1).mean().iloc[-1] if len(qty) >= 1 else 0.0
    last_qty = float(qty.iloc[-1]) if len(qty) >= 1 else 0.0
    feats["ratio_to_cat_28d"] = last_qty / (cat_mean_28d + 1e-6)

    # calendar features for target_date
    feats.update({
        "day": int(target_date.day),
        "month": int(target_date.month),
        "dayofweek": int(target_date.dayofweek),
        "is_weekend": int(target_date.dayofweek in (5, 6)),
        "week_of_year": int(target_date.isocalendar().week),
    })

    # ensure no NaNs
    for k, v in list(feats.items()):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            feats[k] = 0.0

    return feats
