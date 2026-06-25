import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    X = pd.DataFrame(index=df.index)

    # -------------------------
    # 1. Time-based features
    # -------------------------
    if "hour_ts" in df.columns:
        t = pd.to_datetime(df["hour_ts"], errors="coerce")
    elif "target_hour_start" in df.columns:
        t = pd.to_datetime(df["target_hour_start"], errors="coerce")
    else:
        t = None

    if t is not None:
        X["hour"] = t.dt.hour
        X["weekday"] = t.dt.weekday
        X["month"] = t.dt.month
    else:
        if "hour" in df.columns:
            X["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        if "weekday" in df.columns:
            X["weekday"] = pd.to_numeric(df["weekday"], errors="coerce")
        if "month" in df.columns:
            X["month"] = pd.to_numeric(df["month"], errors="coerce")

    if "weekday" in X.columns:
        X["is_weekend"] = X["weekday"].isin([5, 6]).astype(int)

    if "hour" in X.columns:
        X["is_morning_rush"] = X["hour"].isin([7, 8, 9]).astype(int)
        X["is_evening_rush"] = X["hour"].isin([16, 17, 18, 19]).astype(int)

        X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)

    if "weekday" in X.columns:
        X["weekday_sin"] = np.sin(2 * np.pi * X["weekday"] / 7)
        X["weekday_cos"] = np.cos(2 * np.pi * X["weekday"] / 7)

    if "month" in X.columns:
        X["month_sin"] = np.sin(2 * np.pi * X["month"] / 12)
        X["month_cos"] = np.cos(2 * np.pi * X["month"] / 12)

    # -------------------------
    # 2. Calendar features
    # -------------------------
    calendar_cols = [
        "weekend",
        "holiday",
        "working_day",
    ]

    for col in calendar_cols:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------
    # 3. Weather features
    # -------------------------
    weather_cols = [
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "rain",
        "precipitation",
        "snowfall",
        "cloud_cover",
        "wind_speed_10m",
    ]

    for col in weather_cols:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce")

    if "rain" in X.columns:
        X["is_raining"] = (X["rain"] > 0).astype(int)

    if "precipitation" in X.columns:
        X["has_precipitation"] = (X["precipitation"] > 0).astype(int)

    if "wind_speed_10m" in X.columns:
        X["is_windy"] = (X["wind_speed_10m"] > 20).astype(int)

    if "apparent_temperature" in X.columns:
        X["is_cold"] = (X["apparent_temperature"] < 10).astype(int)
        X["is_hot"] = (X["apparent_temperature"] > 30).astype(int)

    # -------------------------
    # 4. Station environment features
    # -------------------------
    station_environment_cols = [
        "distance_to_city_center",
        "bike_lane_length_500m",
        "transit_stop_count_500m",
        "distance_to_nearest_rail_station",
        "park_area_500m",
        "start_lat",
        "start_lng",
    ]

    for col in station_environment_cols:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------
    # 5. Points-of-interest features
    # -------------------------
    poi_cols = [
        "restaurant_cafe_count_500m",
        "office_poi_count_1000m",
        "university_count_1000m",
        "retail_poi_count_1000m",
    ]

    for col in poi_cols:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce")

    X = X.fillna(0)

    return X


class BikeDemandModel:
    """
    Put your actual model logic here.

    This file should contain:
        - feature creation used during prediction
        - model-specific preprocessing
        - prediction logic

    Do NOT load weights.joblib here.
    The weights are loaded in predict.py and passed into this class.
    """

    def __init__(self):
        self.artifacts = None

    def load_artifacts(self, artifacts: dict) -> None:
        """
        Store all objects created by train.py.

        Examples:
            artifacts["model"]
            artifacts["feature_columns"]
            artifacts["scaler"]
        """
        self.artifacts = artifacts

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict bike demand for each row in test_df.

        Parameters
        ----------
        test_df:
            Hidden station-hour test features provided by the evaluator.
            It does NOT contain the demand column.

        Returns
        -------
        np.ndarray:
            One numeric prediction per row in test_df.
        """
        if self.artifacts is None:
            raise RuntimeError("Model is not loaded. Call load_artifacts() first.")

        X = create_features(test_df) # the same features used during training.

        # Convert the feature table to exactly the same columns used during training.
        # This is important because train and test may contain different categorical values
        # or columns may appear in a different order.
        feature_columns = self.artifacts["feature_columns"]
        X = X.reindex(columns=feature_columns, fill_value=0)

        # Use the trained regression model saved by train.py.
        model = self.artifacts["model"]
        preds = model.predict(X)

        # Bike demand cannot be negative.
        return np.maximum(0.0, preds)