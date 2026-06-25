import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# Define file paths according to the project's directory structure
INPUT_FILE_PATH = 'dataset/train_set.csv'
OUTPUT_FILE_PATH_LOCAL_TRAIN_SET_SAMPLE = 'dataset/samples/local_train_set_sample_100_rows.csv'
LOCAL_TRAIN_PATH = 'dataset/local_train_set.csv'
LOCAL_CLEAN_TRAIN_PATH = 'dataset/local_clean_train_set.csv'
LOCAL_VAL_PATH = 'dataset/local_validation_set.csv'
NUM_ROWS_TO_SAMPLE = 100


def extract_sample(input_path, output_path, num_rows):
    """
    Reads a small sample from a large CSV file and saves it to a new file.
    """
    # Ensure the target directory exists (create it if it doesn't)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"⏳ Reading the first {num_rows} rows from the file...")

    try:
        # Read only the specified number of rows
        df_sample = pd.read_csv(input_path, nrows=num_rows)

        # Save the sample to a new CSV file without the index column
        df_sample.to_csv(output_path, index=False)

        print(f"✅ Success! File saved to: {output_path}")
        print("Column preview:")
        print(df_sample.columns.tolist())

    except FileNotFoundError:
        print(f"❌ Error: File '{input_path}' not found. Please ensure the original file is in the 'dataset' directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


def check_city_timelines(input_path):
    print("⏳ Loading data to check timelines...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"❌ Error: The file '{input_path}' was not found.")
        return

    # Convert the date column to datetime objects for time-based calculations
    df['date_dt'] = pd.to_datetime(df['date'])

    print("\n📊 Time range summary by city:")
    print("-" * 50)

    # Group by city and extract timeline statistics
    for city in df['city'].unique():
        city_df = df[df['city'] == city]

        min_date = city_df['date_dt'].min()
        max_date = city_df['date_dt'].max()
        total_days = (max_date - min_date).days + 1
        total_rows = len(city_df)

        print(f"🏙️ {city}:")
        print(f"   📅 Start Date: {min_date.strftime('%Y-%m-%d')}")
        print(f"   📅 End Date:   {max_date.strftime('%Y-%m-%d')}")
        print(f"   ⏳ Total Days: {total_days} days")
        print(f"   📝 Total Rows: {total_rows:,}")
        print("-" * 50)

def smart_temporal_split(input_path, train_path, val_path):
    print("⏳ Loading dataset for smart temporal splitting...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"❌ Error: '{input_path}' not found.")
        return

    df['date_dt'] = pd.to_datetime(df['date'])

    train_list = []
    val_list = []

    print("\n✂️ Processing split per city based on available history...")

    for city in df['city'].unique():
        city_df = df[df['city'] == city]

        min_date = city_df['date_dt'].min()
        max_date = city_df['date_dt'].max()
        total_days = (max_date - min_date).days + 1

        # Determine holdout period based on total available days
        if total_days > 30:
            holdout_days = 14  # 2 weeks for large cities
        else:
            holdout_days = 2  # 2 days for cities with minimal data

        cutoff_date = max_date - pd.Timedelta(days=holdout_days)

        city_train = city_df[city_df['date_dt'] < cutoff_date]
        city_val = city_df[city_df['date_dt'] >= cutoff_date]

        train_list.append(city_train)
        val_list.append(city_val)

        print(f" 🏙️ {city}:")
        print(f"   - Total history: {total_days} days")
        print(f"   - Validation size: Last {holdout_days} days (from {cutoff_date.strftime('%Y-%m-%d')})")
        print(f"   - Train rows: {len(city_train):,} | Val rows: {len(city_val):,}\n")

    # Merge all splits and drop the temporary datetime column
    final_train = pd.concat(train_list).drop(columns=['date_dt'])
    final_val = pd.concat(val_list).drop(columns=['date_dt'])

    os.makedirs(os.path.dirname(train_path), exist_ok=True)

    print("💾 Saving split datasets to disk...")
    final_train.to_csv(train_path, index=False)
    final_val.to_csv(val_path, index=False)

    print("✅ Success! Files are ready in the 'dataset' directory.")


def aggregate_and_clean_data(input_path, output_path):
    """
    Reads the raw rides CSV from input_path, cleans it, handles missing values,
    aggregates to the Station-Hour level, and saves the result to output_path.
    """
    print(f"⏳ Loading raw data from: {input_path}")
    try:
        raw_df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"❌ Error: File '{input_path}' not found. Check your dataset directory.")
        return None

    print("🔄 Starting data aggregation and cleaning...")

    # 1. Drop leaky columns (Target Leakage) and empty/problematic columns
    columns_to_drop = [
        'started_at', 'ended_at', 'end_station_id',
        'usage_time_minutes', 'distance_meters', 'user_type',
        'start_lat', 'start_lng', 'holiday_name', 'distance_to_nearest_rail_station'
    ]

    # Only drop columns that actually exist in the dataframe to avoid errors
    existing_cols_to_drop = [col for col in columns_to_drop if col in raw_df.columns]
    df_clean = raw_df.drop(columns=existing_cols_to_drop)

    # 2. Define grouping keys (these define a single "row" in our target dataset)
    groupby_keys = ['city', 'start_station_id', 'date', 'hour_ts']

    # 3. Aggregation: Count the number of rows (rides) in each group
    # We use the first remaining column as an anchor for counting
    first_col = [c for c in df_clean.columns if c not in groupby_keys][0]

    # Calculate total rides (our Target variable)
    target_df = df_clean.groupby(groupby_keys)[first_col].count().reset_index()
    target_df.rename(columns={first_col: 'demand'}, inplace=True)

    # 4. Preserve static features (weather, points of interest, dates)
    # We use first() because weather and environment are identical for all rides in the same hour
    features_df = df_clean.groupby(groupby_keys).first().reset_index()

    # 5. Merge the target (demand) with the features
    final_df = pd.merge(features_df, target_df, on=groupby_keys, how='inner')

    print("🧹 Handling missing values with smart strategies...")

    # Strategy A: Weather -> Fill with Mean
    weather_cols = [
        'temperature_2m', 'relative_humidity_2m', 'apparent_temperature',
        'precipitation', 'rain', 'snowfall', 'wind_speed_10m', 'cloud_cover'
    ]
    for col in weather_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(final_df[col].mean())

    # Strategy B: Points of Interest (POIs) -> Fill with median
    poi_cols = [
        'bike_lane_length_500m', 'park_area_500m', 'university_count_1000m',
        'office_poi_count_1000m', 'retail_poi_count_1000m',
        'restaurant_cafe_count_500m', 'transit_stop_count_500m'
    ]
    for col in poi_cols:
        if col in final_df.columns:
            # 1. Convert hidden placeholders (empty strings, whitespace, -1) to actual NaN
            final_df[col] = final_df[col].replace(['', ' ', -1], np.nan)

            # 2. Calculate median (pandas automatically ignores NaN here)
            median_val = final_df[col].median()

            # 3. Fill the true NaNs with the calculated median
            final_df[col] = final_df[col].fillna(median_val)

    # Strategy C: Distances -> Fill with city mean or overall median
    print("🚉 Handling distance placeholders...")

    distance_cols = ['distance_to_city_center']
    for col in distance_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].fillna(final_df[col].median())

    # Fallback: Fill any remaining numeric columns with 0 just to be safe
    numeric_cols = final_df.select_dtypes(include=['number']).columns
    final_df[numeric_cols] = final_df[numeric_cols].fillna(0)

    # 6. Save the cleaned and aggregated dataset to disk
    print(f"💾 Saving cleaned dataset to: {output_path}")

    # Ensure the target directory exists before saving
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save without the pandas index column
    final_df.to_csv(output_path, index=False)

    print(f"✅ Done! Final dataset dimensions: {final_df.shape}")
    return final_df


def generate_missing_values_map(input_path):
    print(f"⏳ Loading data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"❌ Error: File '{input_path}' not found.")
        return

    print("\n🗺️ Missing Values Map by City")
    print("=" * 50)

    # Get a list of all cities in the dataset
    cities = df['city'].unique()

    for city in cities:
        city_df = df[df['city'] == city]
        total_rows = len(city_df)

        print(f"\n🏙️ {city} (Total Rows: {total_rows:,})")
        print("-" * 30)

        # Calculate the percentage of missing values for each column
        missing_percentages = (city_df.isnull().sum() / total_rows) * 100

        # Filter to show only columns that actually have missing values
        missing_columns = missing_percentages[missing_percentages > 0].sort_values(ascending=False)

        if missing_columns.empty:
            print("   ✅ No missing values found in this city!")
        else:
            for col, pct in missing_columns.items():
                print(f"   ⚠️ {col}: {pct:.2f}% missing")


def analyze_temporal_patterns(input_path):
    print(f"⏳ Loading cleaned data from: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"❌ Error: File '{input_path}' not found.")
        return

    # 1. Feature Extraction: Extract the exact hour (0-23) from the timestamp
    df['hour_of_day'] = pd.to_datetime(df['hour_ts']).dt.hour

    # 2. Aggregation for EDA: Calculate the mean demand per city per hour
    print("\n📊 Calculating average demand per hour for each city...")
    hourly_demand = df.groupby(['city', 'hour_of_day'])['demand'].mean().reset_index()

    # Pivot the table so cities are columns and hours are rows (easier to read)
    pivot_df = hourly_demand.pivot(index='hour_of_day', columns='city', values='demand')

    print("\n--- Mean Hourly Demand by City ---")
    print(pivot_df.round(2))

    # 3. Plotting the results (if your environment supports displaying plots)
    try:
        pivot_df.plot(kind='line', figsize=(12, 6), marker='o')
        plt.title('Average Bike Demand by Hour of Day across Cities')
        plt.xlabel('Hour of Day (0-23)')
        plt.ylabel('Average Number of Rides (Demand)')
        plt.grid(True)
        plt.xticks(range(0, 24))
        plt.legend(title='City')
        plt.tight_layout()

        # Save the plot as an image so you can look at it!
        plt.savefig('dataset/hourly_demand_comparison.png')
        print("\n📈 Awesome! A plot has been saved to 'dataset/hourly_demand_comparison.png'")
    except Exception as e:
        print(f"\n⚠️ Could not generate plot: {e}")


def validate_time_columns(raw_df):
    """
    Validates the integrity of the hour_ts column against the started_at column
    in the raw dataset.
    """
    print("🔍 Starting time columns validation...")

    # Create a copy of the relevant columns to avoid modifying the original dataframe
    df = raw_df[['started_at', 'hour_ts']].copy()

    # ==========================================
    # Check 2: Are there rows with started_at but missing hour_ts?
    # ==========================================
    print("\n--- Check 2: Rows with started_at but missing hour_ts ---")
    missing_hour_ts = df['hour_ts'].isna()
    exists_started_at = df['started_at'].notna()

    missing_hour_violations = df[missing_hour_ts & exists_started_at]

    if len(missing_hour_violations) > 0:
        print(f"⚠️ WARNING: Found {len(missing_hour_violations):,} rows where hour_ts is missing!")
    else:
        print("✅ Passed! Every row with a start time has a corresponding grouping hour.")

    # ==========================================
    # Check 1: Contradictions between the times
    # ==========================================
    print("\n--- Check 1: Contradictions between started_at and hour_ts ---")

    # Filter out rows with missing data so we can safely compare the times
    df_valid = df.dropna(subset=['started_at', 'hour_ts']).copy()

    # Convert to datetime format (coercing errors in case of corrupted text)
    df_valid['started_dt'] = pd.to_datetime(df_valid['started_at'], errors='coerce')
    df_valid['hour_dt'] = pd.to_datetime(df_valid['hour_ts'], errors='coerce')

    # Calculate the expected rounded hour (floor down the start time)
    # Example: 11:45:00 -> 11:00:00
    df_valid['expected_hour_ts'] = df_valid['started_dt'].dt.floor('h')

    # Find rows where the calculated hour does not match the recorded hour in the data
    contradictions = df_valid[df_valid['expected_hour_ts'] != df_valid['hour_dt']]

    if len(contradictions) > 0:
        print(f"⚠️ DANGER: Found {len(contradictions):,} rows with time contradictions!")
        print("Examples of contradictions:")
        print(contradictions[['started_at', 'hour_ts']].head())
    else:
        print("✅ Passed! No contradictions. All start times perfectly match their grouping hour (hour_ts).")

    print("\nValidation complete.")
    return len(contradictions), len(missing_hour_violations)


def check_rail_station_placeholders(raw_df):
    """
    Calculates the percentage of -1 values in the distance_to_nearest_rail_station column,
    both overall and broken down by city.
    """
    print("🚆 Checking for -1 placeholders in rail station distance...")

    col_name = 'distance_to_nearest_rail_station'

    if col_name not in raw_df.columns:
        print(f"❌ Column '{col_name}' not found in the dataset.")
        return

    # Overall percentage calculation
    total_rows = len(raw_df)
    total_minus_one = (raw_df[col_name] == -1).sum()
    overall_pct = (total_minus_one / total_rows) * 100

    print(f"\n📊 Overall Dataset:")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Rows with -1: {total_minus_one:,} ({overall_pct:.2f}%)")

    # Breakdown by city calculation
    print("\n🏙️ Breakdown by City:")
    for city in raw_df['city'].unique():
        city_df = raw_df[raw_df['city'] == city]
        city_total = len(city_df)
        city_minus_one = (city_df[col_name] == -1).sum()
        city_pct = (city_minus_one / city_total) * 100

        print(f"   {city}: {city_minus_one:,} out of {city_total:,} rows ({city_pct:.2f}%)")


def check_poi_missing_data(raw_df):
    """
    Checks the percentage of missing values (NaN and -1) in POI columns,
    broken down by city.
    """
    print("🔍 Scanning for missing values in POI columns...")

    poi_cols = [
        'bike_lane_length_500m', 'park_area_500m', 'university_count_1000m',
        'office_poi_count_1000m', 'retail_poi_count_1000m',
        'restaurant_cafe_count_500m', 'transit_stop_count_500m'
    ]

    existing_pois = [col for col in poi_cols if col in raw_df.columns]

    if not existing_pois:
        print("❌ No POI columns found in this dataset.")
        return

    for col in existing_pois:
        print(f"\n📍 Column: {col}")

        for city in raw_df['city'].unique():
            city_df = raw_df[raw_df['city'] == city]
            total_rows = len(city_df)

            if total_rows == 0:
                continue

            missing_nan = city_df[col].isna().sum()
            missing_minus_one = (city_df[col] == -1).sum()

            total_missing = missing_nan + missing_minus_one
            pct = (total_missing / total_rows) * 100

            if pct > 0:
                print(f"   ⚠️ {city}: {total_missing:,} missing out of {total_rows:,} ({pct:.2f}%)")
            else:
                print(f"   ✅ {city}: 0% missing (Complete data)")

if __name__ == "__main__":
    # extract_sample(INPUT_FILE_PATH, OUTPUT_FILE_PATH_LOCAL_TRAIN_SET_SAMPLE, NUM_ROWS_TO_SAMPLE)
    # check_city_timelines(INPUT_FILE_PATH)
    # generate_missing_values_map(LOCAL_CLEAN_TRAIN_PATH)
    # analyze_temporal_patterns(LOCAL_CLEAN_TRAIN_PATH)
    # raw_data = pd.read_csv('dataset/local_train_set.csv')
    # validate_time_columns(raw_data)
    # check_rail_station_placeholders(raw_data)
    # check_poi_missing_data(raw_data)

    smart_temporal_split(INPUT_FILE_PATH, LOCAL_TRAIN_PATH, LOCAL_VAL_PATH)
    clean_training_data = aggregate_and_clean_data(LOCAL_TRAIN_PATH, LOCAL_CLEAN_TRAIN_PATH)
    pass