import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def preprocess_data(df):
    """
    Preprocess the SpaceX dataframe for ML modeling.
    Encodes categorical variables and selects relevant features.
    Returns X (features), y (target), and the encoders.
    """
    df = df.copy()

    # Drop rows with missing values in required columns
    df = df.dropna(subset=['rocket_name', 'launch_site', 'payload_mass_kg', 'year', 'success'])

    # Encode categorical variables
    le_rocket = LabelEncoder()
    le_site = LabelEncoder()
    df['rocket_encoded'] = le_rocket.fit_transform(df['rocket_name'])
    df['site_encoded'] = le_site.fit_transform(df['launch_site'])

    # Features and target
    X = df[['rocket_encoded', 'site_encoded', 'payload_mass_kg', 'year']]
    if 'temp' in df.columns and 'humidity' in df.columns and 'wind_speed' in df.columns:
        # Include weather features if available
        X['temp'] = df['temp']
        X['humidity'] = df['humidity']
        X['wind_speed'] = df['wind_speed']

    y = df['success']

    return X, y, le_rocket, le_site


def train_model(X, y):
    """
    Train a RandomForestClassifier on the given data.
    Returns the trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model