import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from spacex import preprocess_data, train_model  # your helper functions
import os
import math
import concurrent.futures

DATA_CACHE_FILE = 'spacex_launches.pkl'
WEATHER_CACHE_FILE = 'weather_cache.json'

# --- Load and save cached SpaceX data ---
def load_cached_data():
    if os.path.exists(DATA_CACHE_FILE):
        return pd.read_pickle(DATA_CACHE_FILE)
    return None

def save_cached_data(df):
    df.to_pickle(DATA_CACHE_FILE)

# --- Load and save cached weather data ---
if os.path.exists(WEATHER_CACHE_FILE):
    with open(WEATHER_CACHE_FILE, 'r') as f:
        weather_cache = json.load(f)
else:
    weather_cache = {}

def save_weather_cache():
    with open(WEATHER_CACHE_FILE, 'w') as f:
        json.dump(weather_cache, f)

# --- OpenWeather API Key ---
OPENWEATHER_API_KEY = "49a24954742ac4b0088a0bb88941a160"  # <-- Replace with your actual key

# --- Fetch weather data with caching ---
@st.cache_data(show_spinner=True)
def fetch_weather(lat, lon, date_utc):
    key = f"{lat}_{lon}_{date_utc}"
    if key in weather_cache:
        return weather_cache[key]

    dt = int(datetime.fromisoformat(date_utc.replace('Z', '+00:00')).timestamp())
    url = (f"https://api.openweathermap.org/data/3.0/onecall/timemachine"
           f"?lat={lat}&lon={lon}&dt={dt}&appid={OPENWEATHER_API_KEY}&units=metric")

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        weather = data.get('data', [])[0] if 'data' in data else data.get('hourly', [{}])[0]

        result = {
            'temp': weather.get('temp'),
            'humidity': weather.get('humidity'),
            'wind_speed': weather.get('wind_speed')
        }
        weather_cache[key] = result
        save_weather_cache()
        return result
    except Exception as e:
        print(f"Failed to fetch weather for {lat}, {lon} at {date_utc}: {e}")
        return {'temp': None, 'humidity': None, 'wind_speed': None}

def fetch_weather_safe(lat, lon, date_utc):
    try:
        return fetch_weather(lat, lon, date_utc)
    except Exception as e:
        print(f"Error fetching weather: {e}")
        return {'temp': None, 'humidity': None, 'wind_speed': None}

# --- Parallel weather enrichment ---
def enrich_weather_parallel(df, max_workers=10):
    results = [None] * len(df)

    def task(idx, lat, lon, date_utc):
        if pd.notna(lat) and pd.notna(lon) and pd.notna(date_utc):
            weather = fetch_weather_safe(lat, lon, date_utc)
        else:
            weather = {'temp': None, 'humidity': None, 'wind_speed': None}
        results[idx] = weather

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, row in df.iterrows():
            futures.append(executor.submit(task, idx, row['latitude'], row['longitude'], row['date_utc']))
        concurrent.futures.wait(futures)

    temps = [r['temp'] for r in results]
    humidities = [r['humidity'] for r in results]
    wind_speeds = [r['wind_speed'] for r in results]

    df['temp'] = temps
    df['humidity'] = humidities
    df['wind_speed'] = wind_speeds

    return df

# --- Fetch SpaceX Launch data ---
@st.cache_data(show_spinner=True)
def fetch_spacex_launch_data():
    cached = load_cached_data()
    if cached is not None:
        return cached

    launches = requests.get("https://api.spacexdata.com/v4/launches").json()
    rockets = requests.get("https://api.spacexdata.com/v4/rockets").json()
    launchpads = requests.get("https://api.spacexdata.com/v4/launchpads").json()
    payloads = requests.get("https://api.spacexdata.com/v4/payloads").json()

    rocket_dict = {r['id']: r['name'] for r in rockets}
    launchpad_dict = {l['id']: {'name': l['name'], 'region': l.get('region', ''),
                                'latitude': l['latitude'], 'longitude': l['longitude']} for l in launchpads}
    payload_dict = {p['id']: p for p in payloads}

    data = []
    for launch in launches:
        rocket_id = launch.get('rocket')
        launchpad_id = launch.get('launchpad')

        rocket_name = rocket_dict.get(rocket_id, None)
        launchpad_info = launchpad_dict.get(launchpad_id, {})

        payload_ids = launch.get('payloads', [])
        payload_mass_kg = 0
        for pid in payload_ids:
            payload_data = payload_dict.get(pid, {})
            mass = payload_data.get('mass_kg')
            if mass:
                payload_mass_kg += mass

        date_utc = launch.get('date_utc')
        year = None
        if date_utc:
            try:
                year = datetime.fromisoformat(date_utc.replace('Z', '+00:00')).year
            except Exception:
                year = None

        success_val = launch.get('success')
        success = 1 if success_val is True else 0

        data.append({
            'name': launch.get('name'),
            'date_utc': date_utc,
            'rocket': rocket_id,
            'launchpad': launchpad_id,
            'success': success,
            'payloads': payload_ids,
            'rocket_name': rocket_name,
            'launch_site': launchpad_info.get('name'),
            'launch_region': launchpad_info.get('region'),
            'payload_mass_kg': payload_mass_kg if payload_mass_kg > 0 else None,
            'year': year,
            'latitude': launchpad_info.get('latitude'),
            'longitude': launchpad_info.get('longitude')
        })

    df = pd.DataFrame(data)
    save_cached_data(df)
    return df

# --- Streamlit App ---
st.set_page_config(page_title="SpaceX Launch Dashboard", layout="wide")
st.title("🚀 SpaceX Launch Analysis & Prediction Platform")

with st.spinner("Loading SpaceX launch data..."):
    df_raw = fetch_spacex_launch_data()

with st.spinner("Enriching data with weather info (faster)..."):
    df = enrich_weather_parallel(df_raw)

filtered_df = df.copy()

st.write("### Sample Launch Data", filtered_df.head())

# Preprocess and train model
X, y, le_rocket, le_site = preprocess_data(filtered_df)
model = train_model(X, y)
st.success("Model trained successfully!")

tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🌍 Launch Map", "🤖 Predict Launch Success"])

with tab1:
    st.subheader("📈 Launch History")
    st.dataframe(filtered_df[['name', 'date_utc', 'rocket_name', 'launch_site', 'success', 'payload_mass_kg']])

    st.subheader("✅ Success Rate by Rocket")
    fig1 = plt.figure(figsize=(8, 4))
    sns.barplot(data=filtered_df, x='rocket_name', y='success')
    plt.ylim(0, 1)
    st.pyplot(fig1)

    st.subheader("📦 Payload Mass vs Success")
    fig2 = plt.figure(figsize=(8, 4))
    sns.boxplot(data=filtered_df, x='success', y='payload_mass_kg')
    st.pyplot(fig2)

    st.subheader("📅 Success Over Time")
    success_by_year = filtered_df.groupby('year')['success'].mean().reset_index()
    fig3 = plt.figure(figsize=(8, 4))
    sns.lineplot(data=success_by_year, x='year', y='success', marker='o')
    st.pyplot(fig3)

    st.subheader("🌡️ Weather Impact on Launch Success")
    fig4, axes = plt.subplots(1, 3, figsize=(18, 5))

    temp_data = filtered_df[pd.notna(filtered_df['temp'])]
    humidity_data = filtered_df[pd.notna(filtered_df['humidity'])]
    wind_data = filtered_df[pd.notna(filtered_df['wind_speed'])]

    if not temp_data.empty:
        sns.boxplot(x='success', y='temp', data=temp_data, ax=axes[0])
    else:
        axes[0].text(0.5, 0.5, "No temp data", ha='center', va='center')
    axes[0].set_title('Temperature vs Launch Success')

    if not humidity_data.empty:
        sns.boxplot(x='success', y='humidity', data=humidity_data, ax=axes[1])
    else:
        axes[1].text(0.5, 0.5, "No humidity data", ha='center', va='center')
    axes[1].set_title('Humidity vs Launch Success')

    if not wind_data.empty:
        sns.boxplot(x='success', y='wind_speed', data=wind_data, ax=axes[2])
    else:
        axes[2].text(0.5, 0.5, "No wind speed data", ha='center', va='center')
    axes[2].set_title('Wind Speed vs Launch Success')

    st.pyplot(fig4)

with tab2:
    st.subheader("🗺️ Launch Site Map")
    m = folium.Map(location=[25, -80], zoom_start=3)
    marker_cluster = MarkerCluster().add_to(m)

    for _, row in filtered_df.iterrows():
        if pd.isna(row['latitude']) or pd.isna(row['longitude']):
            continue
        result = "✅ Success" if row['success'] == 1 else "❌ Failure"
        popup_text = (
            f"{row['name']} ({result})<br>"
            f"Temp: {row['temp']}\u00b0C<br>"
            f"Humidity: {row['humidity']}%<br>"
            f"Wind Speed: {row['wind_speed']} m/s"
        )
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup_text,
            icon=folium.Icon(color="green" if row['success'] == 1 else "red")
        ).add_to(marker_cluster)

    st_folium(m, width=700)

with tab3:
    st.subheader("🤖 Predict Launch Success")

    rocket_choices = filtered_df['rocket_name'].dropna().unique()
    launch_site_choices = filtered_df['launch_site'].dropna().unique()

    selected_rocket = st.selectbox("Select Rocket", rocket_choices)
    selected_site = st.selectbox("Select Launch Site", launch_site_choices)
    payload_mass_input = st.number_input("Payload Mass (kg)", min_value=0, max_value=50000, value=1000)
    temp_input = st.number_input("Temperature (°C)", value=15.0)
    humidity_input = st.number_input("Humidity (%)", value=50)
    wind_speed_input = st.number_input("Wind Speed (m/s)", value=5.0)

    if st.button("Predict Launch Success"):
        # Encode categorical inputs
        try:
            rocket_encoded = le_rocket.transform([selected_rocket])[0]
            site_encoded = le_site.transform([selected_site])[0]
        except Exception:
            st.error("Selected rocket or launch site not in model encoding. Please select different options.")
            rocket_encoded = 0
            site_encoded = 0

        input_df = pd.DataFrame([[
            rocket_encoded,
            site_encoded,
            payload_mass_input,
            temp_input,
            humidity_input,
            wind_speed_input
        ]], columns=['rocket', 'launch_site', 'payload_mass_kg', 'temp', 'humidity', 'wind_speed'])

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        st.write(f"### Prediction: {'Success' if prediction == 1 else 'Failure'}")
        st.write(f"Probability of success: {prob:.2%}")

st.write("© 2025 SpaceX Launch Dashboard")
