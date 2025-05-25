import streamlit as st
import pandas as pd
import requests
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import concurrent.futures

# --- Caching ---
DATA_CACHE_FILE = 'spacex_launches.pkl'
WEATHER_CACHE_FILE = 'weather_cache.json'

# --- Load Cached Data ---
def load_cached_data():
    if os.path.exists(DATA_CACHE_FILE):
        return pd.read_pickle(DATA_CACHE_FILE)
    return None

def save_cached_data(df):
    df.to_pickle(DATA_CACHE_FILE)

# --- Weather Caching ---
if os.path.exists(WEATHER_CACHE_FILE):
    with open(WEATHER_CACHE_FILE, 'r') as f:
        weather_cache = json.load(f)
else:
    weather_cache = {}

def save_weather_cache():
    with open(WEATHER_CACHE_FILE, 'w') as f:
        json.dump(weather_cache, f)

# --- Fetch Weather Data ---
def fetch_weather(lat, lon, date_utc):
    key = f"{lat}_{lon}_{date_utc}"
    if key in weather_cache:
        return weather_cache[key]

    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&start_date={date_utc[:10]}&end_date={date_utc[:10]}&hourly=temperature_2m"
        f"&temperature_unit=celsius&timezone=UTC"
    )

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        temp = data.get('hourly', {}).get('temperature_2m', [None])[0]
        result = {'temp': temp, 'humidity': 50, 'wind_speed': 5}
        weather_cache[key] = result
        save_weather_cache()
        return result
    except Exception as e:
        print(f"Weather fetch failed: {e}")
        return {'temp': None, 'humidity': None, 'wind_speed': None}

# --- Enrich Weather in Parallel ---
def enrich_weather_parallel(df, max_workers=20):
    results = [None] * len(df)

    def task(idx, lat, lon, date_utc):
        weather = fetch_weather(lat, lon, date_utc) if pd.notna(lat) and pd.notna(lon) else {'temp': None, 'humidity': None, 'wind_speed': None}
        results[idx] = weather

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(task, idx, row['latitude'], row['longitude'], row['date_utc']) for idx, row in df.iterrows()]
        concurrent.futures.wait(futures)

    df['temp'] = [r['temp'] for r in results]
    df['humidity'] = [r['humidity'] for r in results]
    df['wind_speed'] = [r['wind_speed'] for r in results]
    return df

# --- Fetch SpaceX API Data ---
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
    launchpad_dict = {l['id']: {'name': l['name'], 'region': l.get('region', ''), 'latitude': l['latitude'], 'longitude': l['longitude']} for l in launchpads}
    payload_dict = {p['id']: p for p in payloads}

    data = []
    for launch in launches:
        rocket_id = launch.get('rocket')
        launchpad_id = launch.get('launchpad')
        date_utc = launch.get('date_utc')
        year = None
        try:
            year = datetime.fromisoformat(date_utc.replace('Z', '+00:00')).year if date_utc else None
        except:
            pass
        payload_ids = launch.get('payloads', [])
        payload_mass_kg = sum(payload_dict.get(pid, {}).get('mass_kg', 0) or 0 for pid in payload_ids)
        data.append({
            'name': launch.get('name'),
            'date_utc': date_utc,
            'rocket_name': rocket_dict.get(rocket_id),
            'launch_site': launchpad_dict.get(launchpad_id, {}).get('name'),
            'launch_region': launchpad_dict.get(launchpad_id, {}).get('region'),
            'success': 1 if launch.get('success') is True else 0,
            'payload_mass_kg': payload_mass_kg if payload_mass_kg > 0 else None,
            'year': year,
            'latitude': launchpad_dict.get(launchpad_id, {}).get('latitude'),
            'longitude': launchpad_dict.get(launchpad_id, {}).get('longitude')
        })

    df = pd.DataFrame(data)
    save_cached_data(df)
    return df

# --- Streamlit App ---
st.set_page_config(page_title="SpaceX Launch Dashboard", layout="wide")
st.title("🚀 SpaceX Launch Analysis & Prediction Platform")

with st.spinner("Loading SpaceX launch data..."):
    df_raw = fetch_spacex_launch_data()

with st.spinner("Enriching with weather data..."):
    df = enrich_weather_parallel(df_raw)

filtered_df = df.dropna(subset=['rocket_name', 'launch_site', 'payload_mass_kg', 'temp', 'humidity', 'wind_speed', 'year'])

# --- Preprocessing ---
le_rocket = LabelEncoder()
le_site = LabelEncoder()

filtered_df['rocket_encoded'] = le_rocket.fit_transform(filtered_df['rocket_name'])
filtered_df['site_encoded'] = le_site.fit_transform(filtered_df['launch_site'])

X = filtered_df[['rocket_encoded', 'site_encoded', 'payload_mass_kg', 'temp', 'humidity', 'wind_speed', 'year']]
y = filtered_df['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

train_accuracy = accuracy_score(y_train, model.predict(X_train))
train_precision = precision_score(y_train, model.predict(X_train))
train_recall = recall_score(y_train, model.predict(X_train))

test_accuracy = accuracy_score(y_test, model.predict(X_test))
test_precision = precision_score(y_test, model.predict(X_test))
test_recall = recall_score(y_test, model.predict(X_test))

# --- Show Metrics ---
st.subheader("📊 Model Evaluation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏋️ Training Set")
    st.write(f"**Accuracy:** {train_accuracy:.2%}")
    st.write(f"**Precision:** {train_precision:.2%}")
    st.write(f"**Recall:** {train_recall:.2%}")

with col2:
    st.markdown("#### 🧪 Test Set")
    st.write(f"**Accuracy:** {test_accuracy:.2%}")
    st.write(f"**Precision:** {test_precision:.2%}")
    st.write(f"**Recall:** {test_recall:.2%}")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Analytics", "🌍 Map", "🤖 Prediction"])

with tab1:
    st.subheader("📋 Full Launch Dataset")

    # Optional: Year filter
    years = sorted(filtered_df['year'].dropna().unique())
    selected_years = st.multiselect("Filter by Year", years, default=years)

    filtered_data = filtered_df[filtered_df['year'].isin(selected_years)]

    st.dataframe(
        filtered_data[['name', 'date_utc', 'rocket_name', 'launch_site', 'success', 'payload_mass_kg']],
        use_container_width=True
    )

    # Download CSV
    csv = filtered_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name='spacex_launch_data.csv',
        mime='text/csv'
    )

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

    for ax, feature, title in zip(axes, ['temp', 'humidity', 'wind_speed'],
                                   ['Temperature vs Launch Success', 'Humidity vs Launch Success', 'Wind Speed vs Launch Success']):
        data = filtered_df[pd.notna(filtered_df[feature])]
        if not data.empty:
            sns.boxplot(x='success', y=feature, data=data, ax=ax)
        else:
            ax.text(0.5, 0.5, f"No {feature} data", ha='center', va='center')
        ax.set_title(title)

    st.pyplot(fig4)

with tab2:
    st.write("### Launch Map")
    m = folium.Map(location=[25, -80], zoom_start=3)
    marker_cluster = MarkerCluster().add_to(m)
    for _, row in filtered_df.iterrows():
        result = "✅ Success" if row['success'] == 1 else "❌ Failure"
        popup = f"{row['name']}<br>{result}<br>Temp: {row['temp']}°C"
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=popup,
            icon=folium.Icon(color="green" if row['success'] else "red")
        ).add_to(marker_cluster)
    st_folium(m, width=700)

with tab3:
    st.write("### Predict Launch Success")

    rocket = st.selectbox("Rocket", le_rocket.classes_)
    site = st.selectbox("Launch Site", le_site.classes_)
    mass = st.number_input("Payload Mass (kg)", 0, 50000, 1000)
    temp = st.number_input("Temperature (°C)", -50, 50, 20)
    humidity = st.number_input("Humidity (%)", 0, 100, 50)
    wind = st.number_input("Wind Speed (m/s)", 0.0, 50.0, 5.0)
    year_input = datetime.now().year

    if st.button("Predict"):
        input_data = pd.DataFrame([[
            le_rocket.transform([rocket])[0],
            le_site.transform([site])[0],
            mass, temp, humidity, wind, year_input
        ]], columns=X.columns)

        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.write("### Result")
        st.success("✅ Likely to Succeed" if pred else "❌ Likely to Fail")
        st.progress(int(prob * 100))
        st.caption(f"Confidence: {prob:.2%}")

st.write("© 2025 SpaceX Launch Dashboard created by Muhammad Ali SMITian")