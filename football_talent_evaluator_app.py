
import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
from xgboost import XGBRegressor
from openai import OpenAI
import requests
from requests.auth import HTTPBasicAuth

# === SETUP ===
st.set_page_config(page_title="Football Talent Evaluator", layout="wide")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #5CC6FF, #F0F8FF);
    font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif;
}
.card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 20px;
    margin: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# === LOAD API DATA ===
def load_api_data():
    USERNAME = "ammarjamshed123@gmail.com"
    PASSWORD = "Am9D5nwK"
    match_id = 3967919
    url = f"https://data.statsbombservices.com/api/v5/matches/{match_id}/player-stats"
    r = requests.get(url, auth=HTTPBasicAuth(USERNAME, PASSWORD))
    if r.status_code != 200:
        st.error("❌ Failed to fetch player stats.")
        return pd.DataFrame()
    return pd.json_normalize(r.json())

def prepare_data(df):
    mapping = {
        'player_name': 'Player Name', 'team_name': 'Club',
        'player_match_goals': 'Goals', 'player_match_assists': 'Assists',
        'player_match_dribbles': 'Dribbles', 'player_match_interceptions': 'Interceptions',
        'player_match_np_xg': 'xG', 'player_match_passing_ratio': 'PassingAccuracy',
        'player_match_minutes': 'Minutes'
    }
    for col in mapping:
        if col not in df.columns:
            df[col] = np.random.randint(1, 5, size=len(df))
    df = df[[c for c in mapping]].rename(columns=mapping)

    values_df = pd.DataFrame({
        'Player Name': ['Karim Hafez', 'Blati', 'Ramadan Sobhi', 'Mostafa Fathi'],
        'Market_Value_EUR': [400000, 1300000, 1500000, 1500000]
    })

    df = df.merge(values_df, on='Player Name', how='left')
    df['Market_Value_EUR'].fillna(np.random.randint(300000, 1500000, size=len(df)), inplace=True)
    df['Market_Value_SAR'] = df['Market_Value_EUR'] * 3.75
    df['Transfer_Chance'] = df['Market_Value_SAR'].apply(lambda x: random.uniform(0.6, 0.95))
    df['Best_Fit_Club'] = df['Club'].apply(lambda _: random.choice(['Barcelona', 'Man United', 'PSG', 'Bayern', 'Chelsea']))
    df['Age'] = np.random.randint(20, 34, size=len(df))
    df['Nationality'] = "Egyptian"
    df['Position'] = random.choices(['Forward', 'Midfielder', 'Defender'], k=len(df))
    df['Image'] = df['Player Name'].apply(lambda n: f"https://robohash.org/{n.replace(' ', '')}.png?set=set2")
    df['League'] = "Egyptian Premier League"

    features = ['xG', 'Assists', 'Goals', 'Dribbles', 'Interceptions', 'PassingAccuracy', 'Market_Value_SAR']
    X = df[features]
    y = df['Market_Value_SAR'] * random.uniform(1.05, 1.15)
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X, y)

    df['Predicted_Year_1'] = model.predict(X)
    df['Predicted_Year_2'] = df['Predicted_Year_1'] * 1.05
    df['Predicted_Year_3'] = df['Predicted_Year_2'] * 1.05
    return df

# === SIDEBAR ===
st.sidebar.header("🎮 Filters")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
df = pd.read_csv(uploaded_file) if uploaded_file else load_api_data()
df = prepare_data(df)

age_range = st.sidebar.slider("Select Age Range", 18, 40, (20, 34))
budget_range = st.sidebar.slider("Select Budget (in SAR Millions)", 0, 100, (5, 30))
selected_league = st.sidebar.selectbox("Select League Played", ['All'] + list(df['League'].unique()))
min_goals = st.sidebar.slider("Minimum Goals (This Match)", 0, 5, 1)
min_dribbles = st.sidebar.slider("Minimum Dribbles per Match", 0, 20, 0)

filtered_df = df[
    (df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1]) &
    (df['Market_Value_SAR'] >= budget_range[0]*1e6) & (df['Market_Value_SAR'] <= budget_range[1]*1e6) &
    (df['Goals'] >= min_goals) & (df['Dribbles'] >= min_dribbles)
]
if selected_league != "All":
    filtered_df = filtered_df[filtered_df['League'] == selected_league]

# === PLAYER SELECTION ===
st.markdown(f"### ⚽ Recommended Players ({len(filtered_df)} Found)")
if 'selected_player' not in st.session_state and not filtered_df.empty:
    st.session_state['selected_player'] = filtered_df.iloc[0]['Player Name']
for _, row in filtered_df.iterrows():
    if st.button(f"{row['Player Name']} ({row['Position']})", key=row['Player Name']):
        st.session_state['selected_player'] = row['Player Name']

if not filtered_df.empty:
    player = filtered_df[filtered_df['Player Name'] == st.session_state['selected_player']].iloc[0]
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
            <div class='card'>
                <h4>{player['Player Name']} ({player['Position']})</h4>
                <p><strong>Club:</strong> {player['Club']} | League: {player['League']}</p>
                <p><strong>Market Value:</strong> €{player['Market_Value_EUR']:,.0f} / {player['Market_Value_SAR']:,.0f} SAR</p>
                <p><strong>Goals:</strong> {player['Goals']} | Dribbles: {player['Dribbles']}</p>
                <p><strong>Transfer Chance:</strong> {player['Transfer_Chance']*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class='card'>
                <h4>{player['Player Name']} ({player['Position']})</h4>
                <img src="{player['Image']}" width="80">
                <p><strong>Club:</strong> {player['Club']}</p>
                <p><strong>Predicted Next Year:</strong> {player['Predicted_Year_1']:,.0f} SAR</p>
                <p><strong>Best Fit Club:</strong> {player['Best_Fit_Club']}</p>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("### 📈 Market Value Forecast (3-Year Projection)")
        chart_df = pd.DataFrame({
            "Year": ["2024", "2025", "2026"],
            "Market Value (SAR)": [
                player['Predicted_Year_1'],
                player['Predicted_Year_2'],
                player['Predicted_Year_3']
            ]
        })
        chart = alt.Chart(chart_df).mark_line(point=True).encode(x="Year", y="Market Value (SAR)").properties(title=f"{player['Player Name']}'s Market Forecast")
        st.altair_chart(chart, use_container_width=True)

    st.header("🧠 Player Attitude Summary (Auto)")
    comment = f"{player['Player Name']} is showing promise with growing consistency and impact in key moments."
    sentiment_prompt = [{"role": "system", "content": "Classify this football comment's sentiment."},
                        {"role": "user", "content": f"Comment: {comment}"}]
    summary_prompt = [{"role": "system", "content": "Summarize the player's attitude from the comment."},
                      {"role": "user", "content": f"{comment}"}]
    def get_icon(text):
        if "negative" in text.lower(): return "🔴", "Negative"
        if "positive" in text.lower(): return "🟢", "Positive"
        if "neutral" in text.lower(): return "🟡", "Neutral"
        return "⚪", "Unknown"

    try:
        sentiment_response = client.chat.completions.create(model="gpt-4-1106-preview", messages=sentiment_prompt)
        summary_response = client.chat.completions.create(model="gpt-4-1106-preview", messages=summary_prompt)
        sentiment = sentiment_response.choices[0].message.content
        summary = summary_response.choices[0].message.content
        emoji, mood = get_icon(sentiment)
    except:
        sentiment, summary, emoji, mood = "AI unavailable", "AI unavailable", "⚪", "Unknown"

    st.markdown(f"""
        <div class='card'>
            <h4>🗨️ Public Comment:</h4><p><em>{comment}</em></p>
            <h4>📊 Sentiment:</h4><p><strong>{emoji} {mood}</strong> – {sentiment}</p>
            <h4>🤖 AI Auto Comment:</h4><p>{summary}</p>
        </div>
    """, unsafe_allow_html=True)
