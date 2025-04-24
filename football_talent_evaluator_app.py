import streamlit as st
import pandas as pd
import numpy as np
import random
import altair as alt
import requests
from requests.auth import HTTPBasicAuth
from xgboost import XGBRegressor
from openai import OpenAI

st.set_page_config(page_title="Football Talent Evaluator", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ==== UI CLEANUP ====
st.markdown("""
    <style>
    #MainMenu, header, footer,
    .css-164nlkn, .viewerBadge_link__1S137,
    .css-1r6slb0.egzxvld1, .css-1dp5vir,
    .st-emotion-cache-zq5wmm.ezrtsby0, .st-emotion-cache-1dp5vir,
    .st-emotion-cache-30xxz9 {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==== Reference Values ====
reference_values = {
    "Karim Hafez": 400000, "Blati": 1300000, "Ramadan Sobhi": 2500000,
    "Mostafa Fathi": 1500000, "Emmanuel Apeh": 350000, "Ali Gabr": 500000,
    "Marwan Hamdy": 1000000, "Ahmed El Shimi": 250000, "Fady Farid": 180000,
    "Ahmed Tawfik": 300000, "Ahmed Samy": 350000, "Al Mahdi Soliman": 200000,
    "Ahmed Ayman": 150000, "Mohanad Mostafa Lasheen": 750000, "Karim El Deeb": 450000
}

def validate_market_value(player_name, input_value):
    ref = reference_values.get(player_name)
    return "🟡 Unknown" if ref is None else ("🔴 Off by >€2M" if abs(ref - input_value) > 2_000_000 else "🟢 Verified")

# ==== Load Data ====
@st.cache_data
def load_api_data():
    url = "https://data.statsbombservices.com/api/v5/matches/3967919/player-stats"
    r = requests.get(url, auth=HTTPBasicAuth("ammarjamshed123@gmail.com", "Am9D5nwK"))
    return pd.json_normalize(r.json()) if r.status_code == 200 else pd.DataFrame()

# ==== Model Cache ====
@st.cache_resource
def train_model(X, y):
    model = XGBRegressor(objective="reg:squarederror")
    model.fit(X, y)
    return model

# ==== Prepare Data ====
@st.cache_data
def prepare_data(raw_df):
    np.random.seed(42)
    mapping = {
        'player_name': 'Player Name', 'team_name': 'Club',
        'player_match_goals': 'Goals', 'player_match_assists': 'Assists',
        'player_match_dribbles': 'Dribbles', 'player_match_interceptions': 'Interceptions',
        'player_match_np_xg': 'xG', 'player_match_passing_ratio': 'PassingAccuracy',
        'player_match_minutes': 'Minutes'
    }

    df = raw_df.rename(columns={k: v for k, v in mapping.items() if k in raw_df.columns})
    for col in mapping.values():
        if col not in df.columns:
            df[col] = np.random.randint(0, 5, size=len(df))

    df['Asking_Price_EUR'] = df['Player Name'].map(reference_values)
    df['Asking_Price_SAR'] = df['Asking_Price_EUR'] * 3.75
    df['Asking_Price_SAR'] = df['Asking_Price_SAR'] * np.random.uniform(1.05, 1.2, size=len(df))
    df['Verification'] = [validate_market_value(p, v) for p, v in zip(df['Player Name'], df['Market_Value_EUR'])]
    df['Age'] = np.random.randint(22, 30, size=len(df))
    df['Image'] = df['Player Name'].apply(lambda n: f"https://robohash.org/{n.replace(' ', '')}.png?set=set2")
    df['Nationality'] = "Egyptian"
    df['Position'] = np.random.choice(['Forward', 'Midfielder', 'Defender'], size=len(df))
    df['League'] = "Egyptian Premier League"
    df['Transfer_Chance'] = np.random.uniform(0.6, 0.95, size=len(df))
    df['Best_Fit_Club'] = np.random.choice(['Man United', 'Al Hilal', 'Barcelona', 'PSG'], size=len(df))

    features = ['xG', 'Assists', 'Goals', 'Dribbles', 'Interceptions', 'PassingAccuracy', 'Asking_Price_SAR']
    df = df.dropna(subset=features)
    X = df[features]
    y = df['Asking_Price_SAR'] * np.random.uniform(1.05, 1.15, size=len(df))

    model = train_model(X, y)
    df['Predicted_Year_1'] = model.predict(X)
    df['Predicted_Year_2'] = df['Predicted_Year_1'] * 1.05
    df['Predicted_Year_3'] = df['Predicted_Year_2'] * 1.05

    return df

# ==== Upload Button ====
st.sidebar.markdown("### 📁 Upload Player Data")
st.sidebar.link_button("🔁 Go to Upload Portal", url="https://testmodelcheck.streamlit.app/")
st.sidebar.markdown("💡Loading Data from Academies and Clubs")

df = prepare_data(load_api_data())

# ==== Filters ====
st.sidebar.header("🎮 Filters")
age_range = st.sidebar.slider("Age", 18, 40, (20, 34))
budget_range = st.sidebar.slider("Budget (SAR)", 0, 30, (5, 25))
min_goals = st.sidebar.slider("Min Goals", 0, 5, 1)
min_dribbles = st.sidebar.slider("Min Dribbles", 0, 20, 5)

filtered_df = df[
    (df['Age'].between(age_range[0], age_range[1])) &
    (df['Asking_Price_SAR'].between(budget_range[0]*1e6, budget_range[1]*1e6)) &
    (df['Goals'] >= min_goals) &
    (df['Dribbles'] >= min_dribbles)
]

if 'selected_player' not in st.session_state and not filtered_df.empty:
    st.session_state['selected_player'] = filtered_df.iloc[0]['Player Name']

# ==== Layout ====
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"### ⚽ Recommended Players ({len(filtered_df)} Found)")
    for _, row in filtered_df.iterrows():
        if st.button(f"{row['Player Name']} ({row['Position']})", key=row['Player Name']):
            st.session_state['selected_player'] = row['Player Name']
        st.markdown(f"""
        <div class='card'>
            <h4>{row['Player Name']} ({row['Position']})</h4>
            <p><strong>Club:</strong> {row['Club']} | League: {row['League']}</p>
            <p><strong>Market Value:</strong> €{row['Asking_Price_EUR']:,.0f}</p>
            <p><strong>Transfer Chance:</strong> {row['Transfer_Chance']*100:.1f}%</p>
            <p><strong>Verification:</strong> {row['Verification']}</p>
        </div>
        """, unsafe_allow_html=True)

with col2:
    if not filtered_df.empty:
        player = filtered_df[filtered_df['Player Name'] == st.session_state['selected_player']].iloc[0]
        st.markdown(f"""
        <div class='card'>
            <h3>{player['Player Name']} ({player['Position']})</h3>
            <img src="{player['Image']}" width="80">
            <p><strong>Club:</strong> {player['Club']}</p>
            <p><strong>Predicted Next Year:</strong> {player['Predicted_Year_1']:,.0f} SAR</p>
            <p><strong>Asking Price:</strong> {player['Asking_Price_SAR']:,.0f} SAR</p>
            <p><strong>Verification:</strong> {player['Verification']}</p>
            <p><strong>Best Fit:</strong> {player['Best_Fit_Club']}</p>
        </div>
        """, unsafe_allow_html=True)

        chart_df = pd.DataFrame({
            "Year": ["2024", "2025", "2026"],
            "Predicted Value (SAR)": [
                player['Predicted_Year_1'], player['Predicted_Year_2'], player['Predicted_Year_3']
            ],
            "Club Asking Price (SAR)": [player['Asking_Price_SAR']] * 3
        })

        chart = alt.Chart(chart_df).transform_fold(
            ['Predicted Value (SAR)', 'Club Asking Price (SAR)'],
            as_=['Metric', 'SAR Value']
        ).mark_line(point=True).encode(
            x='Year:N', y='SAR Value:Q', color='Metric:N'
        )
        st.altair_chart(chart, use_container_width=True)

        st.header("🧠 Player Attitude Summary (Auto)")
        comment = f"{player['Player Name']} is showing promise with growing consistency and impact in key moments."
        sentiment_prompt = [
            {"role": "system", "content": "You are a football sentiment expert. Classify the comment."},
            {"role": "user", "content": comment}
        ]
        summary_prompt = [
            {"role": "system", "content": "You're a football expert. Write a 1-sentence summary."},
            {"role": "user", "content": comment}
        ]

        def get_icon(text):
            return ("🔴", "Negative") if "negative" in text.lower() else (
                   ("🟢", "Positive") if "positive" in text.lower() else
                   ("🟡", "Neutral") if "neutral" in text.lower() else ("⚪", "Unclear"))

        try:
            sentiment_response = client.chat.completions.create(model="gpt-4-1106-preview", messages=sentiment_prompt)
            sentiment_reply = sentiment_response.choices[0].message.content
            emoji, mood = get_icon(sentiment_reply)

            summary_response = client.chat.completions.create(model="gpt-4-1106-preview", messages=summary_prompt)
            summary_text = summary_response.choices[0].message.content
        except Exception as e:
            sentiment_reply = "Unable to classify sentiment."
            summary_text = "AI summary unavailable."
            emoji, mood = "⚪", "Unknown"
            st.warning(f"⚠️ AI fallback used: {e}")

        st.markdown(f"""
        <div class='card'>
            <h4>🗨️ Public Comment:</h4>
            <p><em>{comment}</em></p>
            <h4>📊 Sentiment:</h4>
            <p><strong>{emoji} {mood}</strong> – {sentiment_reply}</p>
            <h4>🤖 AI Auto Comment:</h4>
            <p>{summary_text}</p>
        </div>
        """, unsafe_allow_html=True)
