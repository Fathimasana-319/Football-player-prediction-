import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Football Prediction",
    page_icon="⚽",
    layout="wide"
)

# ---------------- DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071b2b, #102c40, #123d38);
    color: white;
}

.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 35px;
    border-radius: 24px;
    background: linear-gradient(120deg, #123b4b, #164d42);
    border: 1px solid #527d78;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 44px;
    color: white !important;
}

.hero p {
    color: #e4f5f1 !important;
    font-size: 17px;
}

/* DARK PLAYER DETAIL BOXES */

div[data-testid="stMetric"] {
    background: #164d42 !important;
    border: 1px solid #5b9b83 !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #d8f3dc !important;
}

div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] *,
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: transparent !important;
}

/* INPUT BOXES */

div[data-testid="stNumberInput"] input {
    background: #173b42 !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
    border: 1px solid #5b9b83 !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background: #173b42 !important;
    color: white !important;
}

div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
    color: #ffffff !important;
}

/* BUTTON */

div.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    background: #b9f56b !important;
    color: #10251a !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: bold !important;
    min-height: 48px;
}

/* FOOTBALL BANNERS */

.football-banner {
    padding: 24px;
    margin: 22px 0;
    text-align: center;
    border-radius: 20px;
    background: linear-gradient(120deg, #123b4b, #164d42);
    border: 1px solid #527d78;
}

.football-banner h3 {
    color: white !important;
}

.football-banner p {
    color: #e4f5f1 !important;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown("""
<div class="hero">
    <p>⚽ PLAYER PERFORMANCE ESTIMATOR</p>
    <h1>Football Prediction</h1>
    <p>
        Discover football players, explore their performance,
        and predict their overall rating using machine learning.
    </p>
    <h2>⚽ 🏆</h2>
</div>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------

@st.cache_data
def load_data():
    data = pd.read_csv(
        "players_3120.csv",
        encoding="utf-8-sig",
        low_memory=False
    )

    data.columns = (
        data.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    return data


def find_column(data, names):
    for name in names:
        for col in data.columns:
            if col.strip().lower() == name.lower():
                return col
    return None


def to_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


# ---------------- MAIN ----------------

try:
    df = load_data()

    name_col = find_column(
        df, ["name", "player name", "player", "full name"]
    )

    age_col = find_column(df, ["Age"])

    overall_col = find_column(
        df, ["Overall rating", "Overall_Rating", "Overall"]
    )

    potential_col = find_column(df, ["Potential"])

    team_col = find_column(
        df, ["Team", "Club", "Team & Contract", "Club & Contract"]
    )

    if name_col is None:
        st.error("Player name column not found.")
        st.write("CSV columns:", df.columns.tolist())
        st.stop()

    if age_col is None or overall_col is None or potential_col is None:
        st.error("Age, Overall Rating, or Potential column not found.")
        st.write("CSV columns:", df.columns.tolist())
        st.stop()

    df[name_col] = df[name_col].astype("string").str.strip()

    df["_age"] = to_number(df[age_col])
    df["_overall"] = to_number(df[overall_col])
    df["_potential"] = to_number(df[potential_col])

    clean = df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna(
        subset=[name_col, "_age", "_overall", "_potential"]
    ).copy()

    clean = clean[clean[name_col] != ""]

    if len(clean) < 10:
        st.error("Not enough valid player data in the CSV.")
        st.stop()

    # ---------------- TRAIN MODEL ----------------

    X = clean[["_age", "_potential"]]
    y = clean["_overall"]

    model = RandomForestRegressor(
        n_estimators=150,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # ---------------- PLAYER SELECTION ----------------

    st.markdown("""
    <div class="football-banner">
        <h3>⚽ Choose Your Player 🏆</h3>
        <p>Select a player to view their details.</p>
    </div>
    """, unsafe_allow_html=True)

    player_names = sorted(
        clean[name_col].drop_duplicates().tolist()
    )

    selected_name = st.selectbox(
        "Player Name",
        player_names
    )

    player = clean[clean[name_col] == selected_name].iloc[0]

    # ---------------- PLAYER DETAILS ----------------

    st.subheader("Player Details")
    st.markdown(f"### ⚽ {selected_name}")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Age", f"{player['_age']:.0f} years")

    with c2:
        st.metric(
            "Overall Rating",
            f"{player['_overall']:.0f} / 100"
        )

    with c3:
        st.metric(
            "Potential",
            f"{player['_potential']:.0f} / 100"
        )

    with c4:
        team = "Not available"

        if team_col:
            team_value = str(player[team_col]).strip()

            if team_value.lower() not in ["nan", "none", ""]:
                team = team_value

        st.metric("Team", team)

    # ---------------- PREDICTION ----------------

    st.markdown("""
    <div class="football-banner">
        <h3>⚽ Predict Player Rating 🏆</h3>
        <p>Adjust the player's age and potential.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            input_age = st.number_input(
                "Player Age",
                min_value=1.0,
                max_value=100.0,
                value=float(player["_age"]),
                step=1.0
            )

        with col2:
            input_potential = st.number_input(
                "Player Potential",
                min_value=0.0,
                max_value=100.0,
                value=float(player["_potential"]),
                step=1.0
            )

        submitted = st.form_submit_button(
            "⚽ Predict Overall Rating",
            use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame({
            "_age": [input_age],
            "_potential": [input_potential]
        })

        prediction = float(model.predict(input_data)[0])
        prediction = max(0.0, min(100.0, prediction))

        st.markdown("""
        <div class="football-banner">
            <h3>🏆 Matchday Prediction</h3>
            <p>Your player's estimated rating is ready.</p>
        </div>
        """, unsafe_allow_html=True)

        st.metric(
            "Predicted Overall Rating",
            f"{prediction:.1f} / 100"
        )

        st.progress(int(round(prediction)))

        st.caption(
            "This machine-learning estimate is based on the dataset "
            "and is not a guarantee of real-world performance."
        )

    # ---------------- DATASET ----------------

    with st.expander("📊 View Football Player Dataset"):
        st.write(f"Total players: {len(clean)}")
        st.dataframe(df, use_container_width=True)

    # ---------------- FOOTER ----------------

    st.markdown("""
    <div class="football-banner">
        <h3>⚽ Football Prediction 🏆</h3>
        <p>Powered by Python, Streamlit and Machine Learning.</p>
    </div>
    """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error(
        "players_3120.csv not found. Keep it in the same "
        "folder as app.py."
    )

except Exception as e:
    st.error(f"Something went wrong: {e}")