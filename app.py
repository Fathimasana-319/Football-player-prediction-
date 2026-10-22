import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Football Prediction",
    page_icon="⚽",
    layout="wide"
)

CSV_PATH = "players_3120.csv"

TIERS = [
    (80, "Best", "#7be495"),
    (70, "Average", "#f5c542"),
    (0, "Weak", "#e07a5f")
]

st.markdown("""
<style>
.stApp {
    background:
        repeating-linear-gradient(
            90deg,
            rgba(255,255,255,.035) 0 90px,
            transparent 90px 180px
        ),
        linear-gradient(180deg,#0a3318,#12502a 55%,#0d3b1e);
}

.stApp::before {
    content: "";
    position: fixed;
    left: 50%;
    top: 50%;
    width: 420px;
    height: 420px;
    margin: -210px 0 0 -210px;
    border: 3px solid rgba(255,255,255,.06);
    border-radius: 50%;
    pointer-events: none;
}

h1, h2, h3, p, label {
    color: #ffffff;
}

.card {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.2);
    border-radius: 18px;
    padding: 18px 22px;
    margin-bottom: 12px;
}

.card small {
    color: #e5e7eb !important;
    opacity: 1;
}

.card h2 {
    color: #ffffff !important;
    margin: 4px 0 0 0;
}

.badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 999px;
    color: #102b19 !important;
    font-weight: 700;
    font-size: 1.1rem;
}

/* SELECTED DROPDOWN: WHITE BACKGROUND, DARK TEXT */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 2px solid #c6e84a !important;
}

/* Selected player name, tier and position */
div[data-baseweb="select"] input,
div[data-baseweb="select"] input::placeholder {
    color: #12351f !important;
    -webkit-text-fill-color: #12351f !important;
    opacity: 1 !important;
}

/* Dropdown selected value */
div[data-baseweb="select"] div,
div[data-baseweb="select"] span {
    color: #12351f !important;
    -webkit-text-fill-color: #12351f !important;
    opacity: 1 !important;
}

/* Dropdown arrow */
div[data-baseweb="select"] svg {
    fill: #12351f !important;
    color: #12351f !important;
}

/* OPEN DROPDOWN MENU */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
ul[role="listbox"] {
    background-color: #ffffff !important;
    opacity: 1 !important;
}

/* All player names, Best, Average, Weak, CAM, CM, CDM, CF */
li[role="option"],
li[role="option"] div,
li[role="option"] span,
ul[role="listbox"] *,
div[data-baseweb="popover"] * {
    color: #12351f !important;
    -webkit-text-fill-color: #12351f !important;
    opacity: 1 !important;
}

/* Hover and selected dropdown option */
li[role="option"]:hover,
li[role="option"]:hover div,
li[role="option"]:hover span,
li[aria-selected="true"],
li[aria-selected="true"] div,
li[aria-selected="true"] span {
    background-color: #d9f99d !important;
    color: #12351f !important;
    -webkit-text-fill-color: #12351f !important;
}

/* Number input */
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-baseweb="input"] input {
    background-color: #ffffff !important;
    color: #12351f !important;
    -webkit-text-fill-color: #12351f !important;
}

/* Number input buttons */
.stNumberInput button {
    background: #c6e84a !important;
    color: #12351f !important;
}

/* Predict button */
.stButton button {
    background: #c6e84a !important;
    border: 0 !important;
    font-weight: 700 !important;
}

.stButton button,
.stButton button * {
    color: #12351f !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #ffffff !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #c6e84a !important;
}

/* Data table */
[data-testid="stDataFrame"] {
    color: #12351f !important;
}

</style>
""", unsafe_allow_html=True)


def tier_of(rating):
    for cutoff, name, color in TIERS:
        if rating >= cutoff:
            return name, color


def card(label, value):
    return (
        f'<div class="card">'
        f'<small>{label}</small>'
        f'<h2>{value}</h2>'
        f'</div>'
    )


@st.cache_data
def load(source):
    df = pd.read_csv(source)

    for col in ["Age", "Overall rating", "Potential"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.extract(r"^(\d+)")[0],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Age", "Overall rating", "Potential"]
    ).drop_duplicates(subset="ID")

    df["Team"] = (
        df["Team & Contract"]
        .astype(str)
        .str.split("\n")
        .str[0]
    )

    df["Tier"] = df["Overall rating"].apply(
        lambda r: tier_of(r)[0]
    )

    return df.reset_index(drop=True)


@st.cache_resource
def train(df):
    X = df[["Age", "Potential"]]
    y = df["Overall rating"]

    return RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ).fit(X, y)


try:
    df = load(CSV_PATH)

except FileNotFoundError:
    up = st.file_uploader(
        "Upload the Kaggle CSV file",
        type="csv"
    )

    if not up:
        st.stop()

    df = load(up)


model = train(df)

st.markdown(
    '<div class="card">'
    '<small>🏟️ PLAYER PERFORMANCE ESTIMATOR</small>'
    '<h1>⚽ Football Prediction</h1>'
    '<p>Explore players, predict their overall rating '
    'and see where they rank.</p>'
    '</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs([
    "⚽ Predict a player",
    "🏆 Best / Average / Weak",
    "👟 Weakest players"
])


with tab1:
    name = st.selectbox("Player name", df["name"])

    p = df[df["name"] == name].iloc[0]

    c = st.columns(4)

    details = [
        ("Age", f"{int(p.Age)} years"),
        ("Overall Rating", f"{int(p['Overall rating'])} / 100"),
        ("Potential", f"{int(p.Potential)} / 100"),
        ("Team", p.Team)
    ]

    for col, (k, v) in zip(c, details):
        col.markdown(
            card(k, v),
            unsafe_allow_html=True
        )

    t, color = tier_of(p["Overall rating"])

    st.markdown(
        f'Current tier: <span class="badge" '
        f'style="background:{color}">{t}</span>',
        unsafe_allow_html=True
    )

    st.subheader("🥅 Predict Player Rating")

    a, b = st.columns(2)

    age = a.number_input(
        "Player age",
        min_value=15,
        max_value=45,
        value=int(p.Age)
    )

    pot = b.number_input(
        "Player potential",
        min_value=40,
        max_value=99,
        value=int(p.Potential)
    )

    if st.button(
        "⚽ Predict Overall Rating",
        use_container_width=True
    ):
        pred = float(
            model.predict(
                pd.DataFrame({
                    "Age": [age],
                    "Potential": [pot]
                })
            )[0]
        )

        t, color = tier_of(pred)

        st.markdown(
            card(
                "Predicted overall rating",
                f"{pred:.1f} / 100"
            )
            + f'<span class="badge" '
            f'style="background:{color}">{t}</span>',
            unsafe_allow_html=True
        )

        same = df[
            df["Best position"] == p["Best position"]
        ]

        pct = (
            same["Overall rating"] < pred
        ).mean() * 100

        st.write(
            f"Among {len(same)} {p['Best position']} players, "
            f"this beats {pct:.0f}%. "
            f"Position average: "
            f"{same['Overall rating'].mean():.1f}"
        )


with tab2:
    f1, f2 = st.columns(2)

    tier = f1.selectbox(
        "Tier",
        ["All"] + [t[1] for t in TIERS]
    )

    pos = f2.selectbox(
        "Position",
        ["All"] + sorted(
            df["Best position"].dropna().unique()
        )
    )

    view = df

    if tier != "All":
        view = view[view["Tier"] == tier]

    if pos != "All":
        view = view[view["Best position"] == pos]

    st.dataframe(
        view.sort_values(
            "Overall rating",
            ascending=False
        ).head(10)[[
            "name",
            "Age",
            "Overall rating",
            "Potential",
            "Best position",
            "Team",
            "Tier"
        ]],
        use_container_width=True,
        hide_index=True
    )

    st.subheader("🏆 Players per tier")

    st.bar_chart(
        df["Tier"].value_counts().reindex(
            [t[1] for t in TIERS]
        )
    )


with tab3:
    pos3 = st.selectbox(
        "Position",
        ["All"] + sorted(
            df["Best position"].dropna().unique()
        ),
        key="weak_pos"
    )

    weak = (
        df if pos3 == "All"
        else df[df["Best position"] == pos3]
    )

    weak = weak.sort_values(
        ["Overall rating", "Potential"]
    ).reset_index(drop=True)

    if weak.empty:
        st.warning("No players found for this position.")

    else:
        w = weak.iloc[0]

        t, color = tier_of(w["Overall rating"])

        st.subheader("👟 Weakest player")

        c = st.columns(4)

        details = [
            ("Name", w["name"]),
            ("Age", f"{int(w.Age)} years"),
            ("Overall Rating", f"{int(w['Overall rating'])} / 100"),
            ("Potential", f"{int(w.Potential)} / 100")
        ]

        for col, (k, v) in zip(c, details):
            col.markdown(
                card(k, v),
                unsafe_allow_html=True
            )

        st.markdown(
            f'Tier: <span class="badge" '
            f'style="background:{color}">{t}</span>',
            unsafe_allow_html=True
        )

        avg = df["Overall rating"].mean()

        st.write(
            f"Dataset average rating is {avg:.1f}. "
            f"This player is "
            f"{avg - w['Overall rating']:.1f} "
            f"points below it."
        )

        st.subheader("📉 10 weakest players")

        st.dataframe(
            weak.head(10)[[
                "name",
                "Age",
                "Overall rating",
                "Potential",
                "Best position",
                "Team",
                "Tier"
            ]],
            use_container_width=True,
            hide_index=True
        )
