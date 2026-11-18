import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# ------------------------------------------------------------------ look & feel
LAV, LAV_DARK, BUR, BUR_SOFT = "#EDE7F6", "#D1C4E9", "#6D1A36", "#A0516B"
FEATS = ["TV", "Radio", "Newspaper"]
CH_COLOR = {"TV": BUR, "Radio": "#B5547A", "Newspaper": "#9E9AB0"}

st.set_page_config(page_title="Sales Studio | CodSoft Task 3", page_icon="💜", layout="wide")

CSS = """
<style>
.stApp { background: linear-gradient(160deg, __LAV__ 0%, #F6F1FC 100%); }
header[data-testid="stHeader"] { background: transparent; }
html, body, p, li, label, h1, h2, h3, h4, h5, h6, span, div, td, th { color: __BUR__; }
.hero { padding: 1.4rem 1.8rem; border-radius: 22px; background: __BUR__; margin-bottom: 1rem; }
.hero, .hero * { color: __LAV__ !important; }
.hero h1 { margin: 0; font-size: 2.6rem; letter-spacing: .02em; }
.hero p { margin: .3rem 0 0 0; opacity: .85; font-size: 1.05rem; }
.pill { display:inline-block; padding:.15rem .8rem; border-radius:999px; background:__LAV__; margin-bottom:.6rem;
        font-size:.78rem; letter-spacing:.14em; }
.hero .pill, .hero .pill * { color: __BUR__ !important; }
.card { background:#FFFFFF99; border:2px solid __BUR__; border-radius:18px; padding:1rem 1.2rem; text-align:center; }
.card .t { font-size:.8rem; letter-spacing:.12em; text-transform:uppercase; opacity:.8; }
.card .v { font-size:2.5rem; font-weight:700; line-height:1.15; }
.card .s { font-size:.9rem; opacity:.85; }
.stButton > button { background:__BUR__; border:none; border-radius:12px; padding:.55rem 1.4rem; font-weight:600; }
.stButton > button, .stButton > button * { color:__LAV__ !important; }
.stTabs [data-baseweb="tab"] { font-weight:600; }
.stTabs [aria-selected="true"] { background:__LAV_DARK__; border-radius:10px 10px 0 0; }
[data-testid="stDataFrame"] { border:2px solid __BUR__; border-radius:12px; }
</style>
"""
st.markdown(CSS.replace("__LAV_DARK__", LAV_DARK).replace("__LAV__", LAV).replace("__BUR__", BUR),
            unsafe_allow_html=True)

def card(title, value, sub=""):
    return f'<div class="card"><div class="t">{title}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'

def style_ax(ax):
    ax.set_facecolor("#FFFFFF66")
    for s in ax.spines.values():
        s.set_color(BUR_SOFT)
    ax.tick_params(colors=BUR)
    ax.xaxis.label.set_color(BUR)
    ax.yaxis.label.set_color(BUR)
    ax.title.set_color(BUR)
    ax.grid(alpha=.25, color=BUR_SOFT)

def new_fig(w=7, h=3.8, cols=1):
    fig, ax = plt.subplots(1, cols, figsize=(w, h))
    fig.patch.set_facecolor(LAV)
    for a in (ax if cols > 1 else [ax]):
        style_ax(a)
    return fig, ax

# ------------------------------------------------------------------ data + models
HERE = Path(__file__).parent

@st.cache_data
def load_data():
    for name in ["CodSoft_Sales_Prediction_Dataset.xlsx", "advertising.csv"]:
        p = HERE / name
        if p.exists():
            d = pd.read_excel(p) if name.endswith("xlsx") else pd.read_csv(p)
            names = {"tv": "TV", "radio": "Radio", "newspaper": "Newspaper", "sales": "Sales"}
            d = d.rename(columns={c: names[str(c).lower()] for c in d.columns if str(c).lower() in names})
            return d[["TV", "Radio", "Newspaper", "Sales"]].dropna()
    return None

df = load_data()
if df is None:
    st.error("Dataset not found. Put CodSoft_Sales_Prediction_Dataset.xlsx next to app.py.")
    st.stop()

MAX = {c: float(df[c].max()) for c in FEATS}

@st.cache_resource
def train_models(data):
    kw = dict(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=42)
    mean = GradientBoostingRegressor(**kw).fit(data[FEATS], data["Sales"])
    low = GradientBoostingRegressor(loss="quantile", alpha=0.1, **kw).fit(data[FEATS], data["Sales"])
    high = GradientBoostingRegressor(loss="quantile", alpha=0.9, **kw).fit(data[FEATS], data["Sales"])
    return mean, low, high

M_MEAN, M_LOW, M_HIGH = train_models(df)

def predict(tv, radio, news):
    p = pd.DataFrame([[tv, radio, news]], columns=FEATS)
    s = float(M_MEAN.predict(p)[0])
    return s, min(float(M_LOW.predict(p)[0]), s), max(float(M_HIGH.predict(p)[0]), s)

@st.cache_data
def compare_models(data):
    setups = {
        "Linear, TV only": (LinearRegression(), ["TV"]),
        "Linear, TV + Radio": (LinearRegression(), ["TV", "Radio"]),
        "Linear, all 3 channels": (LinearRegression(), FEATS),
        "Random Forest": (RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=42), FEATS),
        "Gradient Boosting (used here)": (GradientBoostingRegressor(n_estimators=150, learning_rate=0.05,
                                                                    max_depth=2, random_state=42), FEATS),
    }
    rows = []
    for name, (m, cols) in setups.items():
        r2 = cross_val_score(m, data[cols], data["Sales"], cv=5, scoring="r2").mean()
        mae = -cross_val_score(m, data[cols], data["Sales"], cv=5, scoring="neg_mean_absolute_error").mean()
        rows.append([name, round(r2, 3), round(mae, 2)])
    return pd.DataFrame(rows, columns=["Model", "R² (5-fold CV)", "Average error (MAE)"])

# ------------------------------------------------------------------ header
st.markdown("""
<div class="hero">
  <div class="pill">CODSOFT · DATA SCIENCE INTERNSHIP · TASK 3</div>
  <h1>💜 Sales Studio</h1>
  <p>Move the ad budgets, see the predicted sales, and find the smartest way to split your money.</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🎛️ Simulator", "💰 Budget Planner", "📊 Insights"])

# ------------------------------------------------------------------ tab 1: simulator
with tab1:
    left, right = st.columns([1, 1.5], gap="large")
    with left:
        st.subheader("Your ad budgets")
        st.caption("Thousand dollars per market")
        tv = st.slider("📺 TV", 0, int(MAX["TV"]), 150, 5)
        radio = st.slider("📻 Radio", 0, int(MAX["Radio"]), 25, 1)
        news = st.slider("📰 Newspaper", 0, int(MAX["Newspaper"]), 30, 1)

    s, lo, hi = predict(tv, radio, news)
    cur = {"TV": tv, "Radio": radio, "Newspaper": news}
    gains = {}
    for ch in FEATS:
        more = dict(cur)
        more[ch] = min(cur[ch] + 10, MAX[ch])
        gains[ch] = predict(more["TV"], more["Radio"], more["Newspaper"])[0] - s
    best_ch = max(gains, key=gains.get)

    with right:
        c1, c2, c3 = st.columns(3)
        c1.markdown(card("Predicted sales", f"{s:.1f}", "thousand units"), unsafe_allow_html=True)
        c2.markdown(card("Likely range", f"{lo:.1f} – {hi:.1f}", "80% of the time"), unsafe_allow_html=True)
        c3.markdown(card("Next 10 goes to", best_ch, f"+{gains[best_ch]:.2f} sales"), unsafe_allow_html=True)

        st.write("")
        fig, ax = new_fig(7.2, 3.8)
        for ch in FEATS:
            xs = np.linspace(0, MAX[ch], 40)
            grid = pd.DataFrame({c: [cur[c]] * 40 for c in FEATS})
            grid[ch] = xs
            ax.plot(xs / MAX[ch] * 100, M_MEAN.predict(grid), lw=2.8, color=CH_COLOR[ch], label=ch)
            ax.scatter([cur[ch] / MAX[ch] * 100], [s], s=90, color=CH_COLOR[ch], edgecolor=BUR, zorder=5)
        ax.set_xlabel("Spend on one channel (% of the highest spend in the data)")
        ax.set_ylabel("Predicted sales")
        ax.set_title("What if you spend more or less on just one channel?")
        ax.legend(facecolor=LAV, edgecolor=BUR_SOFT, labelcolor=BUR)
        st.pyplot(fig)
        plt.close(fig)

# ------------------------------------------------------------------ tab 2: budget planner
with tab2:
    st.subheader("Find the best split for a total budget")
    cA, cB = st.columns([2, 1])
    total = cA.slider("Total budget (thousand dollars)", 20, 400, 200, 10)
    allow_news = cB.toggle("Allow Newspaper spend", value=True)

    rows = []
    for t in range(0, int(min(total, MAX["TV"])) + 1, 5):
        for r in range(0, int(min(total - t, MAX["Radio"])) + 1, 5):
            n = total - t - r
            if n <= MAX["Newspaper"] and (allow_news or n == 0):
                rows.append((t, r, n))

    if not rows:
        st.warning("That budget is bigger than anything seen in the data without Newspaper. "
                   "Lower the budget or allow Newspaper spend.")
    else:
        grid = pd.DataFrame(rows, columns=FEATS)
        grid["Sales"] = M_MEAN.predict(grid[FEATS])
        top = grid.loc[grid["Sales"].idxmax()]
        even = total / 3
        s_even = predict(even, even, even)[0]
        s_tv = predict(total, 0, 0)[0]

        k1, k2, k3 = st.columns(3)
        k1.markdown(card("Best predicted sales", f"{top['Sales']:.1f}", "with the best split"), unsafe_allow_html=True)
        k2.markdown(card("vs spending evenly", f"+{top['Sales'] - s_even:.1f}", f"even split: {s_even:.1f}"), unsafe_allow_html=True)
        k3.markdown(card("vs all on TV", f"+{top['Sales'] - s_tv:.1f}", f"all on TV: {s_tv:.1f}"), unsafe_allow_html=True)
        st.write("")

        g1, g2 = st.columns(2)
        with g1:
            fig, ax = new_fig(5, 4)
            vals = [top["TV"], top["Radio"], top["Newspaper"]]
            nz = [(n, v) for n, v in zip(FEATS, vals) if v > 0]
            ax.pie([v for _, v in nz], labels=[f"{n}\n{v:.0f}" for n, v in nz],
                   colors=[CH_COLOR[n] for n, _ in nz], startangle=90,
                   wedgeprops=dict(width=0.45, edgecolor=LAV, linewidth=3),
                   textprops=dict(color=BUR, fontsize=11, fontweight="bold"))
            ax.set_title("Recommended split")
            ax.grid(False)
            st.pyplot(fig)
            plt.close(fig)
        with g2:
            fig, ax = new_fig(5, 4)
            names = ["Spend evenly", "All on TV", "Best split"]
            vals = [s_even, s_tv, top["Sales"]]
            bars = ax.bar(names, vals, color=[BUR_SOFT, "#9E9AB0", BUR], width=0.6)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}", ha="center", color=BUR, fontweight="bold")
            ax.set_ylabel("Predicted sales")
            ax.set_title("How much better is the best split?")
            ax.set_ylim(0, max(vals) * 1.18)
            st.pyplot(fig)
            plt.close(fig)

        st.info(f"Best split for {total}: TV {top['TV']:.0f}, Radio {top['Radio']:.0f}, Newspaper {top['Newspaper']:.0f} "
                f"(thousand dollars). Splits are tested in steps of 5 and stay inside the ranges seen in the data.")

# ------------------------------------------------------------------ tab 3: insights
with tab3:
    st.subheader("What drives sales?")
    i1, i2 = st.columns(2, gap="large")
    with i1:
        corr = df.corr()["Sales"].drop("Sales").sort_values()
        fig, ax = new_fig(5.5, 3.4)
        ax.barh(corr.index, corr.values, color=[CH_COLOR[c] for c in corr.index])
        for i, v in enumerate(corr.values):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", color=BUR, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_title("Correlation with Sales")
        st.pyplot(fig)
        plt.close(fig)
    with i2:
        st.markdown("""
**What the data says**
- **TV** is the strongest driver of sales, then **Radio**.
- **Newspaper adds almost nothing** once TV and Radio are in the model.
- TV and Radio work **together**, so tree models beat straight-line models.
""")

    st.subheader("Model comparison")
    st.dataframe(compare_models(df), width="stretch", hide_index=True)
    st.caption("5-fold cross-validation on 200 markets. Higher R² is better; lower error is better.")

    st.subheader("Good to know")
    st.markdown(f"""
- Built from **{len(df)} markets** with TV, Radio and Newspaper budgets (thousand dollars) and sales (thousand units).
- The data shows **association, not proof of cause**. Use it as a planning aid, not a guarantee.
- Predictions stay inside the budget ranges in the data: TV up to {MAX['TV']:.0f}, Radio up to {MAX['Radio']:.0f}, Newspaper up to {MAX['Newspaper']:.0f}.
""")

st.markdown("<p style='text-align:center;opacity:.7;margin-top:2rem'>Made for the CodSoft Data Science Internship · Task 3</p>",
            unsafe_allow_html=True)
