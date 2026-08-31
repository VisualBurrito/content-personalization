"""
app.py
Netflix-style Content Recommendation System — Portfolio Dashboard
Reads from local DuckDB. Three tabs: Genre Correlation, Movie Recommendations, A/B Test Results.
"""

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import json
import os

# ── config ────────────────────────────────────────────────────────────────────
# use MotherDuck in production, local DuckDB in development
MOTHERDUCK_TOKEN = os.environ.get("MOTHERDUCK", "")
if MOTHERDUCK_TOKEN:
    DB_PATH = f"md:movielens_32m?motherduck_token={MOTHERDUCK_TOKEN}"
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), "../../dbt_code/dev.duckdb")
    
N_FACTORS = 50
TOP_K     = 10

st.set_page_config(
    page_title="Content Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Syne:wght@700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0D1117;
    color: #E6EDF3;
}

.main { background-color: #0D1117; }

h1, h2, h3 {
    font-family: 'Syne', sans-serif;
    letter-spacing: -0.02em;
}

.hero {
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid #21262D;
    margin-bottom: 2rem;
}

.hero-tag {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #E8A838;
    margin-bottom: 0.75rem;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.1;
    color: #E6EDF3;
    margin-bottom: 0.5rem;
}

.hero-sub {
    font-size: 1rem;
    color: #8B949E;
    max-width: 640px;
    line-height: 1.6;
}

.metric-row {
    display: flex;
    gap: 1.5rem;
    margin: 1.5rem 0;
    flex-wrap: wrap;
}

.metric-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    min-width: 160px;
    flex: 1;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #E8A838;
    line-height: 1;
    margin-bottom: 4px;
}

.metric-label {
    font-size: 12px;
    color: #8B949E;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #E6EDF3;
    margin-bottom: 0.25rem;
}

.section-sub {
    font-size: 13px;
    color: #8B949E;
    margin-bottom: 1.5rem;
}

.rec-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.rec-rank {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #E8A838;
    min-width: 28px;
}

.rec-title {
    font-size: 14px;
    color: #E6EDF3;
    flex: 1;
    margin: 0 1rem;
}

.rec-score {
    font-size: 13px;
    color: #8B949E;
    font-variant-numeric: tabular-nums;
}

.ab-card {
    background: #161B22;
    border: 1px solid #21262D;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
}

.ab-label {
    font-size: 11px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8B949E;
    margin-bottom: 0.5rem;
}

.ab-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
}

.pill {
    display: inline-block;
    font-size: 11px;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.pill-positive { background: #1B3A2D; color: #3FB950; }
.pill-neutral  { background: #21262D; color: #8B949E; }

.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: transparent;
    border-bottom: 1px solid #21262D;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    color: #8B949E;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    font-weight: 500;
    padding: 0.75rem 1.5rem;
    border-bottom: 2px solid transparent;
}

.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #E6EDF3 !important;
    border-bottom: 2px solid #E8A838 !important;
}

div[data-testid="stSelectbox"] label,
div[data-testid="stTextInput"] label {
    font-size: 12px;
    color: #8B949E;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div > input {
    background: #161B22 !important;
    border: 1px solid #21262D !important;
    color: #E6EDF3 !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)

# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def get_connection():
    if MOTHERDUCK_TOKEN:
        return duckdb.connect(DB_PATH)
    return duckdb.connect(DB_PATH, read_only=True)

@st.cache_data
def load_genre_correlation():
    con = get_connection()
    return con.execute("""
        select genre_a, genre_b, pearson_r,
               correlation_strength, correlation_direction
        from my_db.main.mart_genre_correlation
    """).df()

@st.cache_data
def load_movie_correlation():
    con = get_connection()
    return con.execute("""
        select title_a, title_b, pearson_r
        from my_db.main.mart_movie_correlation
        order by pearson_r desc
    """).df()

@st.cache_data
def load_dim_movies():
    con = get_connection()
    return con.execute("""
        select movie_id, title, genres, avg_rating,
               total_ratings, tmdb_rating, popularity_score
        from my_db.main.dim_movies
        order by total_ratings desc
    """).df()

@st.cache_data
def load_ab_results():
    con = get_connection()
    try:
        return con.execute("select * from my_db.main.mart_ab_test_results").df()
    except Exception:
        return None

@st.cache_data
def load_svd_metrics():
    metrics_path = os.path.join(
        os.path.dirname(__file__), "../../python/modeling/svd_metrics.json"
    )
    try:
        with open(metrics_path) as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_ratings_sample():
    """Load a sample of ratings for SVD — cached so it only runs once."""
    con = get_connection()
    return con.execute("""
        with active_users as (
            select user_id
            from my_db.main.stg_movielens__ratings
            group by user_id
            having count(*) >= 20
            limit 20000
        )
        select r.user_id, r.movie_id, r.rating
        from my_db.main.stg_movielens__ratings r
        inner join active_users a on r.user_id = a.user_id
    """).df()

@st.cache_data
def load_movies_with_recommendations():
    """Only load movies that actually have pre-computed recommendations."""
    con = get_connection()
    return con.execute("""
        select distinct m.movie_id, m.title
        from my_db.main.dim_movies m
        inner join mart_user_profiles p on m.movie_id = p.movie_id
        where m.title is not null
        order by m.title
    """).df()

# ── hero ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-tag">Portfolio Project — Analytics Engineering + ML</div>
    <div class="hero-title">Content Recommendation Engine</div>
    <div class="hero-sub">
        A Netflix-inspired recommendation system built on MovieLens 32M.
        DBT models the data. Python finds the signal. SVD ranks what to watch next.
    </div>
</div>
""", unsafe_allow_html=True)

# ── top metrics ───────────────────────────────────────────────────────────────

svd_metrics = load_svd_metrics()
ab_df       = load_ab_results()
movies_df   = load_dim_movies()
corr_df     = load_movie_correlation()

precision   = f"{svd_metrics['precision_at_k']:.0%}" if svd_metrics else "—"
ndcg        = f"{svd_metrics['ndcg_at_k']:.2f}"      if svd_metrics else "—"
lift        = f"+{ab_df['lift_pct'].iloc[0]:.1f}%"   if ab_df is not None else "—"
n_movies    = f"{len(movies_df):,}"

st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{n_movies}</div>
        <div class="metric-label">Movies modeled</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">32M</div>
        <div class="metric-label">Ratings ingested</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{precision}</div>
        <div class="metric-label">Precision @ 10</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{ndcg}</div>
        <div class="metric-label">NDCG @ 10</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">{lift}</div>
        <div class="metric-label">SVD lift vs baseline</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "  Genre Correlation  ",
    "  Movie Recommendations  ",
    "  A/B Test Results  "
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GENRE CORRELATION HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("""
    <div class="section-header">Genre Correlation Matrix</div>
    <div class="section-sub">
        Pearson correlation between genres computed from shared user ratings.
        Strong positive pairs drive cross-genre recommendations.
        Strong negatives signal distinct audiences — cross-recommending hurts precision.
    </div>
    """, unsafe_allow_html=True)

    genre_df = load_genre_correlation()

    # build symmetric matrix
    genres = sorted(set(genre_df.genre_a) | set(genre_df.genre_b))
    matrix_dict = {}
    for _, row in genre_df.iterrows():
        matrix_dict[(row.genre_a, row.genre_b)] = row.pearson_r
        matrix_dict[(row.genre_b, row.genre_a)] = row.pearson_r

    z = []
    for g1 in genres:
        row_vals = []
        for g2 in genres:
            if g1 == g2:
                row_vals.append(1.0)
            else:
                row_vals.append(matrix_dict.get((g1, g2), None))
        z.append(row_vals)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=genres,
        y=genres,
        colorscale=[
            [0.0,  "#7B1A1A"],
            [0.3,  "#21262D"],
            [0.5,  "#161B22"],
            [0.7,  "#1B3A2D"],
            [1.0,  "#2EA043"],
        ],
        zmid=0,
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 10, "color": "#E6EDF3"},
        hoverongaps=False,
        hovertemplate="<b>%{y} × %{x}</b><br>Pearson r = %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Pearson r", font=dict(color="#8B949E", size=12)),
            tickfont=dict(color="#8B949E"),
            bgcolor="#161B22",
            bordercolor="#21262D",
            borderwidth=1,
            tickvals=[-1, -0.5, 0, 0.5, 1],
        )
    ))

    fig.update_layout(
        paper_bgcolor="#0D1117",
        plot_bgcolor="#0D1117",
        font=dict(color="#E6EDF3", family="Inter"),
        height=600,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(
            tickfont=dict(size=11, color="#8B949E"),
            gridcolor="#21262D",
            tickangle=-35
        ),
        yaxis=dict(
            tickfont=dict(size=11, color="#8B949E"),
            gridcolor="#21262D"
        )
    )
    st.plotly_chart(fig, width='stretch')

    # top pairs table
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header" style="font-size:1rem;">Strongest positive pairs</div>', unsafe_allow_html=True)
        top_pos = genre_df[genre_df.pearson_r > 0].nlargest(8, "pearson_r")[
            ["genre_a", "genre_b", "pearson_r"]
        ].reset_index(drop=True)
        top_pos.columns = ["Genre A", "Genre B", "Pearson r"]
        top_pos["Pearson r"] = top_pos["Pearson r"].map("{:.3f}".format)
        st.dataframe(top_pos, hide_index=True, width='stretch')

    with col2:
        st.markdown('<div class="section-header" style="font-size:1rem;">Strongest negative pairs</div>', unsafe_allow_html=True)
        top_neg = genre_df[genre_df.pearson_r < 0].nsmallest(8, "pearson_r")[
            ["genre_a", "genre_b", "pearson_r"]
        ].reset_index(drop=True)
        top_neg.columns = ["Genre A", "Genre B", "Pearson r"]
        top_neg["Pearson r"] = top_neg["Pearson r"].map("{:.3f}".format)
        st.dataframe(top_neg, hide_index=True, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MOVIE RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("""
    <div class="section-header">Movie Recommendations</div>
    <div class="section-sub">
        Two recommendation modes. Similar titles uses Pearson correlation
        from co-rating patterns. Personalized uses SVD matrix factorization
        trained on 20,000 active users.
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Recommendation mode",
        ["Similar titles (correlation)", "Personalized (SVD)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    if mode == "Similar titles (correlation)":
        all_titles = sorted({
            t for t in set(corr_df.title_a) | set(corr_df.title_b)
            if isinstance(t, str)
        })
        selected = st.selectbox("Choose a movie", all_titles)

        if selected:
        # find matches in both directions
            as_a = corr_df[corr_df.title_a == selected].copy()
            as_a["other"] = as_a["title_b"]

            as_b = corr_df[corr_df.title_b == selected].copy()
            as_b["other"] = as_b["title_a"]

            matches = pd.concat([as_a, as_b], ignore_index=True)
            matches = matches[matches["other"].notna()]
            matches = matches.drop_duplicates(subset="other")
            matches = matches.nlargest(TOP_K, "pearson_r")
            n_found = len(matches)
            if n_found < TOP_K:
                st.markdown(f"""
                <div style="font-size:12px;color:#8B949E;margin-bottom:0.5rem;">
                    {n_found} correlated titles found — try a more popular movie for more results
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin: 1.5rem 0 0.75rem; font-size:13px; color:#8B949E;">
                Top {TOP_K} titles most correlated with <span style="color:#E6EDF3;font-weight:500;">{selected}</span>
            </div>
            """, unsafe_allow_html=True)

            for i, row in enumerate(matches.itertuples(), 1):
                bar_width = int(row.pearson_r * 100)
                st.markdown(f"""
                <div class="rec-card">
                    <span class="rec-rank">{i}</span>
                    <span class="rec-title">{row.other}</span>
                    <span class="rec-score">r = {row.pearson_r:.3f}</span>
                </div>
                <div style="height:3px;background:#21262D;border-radius:2px;margin-bottom:2px;">
                    <div style="height:3px;width:{bar_width}%;background:#E8A838;border-radius:2px;"></div>
                </div>
                """, unsafe_allow_html=True)

    else:
        movies_list = load_movies_with_recommendations()
        seed_title = st.selectbox(
            "Pick a movie to find users who rated it",
            sorted(movies_list.title.dropna().tolist())
        )

        if seed_title:
            con = get_connection()
            seed_id = movies_list[movies_list.title == seed_title].movie_id.values

            if len(seed_id) == 0:
                st.warning("Movie not found.")
            else:
                seed_id = int(seed_id[0])

                user_row = con.execute(f"""
                    select user_id from my_db.main.mart_user_profiles
                    where movie_id = {seed_id} and rating >= 4.0
                    limit 1
                """).df()

                if user_row.empty:
                    st.warning("No users in sample rated this movie highly. Try another.")
                else:
                    user_id = int(user_row.user_id.iloc[0])

                    profile = con.execute(f"""
                        select title, rating from my_db.main.mart_user_profiles
                        where user_id = {user_id}
                        order by rating desc
                        limit 8
                    """).df()

                    recs = con.execute(f"""
                        select rank, title, predicted_rating
                        from my_db.main.mart_svd_recommendations_full
                        where user_id = {user_id}
                        order by rank
                    """).df()

                    col1, col2 = st.columns([1, 1])

                    with col1:
                        st.markdown('<div class="section-header" style="font-size:1rem;">This user rated highly</div>', unsafe_allow_html=True)
                        for _, row in profile.iterrows():
                            st.markdown(f"""
                            <div class="rec-card">
                                <span class="rec-title">{row.title}</span>
                                <span class="rec-score">{'★' * int(row.rating)} {row.rating:.1f}</span>
                            </div>
                            """, unsafe_allow_html=True)

                    with col2:
                        st.markdown('<div class="section-header" style="font-size:1rem;">SVD recommends</div>', unsafe_allow_html=True)
                        for _, row in recs.iterrows():
                            st.markdown(f"""
                            <div class="rec-card">
                                <span class="rec-rank">{int(row['rank'])}</span>
                                <span class="rec-title">{row.title}</span>
                                <span class="rec-score">{row.predicted_rating:.2f} pred</span>
                            </div>
                            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — A/B TEST RESULTS
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("""
    <div class="section-header">A/B Test: SVD vs Popularity Baseline</div>
    <div class="section-sub">
        5,000 users randomly assigned to control (popularity-based) or treatment (SVD).
        Evaluated on Precision@10 against a held-out 20% test set.
        Two-sided t-test for significance.
    </div>
    """, unsafe_allow_html=True)

    if ab_df is None:
        st.error("A/B test results not found. Run python/modeling/ab_test.py first.")
    else:
        row = ab_df.iloc[0]
        ctrl  = float(row.control_precision)
        treat = float(row.treatment_precision)
        lift  = float(row.lift_pct)
        pval  = float(row.p_value)
        sig   = bool(row.significant)

        # top metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="ab-card">
                <div class="ab-label">Control · Popularity</div>
                <div class="ab-value" style="color:#8B949E;">{ctrl:.1%}</div>
                <div style="font-size:12px;color:#8B949E;margin-top:6px;">Precision @ 10</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="ab-card">
                <div class="ab-label">Treatment · SVD</div>
                <div class="ab-value" style="color:#E8A838;">{treat:.1%}</div>
                <div style="font-size:12px;color:#8B949E;margin-top:6px;">Precision @ 10</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="ab-card">
                <div class="ab-label">Lift</div>
                <div class="ab-value" style="color:#3FB950;">+{lift:.1f}%</div>
                <div style="font-size:12px;color:#8B949E;margin-top:6px;">Relative improvement</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            pval_display = "< 0.0001" if pval < 0.0001 else f"{pval:.4f}"
            sig_pill = '<span class="pill pill-positive">✓ Significant</span>' if sig else '<span class="pill pill-neutral">Not significant</span>'
            st.markdown(f"""
            <div class="ab-card">
                <div class="ab-label">p-value</div>
                <div class="ab-value" style="color:#E6EDF3;font-size:1.6rem;">{pval_display}</div>
                <div style="margin-top:8px;">{sig_pill}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # bar chart comparison
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Control (Popularity)",
            x=["Control · Popularity", "Treatment · SVD"],
            y=[ctrl, treat],
            marker_color=["#30363D", "#E8A838"],
            text=[f"{ctrl:.1%}", f"{treat:.1%}"],
            textposition="outside",
            textfont=dict(color="#E6EDF3", size=14, family="Syne"),
            width=0.4
        ))

        fig2.add_shape(
            type="line",
            x0=-0.5, x1=1.5,
            y0=ctrl, y1=ctrl,
            line=dict(color="#8B949E", width=1, dash="dot")
        )
        fig2.add_annotation(
            x=1.5, y=ctrl,
            text=f"Baseline {ctrl:.1%}",
            showarrow=False,
            font=dict(color="#8B949E", size=11),
            xanchor="left"
        )

        fig2.update_layout(
            paper_bgcolor="#0D1117",
            plot_bgcolor="#0D1117",
            font=dict(color="#E6EDF3", family="Inter"),
            height=360,
            showlegend=False,
            margin=dict(l=20, r=80, t=40, b=20),
            yaxis=dict(
                tickformat=".0%",
                tickfont=dict(color="#8B949E"),
                gridcolor="#21262D",
                range=[0, max(ctrl, treat) * 1.3]
            ),
            xaxis=dict(
                tickfont=dict(color="#8B949E", size=13),
                gridcolor="rgba(0,0,0,0)"
            ),
            bargap=0.5
        )
        st.plotly_chart(fig2, width='stretch')

        # methodology notes
        st.markdown("""
        <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;padding:1.25rem 1.5rem;margin-top:1rem;">
            <div style="font-size:12px;font-weight:600;letter-spacing:0.1em;color:#8B949E;text-transform:uppercase;margin-bottom:0.75rem;">
                Methodology
            </div>
            <div style="font-size:13px;color:#8B949E;line-height:1.8;">
                • Users randomly split 50/50 into control and treatment groups<br>
                • 20% of each user's ratings held out as test set before model training<br>
                • Control receives top-N most popular unseen movies<br>
                • Treatment receives SVD predictions trained on remaining 80% of ratings<br>
                • Precision@10 measured against held-out test items rated ≥ 4.0 stars<br>
                • Two-sided independent t-test; α = 0.05
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-top:4rem;padding-top:1.5rem;border-top:1px solid #21262D;
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
    <div style="font-size:12px;color:#8B949E;">
        Built with DBT · DuckDB · Python · Streamlit &nbsp;|&nbsp;
        Data: MovieLens 32M + TMDB
    </div>
    <div style="font-size:12px;color:#8B949E;">
        <a href="https://github.com/VisualBurrito/content_personalization"
           style="color:#E8A838;text-decoration:none;">
            github.com/VisualBurrito/content_personalization
        </a>
    </div>
</div>
""", unsafe_allow_html=True)
