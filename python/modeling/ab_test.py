"""
ab_test.py
Simulates an A/B test comparing two recommendation strategies:
  Control:   Popularity-based (most-rated movies the user hasn't seen)
  Treatment: SVD matrix factorization (personalized predictions)
Evaluates using precision@k and a two-proportion z-test for significance.
"""

import duckdb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import json

DB_PATH   = "C:/Users/lsmit/content_personalization/dbt_code/dev.duckdb"
N_FACTORS = 50
TOP_K     = 10
N_USERS   = 5000    # users to include in A/B test
SEED      = 42

np.random.seed(SEED)

# ── data loading ──────────────────────────────────────────────────────────────

def load_data(con):
    print("Loading ratings...")
    df = con.execute(f"""
        with active_users as (
            select user_id
            from stg_movielens__ratings
            group by user_id
            having count(*) >= 20
            limit {N_USERS}
        )
        select r.user_id, r.movie_id, r.rating
        from stg_movielens__ratings r
        inner join active_users a on r.user_id = a.user_id
    """).df()

    movie_counts = df.groupby("movie_id")["rating"].count()
    popular_movies = movie_counts[movie_counts >= 10].index
    df = df[df["movie_id"].isin(popular_movies)]

    print(f"  {len(df):,} ratings, {df.user_id.nunique():,} users, {df.movie_id.nunique():,} movies")
    return df

# ── matrix construction ───────────────────────────────────────────────────────

def build_matrix(df):
    user_idx  = {u: i for i, u in enumerate(df.user_id.unique())}
    movie_idx = {m: i for i, m in enumerate(df.movie_id.unique())}
    idx_movie = {i: m for m, i in movie_idx.items()}

    matrix = csr_matrix(
        (df.rating.values, (df.user_id.map(user_idx), df.movie_id.map(movie_idx))),
        shape=(len(user_idx), len(movie_idx)),
        dtype=float
    )
    return matrix, user_idx, movie_idx, idx_movie

# ── control: popularity-based ─────────────────────────────────────────────────

def popularity_recommendations(matrix, movie_idx, top_k=TOP_K):
    """Recommend the most-rated movies the user hasn't seen."""
    # global popularity = total rating count per movie
    popularity = np.array(matrix.astype(bool).sum(axis=0)).flatten()
    popular_order = np.argsort(popularity)[::-1]

    recs = {}
    for u in range(matrix.shape[0]):
        rated = set(matrix.getrow(u).indices)
        user_recs = [i for i in popular_order if i not in rated][:top_k]
        recs[u] = user_recs
    return recs

# ── treatment: SVD ────────────────────────────────────────────────────────────

def svd_recommendations(matrix, top_k=TOP_K):
    """Recommend using SVD matrix factorization."""
    user_means = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        if row.nnz > 0:
            user_means[i] = row.data.mean()

    matrix_demeaned = matrix.copy().astype(float)
    rows, cols = matrix_demeaned.nonzero()
    matrix_demeaned.data -= user_means[rows]

    U, sigma, Vt = svds(matrix_demeaned, k=N_FACTORS)
    predicted = U @ np.diag(sigma) @ Vt
    predicted += user_means.reshape(-1, 1)
    predicted = np.clip(predicted, 0.5, 5.0)

    recs = {}
    for u in range(matrix.shape[0]):
        pred = predicted[u].copy()
        rated = matrix.getrow(u).indices
        pred[rated] = -999
        recs[u] = list(np.argsort(pred)[::-1][:top_k])
    return recs

# ── evaluation ────────────────────────────────────────────────────────────────

def split_train_test(matrix, test_ratio=0.2, min_ratings=10):
    """Create consistent train/test splits for all users."""
    train_data = matrix.copy().tolil()
    test_data  = {}

    for u in range(matrix.shape[0]):
        row = matrix.getrow(u)
        if row.nnz < min_ratings:
            continue
        rated   = row.indices
        actual  = row.data
        n_test  = max(1, int(len(rated) * test_ratio))
        test_pos = np.random.choice(len(rated), n_test, replace=False)

        # store test items
        test_data[u] = {
            "indices": rated[test_pos],
            "ratings": actual[test_pos]
        }
        # remove test items from training matrix
        for idx in rated[test_pos]:
            train_data[u, idx] = 0

    train_matrix = train_data.tocsr()
    train_matrix.eliminate_zeros()
    return train_matrix, test_data


def precision_at_k(recs, test_data, top_k=TOP_K):
    """Compute precision@k using pre-built test splits."""
    scores = []
    for u, rec_list in recs.items():
        if u not in test_data:
            continue
        test_indices = test_data[u]["indices"]
        test_ratings = test_data[u]["ratings"]
        relevant     = set(test_indices[test_ratings >= 4.0])
        if not relevant:
            continue
        hits = len(set(rec_list) & relevant)
        scores.append(hits / top_k)
    return np.array(scores)

# ── significance test ─────────────────────────────────────────────────────────

def run_significance_test(control_scores, treatment_scores):
    """Two-sided t-test comparing control vs treatment precision scores."""
    t_stat, p_value = stats.ttest_ind(control_scores, treatment_scores)
    lift = (treatment_scores.mean() - control_scores.mean()) / max(control_scores.mean(), 1e-9)

    return {
        "control_precision":   round(float(control_scores.mean()), 4),
        "treatment_precision": round(float(treatment_scores.mean()), 4),
        "lift_pct":            round(float(lift * 100), 2),
        "t_statistic":         round(float(t_stat), 4),
        "p_value":             round(float(p_value), 6),
        "significant":         bool(p_value < 0.05),
        "n_control":           len(control_scores),
        "n_treatment":         len(treatment_scores)
    }

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    con = duckdb.connect(DB_PATH)

    df = load_data(con)
    matrix, user_idx, movie_idx, idx_movie = build_matrix(df)

    # create shared train/test split
    print("Splitting train/test...")
    train_matrix, test_data = split_train_test(matrix)
    print(f"  {len(test_data):,} users with test splits")

    # A/B user assignment
    eligible_users = list(test_data.keys())
    np.random.shuffle(eligible_users)
    mid = len(eligible_users) // 2
    control_users   = eligible_users[:mid]
    treatment_users = eligible_users[mid:]
    print(f"A/B split: {len(control_users):,} control, {len(treatment_users):,} treatment")

    # control: popularity on train matrix
    print("\nRunning control (popularity-based)...")
    all_popularity_recs = popularity_recommendations(train_matrix, movie_idx)
    control_recs = {u: all_popularity_recs[u] for u in control_users}

    # treatment: SVD on train matrix
    print("Running treatment (SVD)...")
    all_svd_recs = svd_recommendations(train_matrix)
    treatment_recs = {u: all_svd_recs[u] for u in treatment_users}

    # evaluate against test set
    print("\nEvaluating both variants...")
    control_scores   = precision_at_k(control_recs, test_data)
    treatment_scores = precision_at_k(treatment_recs, test_data)

    results = run_significance_test(control_scores, treatment_scores)

    print("\n" + "="*50)
    print("A/B TEST RESULTS")
    print("="*50)
    print(f"  Control   (popularity): Precision@{TOP_K} = {results['control_precision']}")
    print(f"  Treatment (SVD):        Precision@{TOP_K} = {results['treatment_precision']}")
    print(f"  Lift:                   {results['lift_pct']:+.1f}%")
    print(f"  p-value:                {results['p_value']}")
    print(f"  Significant (p<0.05):   {results['significant']}")
    print("="*50)

    results_df = pd.DataFrame([results])
    con.execute("drop table if exists mart_ab_test_results")
    con.execute("create table mart_ab_test_results as select * from results_df")
    print("\nSaved mart_ab_test_results to DuckDB")

    with open("python/modeling/ab_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved python/modeling/ab_test_results.json")

    con.close()
    print("\nDone.")

if __name__ == "__main__":
    main()