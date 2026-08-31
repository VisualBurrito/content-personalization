"""
svd_model.py
Trains an SVD matrix factorization recommendation model on MovieLens ratings.
Reads from local DuckDB, evaluates with precision@k and NDCG, saves results.
"""

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
import json

DB_PATH  = "C:/Users/lsmit/content_personalization/dbt_code/dev.duckdb"
N_FACTORS = 50       # number of latent factors
TOP_K     = 10       # recommendations per user for evaluation

# ── data loading ──────────────────────────────────────────────────────────────

def load_ratings(con, max_users=50000, min_ratings=20):
    print(f"Loading ratings for {max_users:,} users...")

    # sample active users — users with more ratings give better signal
    df = con.execute(f"""
        with active_users as (
            select user_id, count(*) as rating_count
            from stg_movielens__ratings
            group by user_id
            having count(*) >= {min_ratings}
            order by rating_count desc
            limit {max_users}
        )
        select r.user_id, r.movie_id, r.rating
        from stg_movielens__ratings r
        inner join active_users a on r.user_id = a.user_id
    """).df()

    # filter to movies with enough ratings in this sample
    movie_counts = df.groupby("movie_id")["rating"].count()
    popular_movies = movie_counts[movie_counts >= 10].index
    df = df[df["movie_id"].isin(popular_movies)]

    print(f"  {len(df):,} ratings")
    print(f"  {df.user_id.nunique():,} users")
    print(f"  {df.movie_id.nunique():,} movies")
    return df

# ── matrix construction ───────────────────────────────────────────────────────

def build_matrix(df):
    print("Building user-movie matrix...")
    user_idx  = {u: i for i, u in enumerate(df.user_id.unique())}
    movie_idx = {m: i for i, m in enumerate(df.movie_id.unique())}
    idx_user  = {i: u for u, i in user_idx.items()}
    idx_movie = {i: m for m, i in movie_idx.items()}

    rows = df.user_id.map(user_idx)
    cols = df.movie_id.map(movie_idx)

    matrix = csr_matrix(
        (df.rating.values, (rows, cols)),
        shape=(len(user_idx), len(movie_idx)),
        dtype=float
    )
    print(f"  Matrix: {matrix.shape[0]:,} users × {matrix.shape[1]:,} movies")
    return matrix, user_idx, movie_idx, idx_user, idx_movie

# ── SVD model ─────────────────────────────────────────────────────────────────

def train_svd(matrix, n_factors=N_FACTORS):
    print(f"Training SVD with {n_factors} latent factors...")

    # compute user means from non-zero entries only
    user_ratings_mean = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        if row.nnz > 0:
            user_ratings_mean[i] = row.data.mean()

    # subtract user mean from non-zero entries only
    matrix_demeaned = matrix.copy().astype(float)
    rows, cols = matrix_demeaned.nonzero()
    matrix_demeaned.data -= user_ratings_mean[rows]

    # SVD
    U, sigma, Vt = svds(matrix_demeaned, k=n_factors)
    sigma_diag = np.diag(sigma)

    # reconstruct and add user means back
    predicted = U @ sigma_diag @ Vt
    predicted += user_ratings_mean.reshape(-1, 1)
    predicted = np.clip(predicted, 0.5, 5.0)

    print(f"  SVD complete. Predicted matrix shape: {predicted.shape}")
    print(f"  Predicted rating range: {predicted.min():.2f} - {predicted.max():.2f}")
    print(f"  Predicted rating mean:  {predicted.mean():.2f}")
    return predicted, user_ratings_mean

# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(matrix, predicted, top_k=TOP_K):
    print(f"Evaluating precision@{top_k} and NDCG@{top_k}...")

    precision_scores = []
    ndcg_scores      = []

    n_users = matrix.shape[0]
    sample_users = np.random.choice(n_users, min(1000, n_users), replace=False)

    for u in sample_users:
        row = matrix.getrow(u)
        if row.nnz < 10:
            continue

        rated_indices = row.indices
        actual_ratings = row.data

        # hold out 20% as test
        n_test = max(1, int(len(rated_indices) * 0.2))
        test_pos = np.random.choice(len(rated_indices), n_test, replace=False)
        test_indices = rated_indices[test_pos]
        test_ratings = actual_ratings[test_pos]
        train_indices = np.delete(rated_indices, test_pos)

        # predicted scores — mask out training items
        pred = predicted[u].copy()
        pred[train_indices] = -999

        # top-k from remaining
        top_k_idx = np.argsort(pred)[::-1][:top_k]
        top_k_set = set(top_k_idx)

        # precision@k — relevant = test items rated >= 4
        relevant = set(test_indices[test_ratings >= 4.0])
        hits = len(top_k_set & relevant)
        precision_scores.append(hits / top_k)

        # NDCG@k over test items
        if len(test_indices) > 0:
            true_rel  = (test_ratings >= 4.0).astype(float)
            pred_rel  = predicted[u][test_indices]
            if true_rel.sum() > 0:
                ndcg_scores.append(ndcg_score([true_rel], [pred_rel]))

    results = {
        "model":             "SVD",
        "n_factors":         N_FACTORS,
        "top_k":             TOP_K,
        "precision_at_k":    round(float(np.mean(precision_scores)), 4),
        "ndcg_at_k":         round(float(np.mean(ndcg_scores)), 4),
        "n_users_evaluated": len(precision_scores)
    }
    print(f"  Precision@{top_k}: {results['precision_at_k']}")
    print(f"  NDCG@{top_k}:      {results['ndcg_at_k']}")
    return results

# ── recommendations ───────────────────────────────────────────────────────────

def get_recommendations(user_id, predicted, matrix, user_idx, idx_movie, titles, top_k=TOP_K):
    if user_id not in user_idx:
        return []
    u = user_idx[user_id]
    pred = predicted[u].copy()

    # exclude already-rated movies
    already_rated = matrix[u].toarray().flatten() > 0
    pred[already_rated] = -999

    top_k_indices = np.argsort(pred)[::-1][:top_k]
    return [
        {
            "movie_id":       int(idx_movie[i]),
            "title":          titles.get(idx_movie[i], "Unknown"),
            "predicted_rating": round(float(pred[i]), 2)
        }
        for i in top_k_indices
    ]

# ── save results ──────────────────────────────────────────────────────────────

def save_results(metrics, con):
    print("Saving evaluation metrics to DuckDB...")
    metrics_df = pd.DataFrame([metrics])
    con.execute("drop table if exists mart_svd_metrics")
    con.execute("create table mart_svd_metrics as select * from metrics_df")
    print("  Saved mart_svd_metrics")

def save_sample_recommendations(predicted, matrix, user_idx, idx_movie, titles, con, n_sample=100):
    print(f"Saving sample recommendations for {n_sample} users...")
    rows = []
    sample_users = list(user_idx.keys())[:n_sample]
    for user_id in sample_users:
        recs = get_recommendations(user_id, predicted, matrix, user_idx, idx_movie, titles)
        for rank, rec in enumerate(recs, 1):
            rows.append({
                "user_id":          user_id,
                "rank":             rank,
                "movie_id":         rec["movie_id"],
                "title":            rec["title"],
                "predicted_rating": rec["predicted_rating"]
            })
    recs_df = pd.DataFrame(rows)
    con.execute("drop table if exists mart_svd_recommendations")
    con.execute("create table mart_svd_recommendations as select * from recs_df")
    print(f"  Saved {len(rows):,} recommendation rows")

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    con = duckdb.connect(DB_PATH)

    df = load_ratings(con, max_users=50000, min_ratings=20)
    matrix, user_idx, movie_idx, idx_user, idx_movie = build_matrix(df)
    predicted, user_means = train_svd(matrix)
    metrics = evaluate(matrix, predicted)

    titles = con.execute(
        "select movie_id, title from dim_movies"
    ).df().set_index("movie_id")["title"].to_dict()

    save_results(metrics, con)
    save_sample_recommendations(predicted, matrix, user_idx, idx_movie, titles, con)

    first_user = list(user_idx.keys())[0]
    recs = get_recommendations(first_user, predicted, matrix, user_idx, idx_movie, titles)
    print(f"\nSample recommendations for user {first_user}:")
    for r in recs:
        print(f"  {r['title']} (predicted: {r['predicted_rating']})")

    with open("python/modeling/svd_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to python/modeling/svd_metrics.json")

    con.close()
    print("\nDone.")

if __name__ == "__main__":
    main()