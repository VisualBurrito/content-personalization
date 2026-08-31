"""
precompute.py
Pre-computes all recommendations and saves to DuckDB.
Run this locally before pushing to MotherDuck for deployment.
The dashboard reads pre-built tables — no model training at runtime.
"""

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

DB_PATH   = "C:/Users/lsmit/content_personalization/dbt_code/dev.duckdb"
N_FACTORS = 50
TOP_K     = 10
N_USERS   = 50000

def load_ratings(con):
    print("Loading ratings...")
    df = con.execute(f"""
        with active_users as (
            select user_id
            from stg_movielens__ratings
            group by user_id
            having count(*) >= 20
            order by count(*) desc
            limit {N_USERS}
        )
        select r.user_id, r.movie_id, r.rating
        from stg_movielens__ratings r
        inner join active_users a on r.user_id = a.user_id
    """).df()
    movie_counts = df.groupby("movie_id")["rating"].count()
    popular = movie_counts[movie_counts >= 10].index
    df = df[df["movie_id"].isin(popular)]
    print(f"  {len(df):,} ratings, {df.user_id.nunique():,} users, {df.movie_id.nunique():,} movies")
    return df

def build_and_train(df):
    print("Building matrix and training SVD...")
    user_idx  = {u: i for i, u in enumerate(df.user_id.unique())}
    movie_idx = {m: i for i, m in enumerate(df.movie_id.unique())}
    idx_movie = {i: m for m, i in movie_idx.items()}
    idx_user  = {i: u for u, i in user_idx.items()}

    matrix = csr_matrix(
        (df.rating.values,
         (df.user_id.map(user_idx), df.movie_id.map(movie_idx))),
        shape=(len(user_idx), len(movie_idx)), dtype=float
    )

    user_means = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        if row.nnz > 0:
            user_means[i] = row.data.mean()

    matrix_d = matrix.copy().astype(float)
    rows, cols = matrix_d.nonzero()
    matrix_d.data -= user_means[rows]

    U, sigma, Vt = svds(matrix_d, k=N_FACTORS)
    predicted = U @ np.diag(sigma) @ Vt
    predicted += user_means.reshape(-1, 1)
    predicted = np.clip(predicted, 0.5, 5.0)

    print(f"  Done. Predicted range: {predicted.min():.2f} - {predicted.max():.2f}")
    return predicted, matrix, user_idx, movie_idx, idx_movie, idx_user

def save_recommendations(predicted, matrix, user_idx, idx_movie, titles, con):
    print("Saving recommendations for all users...")
    rows = []
    for u_idx in range(predicted.shape[0]):
        pred = predicted[u_idx].copy()
        rated = matrix.getrow(u_idx).indices
        pred[rated] = -999
        top_k = np.argsort(pred)[::-1][:TOP_K]
        user_id = list(user_idx.keys())[u_idx]
        for rank, i in enumerate(top_k, 1):
            mid = idx_movie[i]
            rows.append({
                "user_id":          int(user_id),
                "rank":             rank,
                "movie_id":         int(mid),
                "title":            titles.get(mid, f"Movie {mid}"),
                "predicted_rating": round(float(predicted[u_idx, i]), 2)
            })
        if u_idx % 5000 == 0:
            print(f"  {u_idx:,}/{predicted.shape[0]:,} users processed...")

    recs_df = pd.DataFrame(rows)
    con.execute("drop table if exists mart_svd_recommendations_full")
    con.execute("""
        create table mart_svd_recommendations_full as
        select * from recs_df
    """)
    print(f"  Saved {len(rows):,} rows to mart_svd_recommendations_full")
    return recs_df

def save_user_profiles(matrix, user_idx, idx_movie, titles, con):
    print("Saving user taste profiles (top rated movies per user)...")
    rows = []
    for user_id, u_idx in user_idx.items():
        row = matrix.getrow(u_idx)
        if row.nnz == 0:
            continue
        rated = list(zip(row.indices, row.data))
        rated.sort(key=lambda x: -x[1])
        for mid_idx, rating in rated[:10]:
            mid = idx_movie[mid_idx]
            rows.append({
                "user_id":  int(user_id),
                "movie_id": int(mid),
                "title":    titles.get(mid, f"Movie {mid}"),
                "rating":   float(rating)
            })

    profiles_df = pd.DataFrame(rows)
    con.execute("drop table if exists mart_user_profiles")
    con.execute("create table mart_user_profiles as select * from profiles_df")
    print(f"  Saved {len(rows):,} rows to mart_user_profiles")

def main():
    con = duckdb.connect(DB_PATH)

    titles = con.execute(
        "select movie_id, title from dim_movies"
    ).df().set_index("movie_id")["title"].to_dict()

    df = load_ratings(con)
    predicted, matrix, user_idx, movie_idx, idx_movie, idx_user = build_and_train(df)
    save_recommendations(predicted, matrix, user_idx, idx_movie, titles, con)
    save_user_profiles(matrix, user_idx, idx_movie, titles, con)

    # verify
    count = con.execute(
        "select count(*) from mart_svd_recommendations_full"
    ).fetchone()[0]
    print(f"\nTotal recommendations saved: {count:,}")
    print("Done — ready to push to MotherDuck.")
    con.close()

if __name__ == "__main__":
    main()