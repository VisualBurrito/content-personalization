"""
cooccurrence.py
Computes movie-to-movie Pearson correlation using sparse matrices.
Replaces int_movie_cooccurrence and mart_movie_correlation dbt models.
Reads from local DuckDB, writes results back as a table.
"""

import duckdb
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

DB_PATH = "C:/Users/lsmit/content_personalization/dbt_code/dev.duckdb"
MIN_RATINGS  = 50   # minimum ratings for a movie to be included
MIN_SHARED   = 10    # minimum shared raters for a pair to be kept
TOP_N_PAIRS  = 100000 # only keep top N pairs by absolute correlation

def load_ratings(con):
    print("Loading ratings from DuckDB...")
    df = con.execute("""
        select user_id, movie_id, rating
        from stg_movielens__ratings
    """).df()
    print(f"  {len(df):,} ratings loaded")
    return df

def filter_popular_movies(df, min_ratings=MIN_RATINGS):
    counts = df.groupby("movie_id")["rating"].count()
    popular = counts[counts >= min_ratings].index
    df = df[df["movie_id"].isin(popular)].copy()
    print(f"  {len(popular):,} movies with {min_ratings}+ ratings")
    print(f"  {len(df):,} ratings remaining after filter")
    return df, popular

def build_user_movie_matrix(df):
    print("Building sparse user-movie matrix...")
    # remap ids to contiguous indices
    user_idx  = {u: i for i, u in enumerate(df["user_id"].unique())}
    movie_idx = {m: i for i, m in enumerate(df["movie_id"].unique())}
    idx_movie = {i: m for m, i in movie_idx.items()}  # reverse lookup

    rows = df["user_id"].map(user_idx)
    cols = df["movie_id"].map(movie_idx)
    vals = df["rating"].values

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_idx), len(movie_idx))
    )
    print(f"  Matrix shape: {matrix.shape} ({matrix.nnz:,} non-zero entries)")
    return matrix, movie_idx, idx_movie

def compute_pearson(matrix, movie_idx, idx_movie, min_shared=MIN_SHARED):
    print("Computing adjusted cosine similarity (Pearson)...")
    # subtract user mean rating to adjust for rating bias
    # this is the 'adjusted' part that makes it Pearson not plain cosine
    user_means = np.array(matrix.mean(axis=1))
    # only subtract mean where rating exists (non-zero)
    matrix_csr = matrix.tocsr().astype(float)
    rows, cols = matrix_csr.nonzero()
    matrix_csr.data -= user_means[rows, 0]

    # normalize each movie vector to unit length
    movie_matrix = matrix_csr.T  # now shape: (movies, users)
    movie_matrix_norm = normalize(movie_matrix, norm="l2")

    # dot product of normalized vectors = cosine similarity
    # compute in chunks to avoid memory spike
    n_movies = movie_matrix_norm.shape[0]
    chunk_size = 200
    results = []

    print(f"  Processing {n_movies} movies in chunks of {chunk_size}...")
    for start in range(0, n_movies, chunk_size):
        end = min(start + chunk_size, n_movies)
        chunk = movie_matrix_norm[start:end]
        # similarity between chunk and all movies
        sim = (chunk @ movie_matrix_norm.T).toarray()

        for i, row in enumerate(sim):
            movie_a_idx = start + i
            for j in range(movie_a_idx + 1, n_movies):
                r = row[j]
                if abs(r) > 0.01:  # skip near-zero pairs early
                    results.append((
                        idx_movie[movie_a_idx],
                        idx_movie[j],
                        round(float(r), 4)
                    ))

        if start % 1000 == 0:
            print(f"    processed {start}/{n_movies} movies...")

    print(f"  {len(results):,} pairs computed")
    return results

def build_results_df(results, con, min_shared, top_n):
    print("Building results dataframe...")
    df = pd.DataFrame(results, columns=["movie_a", "movie_b", "pearson_r"])
    df = df.reindex(df["pearson_r"].abs().sort_values(ascending=False).index)
    df = df.head(top_n).reset_index(drop=True)

    # join movie titles from dim_movies
    titles = con.execute("""
        select movie_id, title from dim_movies
    """).df().set_index("movie_id")["title"].to_dict()

    df["title_a"] = df["movie_a"].map(titles)
    df["title_b"] = df["movie_b"].map(titles)
    df["correlation_strength"] = pd.cut(
        df["pearson_r"].abs(),
        bins=[0, 0.25, 0.5, 1.0],
        labels=["weak", "moderate", "strong"]
    )
    df["recommend_pair"] = df["pearson_r"] >= 0.5
    return df

def save_to_duckdb(df, con):
    print("Saving mart_movie_correlation to DuckDB...")
    con.execute("drop table if exists mart_movie_correlation")
    con.execute("""
        create table mart_movie_correlation as
        select * from df
    """)
    count = con.execute(
        "select count(*) from mart_movie_correlation"
    ).fetchone()[0]
    print(f"  Saved {count:,} rows to mart_movie_correlation")

def main():
    con = duckdb.connect(DB_PATH)

    df = load_ratings(con)
    df, popular = filter_popular_movies(df)
    matrix, movie_idx, idx_movie = build_user_movie_matrix(df)
    results = compute_pearson(matrix, movie_idx, idx_movie)
    results_df = build_results_df(results, con, MIN_SHARED, TOP_N_PAIRS)
    save_to_duckdb(results_df, con)

    print("\nTop 10 most correlated movie pairs:")
    print(results_df[["title_a","title_b","pearson_r"]].head(10).to_string())

    con.close()
    print("\nDone.")

if __name__ == "__main__":
    main()