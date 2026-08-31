"""
push_to_motherduck.py
Pushes pre-computed recommendation tables from local DuckDB to MotherDuck.
Run this after precompute.py to update the live dashboard.
"""

import duckdb
import os

LOCAL_DB = "C:/Users/lsmit/content_personalization/dbt_code/dev.duckdb"
MD_DB    = "md:my_db"

TABLES = [
    "mart_svd_recommendations_full",
    "mart_user_profiles",
    "mart_genre_correlation",
    "mart_movie_correlation",
    "mart_ab_test_results",
    "dim_movies",
    "fact_ratings",
]

def main():
    print("Connecting to local DuckDB...")
    local = duckdb.connect(LOCAL_DB)

    print("Connecting to MotherDuck...")
    md = duckdb.connect(MD_DB)

    for table in TABLES:
        try:
            df = local.execute(f"select * from {table}").df()
            md.execute(f"drop table if exists {table}")
            md.execute(f"create table {table} as select * from df")
            print(f"  ✓ {table} ({len(df):,} rows)")
        except Exception as e:
            print(f"  ✗ {table} — {e}")

    local.close()
    md.close()
    print("\nDone. MotherDuck is up to date.")

if __name__ == "__main__":
    main()