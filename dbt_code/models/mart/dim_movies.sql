-- dbt/models/marts/dim_movies.sql
-- one row per movie, all attributes in one place

select
    movie_id,
    title,
    release_year,
    genres,
    genre_count,
    total_ratings,
    round(avg_rating, 2)              as avg_rating,
    round(rating_stddev, 2)           as rating_stddev,
    positive_rating_count,
    round(
        positive_rating_count::numeric
        / nullif(total_ratings, 0), 4
    )                                 as positive_rating_rate,
    tmdb_rating,
    popularity_score,
    runtime_minutes,
    plot_summary

from {{ ref('int_movies_enriched') }}