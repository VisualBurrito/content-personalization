-- intermediate/int_movies_enriched.sql

with movies as (
    select distinct movie_id, title, release_year
    from {{ ref('stg_movielens__movies') }}
),

genres_agg as (
    select
        movie_id,
        array_agg(genre order by genre) as genres,
        count(genre)                    as genre_count
    from {{ ref('stg_movielens__movies') }}
    group by movie_id
),

tmdb as (
    select * from {{ ref('stg_tmdb__metadata') }}
),

ratings_summary as (
    select
        movie_id,
        count(*)                         as total_ratings,
        avg(rating)                      as avg_rating,
        stddev(rating)                   as rating_stddev,
        sum(is_positive_rating::int)     as positive_rating_count
    from {{ ref('stg_movielens__ratings') }}
    group by movie_id
),

joined as (
    select
        m.movie_id,
        m.title,
        m.release_year,
        g.genres,
        g.genre_count,
        rs.total_ratings,
        rs.avg_rating,
        rs.rating_stddev,
        rs.positive_rating_count,
        t.tmdb_rating,
        t.popularity_score,
        t.runtime_minutes,
        t.plot_summary,
        t.original_language
    from movies            m
    left join genres_agg   g  on m.movie_id = g.movie_id
    left join ratings_summary rs on m.movie_id = rs.movie_id
    left join tmdb         t  on m.movie_id = t.movie_id  -- exact join now
)

select *
from joined
where total_ratings >= 20
