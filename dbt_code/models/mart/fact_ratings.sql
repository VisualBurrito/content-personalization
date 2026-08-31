{{ config(materialized='view') }}

-- fact_ratings.sql
-- view keeps this out of MotherDuck and avoids materializing 32M rows

select
    r.user_id,
    r.movie_id,
    m.title,
    m.genres,
    r.rating,
    r.rated_at,
    r.rating_month,
    r.is_positive_rating,
    w.ratings_last_12mo,
    w.avg_rating_last_12mo,
    w.is_active_user

from {{ ref('stg_movielens__ratings') }}         r
left join {{ ref('dim_movies') }}                m
    on r.movie_id = m.movie_id
left join {{ ref('int_user_activity_windows') }} w
    on  r.user_id  = w.user_id
    and r.movie_id = w.movie_id
    and r.rated_at = w.rated_at
