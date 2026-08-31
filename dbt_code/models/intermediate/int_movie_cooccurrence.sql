{{ config(enabled=false) }}
-- Disabled: movie cooccurrence computed in Python using sparse matrices
-- See python/analysis/cooccurrence.py
select 1 as placeholder

-- int_movie_cooccurrence.sql
-- pre-filter to popular movies only before self-joining
-- reduces pair candidates from millions to thousands

-- with popular_movies as (
--     -- only movies with 100+ ratings from our dev user sample
--     -- this cuts the movie pool dramatically before the self-join
--     select movie_id
--     from {{ ref('stg_movielens__ratings') }}
--     group by movie_id
--     having count(*) >= 100
-- ),

-- filtered_ratings as (
--     -- only keep ratings for popular movies
--     select r.user_id, r.movie_id, r.rating
--     from {{ ref('stg_movielens__ratings') }} r
--     inner join popular_movies p on r.movie_id = p.movie_id
-- ),

-- pairs as (
--     select
--         r1.movie_id                          as movie_a,
--         r2.movie_id                          as movie_b,
--         count(*)                             as shared_raters,
--         sum(r1.rating)                       as sum_a,
--         sum(r2.rating)                       as sum_b,
--         sum(r1.rating * r2.rating)           as sum_product,
--         sum(r1.rating * r1.rating)           as sum_sq_a,
--         sum(r2.rating * r2.rating)           as sum_sq_b
--     from filtered_ratings r1
--     join filtered_ratings r2
--         on  r1.user_id  = r2.user_id
--         and r1.movie_id < r2.movie_id
--     group by 1, 2
--     having count(*) >= 50
-- )

-- select * from pairs
