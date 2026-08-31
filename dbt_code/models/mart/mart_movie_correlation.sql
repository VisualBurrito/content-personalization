{{ config(enabled=false) }}
-- disabled: depends on int_movie_cooccurrence which is computed in Python
-- see python/analysis/cooccurrence.py
select 1 as placeholder

-- mart_movie_correlation.sql
-- joins cooccurrence pairs to movie titles for display
-- self-join already done in int_movie_cooccurrence

-- select
--     co.movie_a,
--     co.movie_b,
--     ma.title        as title_a,
--     mb.title        as title_b,
--     co.shared_raters,
--     round(
--         (co.shared_raters * co.sum_product - co.sum_a * co.sum_b)
--         / nullif(
--             sqrt(co.shared_raters * co.sum_sq_a - co.sum_a * co.sum_a)
--             * sqrt(co.shared_raters * co.sum_sq_b - co.sum_b * co.sum_b),
--         0),
--     4)              as pearson_r
-- from {{ ref('int_movie_cooccurrence') }}  co
-- join {{ ref('dim_movies') }}              ma on co.movie_a = ma.movie_id
-- join {{ ref('dim_movies') }}              mb on co.movie_b = mb.movie_id
-- order by abs(pearson_r) desc
-- limit 10000
