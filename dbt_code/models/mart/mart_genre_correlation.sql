-- mart_genre_correlation.sql
-- pre-aggregate to user-genre level before self-joining
-- avoids exploding row count of joining raw ratings through genres

with user_genre_ratings as (
    -- one row per user-genre combination
    -- much smaller than joining raw ratings through genres
    select
        r.user_id,
        m.genre,
        avg(r.rating) as avg_rating,
        count(*)      as rating_count
    from {{ ref('stg_movielens__ratings') }} r
    join {{ ref('stg_movielens__movies') }}  m
        on r.movie_id = m.movie_id
    group by r.user_id, m.genre
    having count(*) >= 3
),

pairs as (
    select
        a.genre                          as genre_a,
        b.genre                          as genre_b,
        count(*)                         as shared_raters,
        sum(a.avg_rating)                as sum_a,
        sum(b.avg_rating)                as sum_b,
        sum(a.avg_rating * b.avg_rating) as sum_product,
        sum(a.avg_rating * a.avg_rating) as sum_sq_a,
        sum(b.avg_rating * b.avg_rating) as sum_sq_b
    from user_genre_ratings a
    join user_genre_ratings b
        on  a.user_id = b.user_id
        and a.genre   < b.genre
    group by 1, 2
    having count(*) >= 50
),

pearson as (
    select
        genre_a,
        genre_b,
        shared_raters,
        round(
            (shared_raters * sum_product - sum_a * sum_b)
            / nullif(
                sqrt(shared_raters * sum_sq_a - sum_a * sum_a)
                * sqrt(shared_raters * sum_sq_b - sum_b * sum_b),
            0),
        4) as pearson_r
    from pairs
)

select
    genre_a,
    genre_b,
    shared_raters,
    pearson_r,
    case
        when abs(pearson_r) >= 0.5  then 'strong'
        when abs(pearson_r) >= 0.25 then 'moderate'
        else 'weak'
    end as correlation_strength,
    case
        when pearson_r >  0.1 then 'positive'
        when pearson_r < -0.1 then 'negative'
        else 'neutral'
    end as correlation_direction,
    pearson_r >= 0.5 as recommend_pair
from pearson
order by abs(pearson_r) desc
