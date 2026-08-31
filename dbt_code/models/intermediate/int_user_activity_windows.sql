-- int_user_activity_windows.sql
-- simplified window approach to reduce memory pressure

with ratings as (
    select
        user_id,
        movie_id,
        rating,
        rated_at,
        rating_month
    from {{ ref('stg_movielens__ratings') }}
),

-- pre-aggregate to user-month level first
-- this dramatically reduces rows before the window runs
user_monthly as (
    select
        user_id,
        rating_month,
        count(*)    as monthly_rating_count,
        avg(rating) as monthly_avg_rating
    from ratings
    group by user_id, rating_month
),

-- rolling 12-month window on the aggregated monthly data
-- operates on ~120 rows per user max instead of all individual ratings
user_windows as (
    select
        user_id,
        rating_month,
        sum(monthly_rating_count) over (
            partition by user_id
            order by rating_month
            rows between 11 preceding and current row
        ) as ratings_last_12mo,

        avg(monthly_avg_rating) over (
            partition by user_id
            order by rating_month
            rows between 11 preceding and current row
        ) as avg_rating_last_12mo,

        sum(monthly_rating_count) over (
            partition by user_id
            order by rating_month
            rows between 11 preceding and current row
        ) >= 3 as is_active_user
    from user_monthly
),

-- join back to individual ratings
final as (
    select
        r.user_id,
        r.movie_id,
        r.rating,
        r.rated_at,
        r.rating_month,
        w.ratings_last_12mo,
        w.avg_rating_last_12mo,
        w.is_active_user
    from ratings      r
    left join user_windows w
        on  r.user_id      = w.user_id
        and r.rating_month = w.rating_month
)

select * from final
