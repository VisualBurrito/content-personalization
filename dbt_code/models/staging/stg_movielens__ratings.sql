-- staging/stg_movielens__ratings.sql

with source as (
    select * from read_csv(
        '{{ var("data_path") }}/movielens/ratings.csv',
        header=true,
        all_varchar=true
    )
),

renamed as (
    select
        try_cast(userId    as bigint)               as user_id,
        try_cast(movieId   as bigint)               as movie_id,
        try_cast(rating    as numeric(3,1))          as rating,
        to_timestamp(try_cast(timestamp as bigint)) as rated_at,
        try_cast(rating as numeric(3,1)) >= 4.0     as is_positive_rating,
        date_trunc('month', to_timestamp(
            try_cast(timestamp as bigint)))          as rating_month
    from source
),

filtered as (
    select *
    from renamed
    where user_id  is not null
      and movie_id is not null
      and rating between 0.5 and 5.0

    -- DEV LIMIT: remove this block for final prod run
    {% if target.name == 'dev' %}
        and user_id <= {{ var('max_user_id', 10000) }}
    {% endif %}
)

select * from filtered
