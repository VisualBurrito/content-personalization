-- staging/stg_movielens__movies.sql

with source as (
    select * from read_csv(
        '{{ var("data_path") }}/movielens/movies.csv',
        header=true,
        all_varchar=true,       -- read everything as text first, cast manually below
        ignore_errors=true      -- skip malformed rows like the September 11 title
    )
),

renamed as (
    select
        try_cast(movieId as bigint)                     as movie_id,
        regexp_replace(title, '\s*\(\d{4}\)\s*$', '')  as title,
        regexp_extract(title, '\((\d{4})\)', 1)         as release_year,
        unnest(string_split(genres, '|'))               as genre
    from source
    where movieId is not null
),

filtered as (
    select *
    from renamed
    where genre    != '(no genres listed)'
      and movie_id is not null
)

select * from filtered
