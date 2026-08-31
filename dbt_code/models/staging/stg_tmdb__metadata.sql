-- staging/stg_tmdb__metadata.sql

with links as (
    select * from read_csv(
        '{{ var("data_path") }}/tmdb/links.csv',
        header=true,
        all_varchar=true
    )
),

metadata as (
    select * from read_csv(
        '{{ var("data_path") }}/tmdb/movies_metadata.csv',
        header=true,
        all_varchar=true,
        ignore_errors=true
    )
),

cleaned as (
    select
        try_cast(id           as integer)       as tmdb_id,
        title                                   as tmdb_title,
        try_cast(vote_average as numeric(4,2))  as tmdb_rating,
        try_cast(vote_count   as integer)       as tmdb_vote_count,
        try_cast(popularity   as numeric(10,4)) as popularity_score,
        try_cast(runtime      as integer)       as runtime_minutes,
        try_cast(budget       as bigint)        as budget_usd,
        try_cast(revenue      as bigint)        as revenue_usd,
        overview                                as plot_summary,
        release_date,
        original_language,
        overview is not null                    as has_plot_summary
    from metadata
    where id is not null
      and id != 'id'
),

joined as (
    select
        try_cast(l.movieId as bigint)   as movie_id,  -- bigint to match movielens
        try_cast(l.tmdbId  as integer)  as tmdb_id,
        c.tmdb_title,
        c.tmdb_rating,
        c.tmdb_vote_count,
        c.popularity_score,
        c.runtime_minutes,
        c.budget_usd,
        c.revenue_usd,
        c.plot_summary,
        c.release_date,
        c.original_language,
        c.has_plot_summary
    from links        l
    left join cleaned c on try_cast(l.tmdbId as integer) = c.tmdb_id
)

select *
from joined
where tmdb_vote_count >= 10
  and movie_id is not null
