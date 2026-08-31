# Content Recommendation Engine

A Netflix-style recommendation system built on MovieLens 32M ratings.
Demonstrates end-to-end analytics engineering: data modeling, correlation
analysis, matrix factorization, and A/B tested evaluation.

**[Live Demo →](https://movie--rec.streamlit.app/)**

---

## Key Results

| Metric | Value |
|---|---|
| Ratings ingested | 32,000,204 |
| Movies modeled | 23,123 |
| SVD Precision@10 | 52% |
| SVD NDCG@10 | 0.93 |
| Lift over popularity baseline | +27.6% (p < 0.0001) |

---

## Motivation

Recommendation systems sit at the intersection of statistics, engineering,
and product thinking — you have to model human taste, build pipelines that
scale, and validate that your work actually improves outcomes.

This project builds the full stack from scratch: raw data to modeled warehouse
to trained model to evaluated experiment to live dashboard. MovieLens and TMDB
provide a rich, realistic dataset with the same structural complexity you'd
find in production — sparse ratings, cold start problems, popularity bias,
and the need to distinguish signal from noise.

---

## Architecture

```
MovieLens 32M CSVs          TMDB Metadata (Kaggle)
       │                           │
       └──────────┬────────────────┘
                  │
            DBT + DuckDB
         ┌────────┴────────┐
      Staging          Staging
   (type casting,    (type casting,
    filtering)        TMDB join key)
         └────────┬────────┘
              Intermediate
          (enrichment, windows,
           co-occurrence inputs)
                  │
                Marts
         (correlation tables,
          dim/fact models)
                  │
           ┌──────┴──────┐
        Python          Python
     (sparse matrix    (SVD model,
      cooccurrence)    A/B test)
                  │
            Streamlit
            Dashboard
```

---

## Stack

| Layer | Tools |
|---|---|
| Data modeling | DBT, DuckDB |
| Data sources | MovieLens 32M, TMDB via Kaggle |
| Correlation analysis | Python, scipy sparse matrices |
| Recommendation model | SVD matrix factorization (scipy.sparse.linalg.svds) |
| Evaluation | Precision@K, NDCG@K, two-sided t-test |
| Dashboard | Streamlit, Plotly |
| Production warehouse | MotherDuck |

---

## Project structure

```
├── dbt_code/
│   └── models/
│       ├── staging/          # type casting, CSV ingestion, filtering
│       ├── intermediate/     # joins, aggregations, window functions
│       └── mart/             # final analysis-ready tables
├── python/
│   ├── analysis/
│   │   └── cooccurrence.py   # sparse matrix Pearson correlation
│   ├── modeling/
│   │   ├── svd_model.py      # SVD recommendation engine
│   │   ├── ab_test.py        # A/B test simulation
│   │   ├── precompute.py     # pre-computes recommendations for dashboard
│   │   ├── push_to_motherduck.py  # syncs local DuckDB to MotherDuck
│   │   └── svd_metrics.json  # evaluation results
│   └── dashboard/
│       └── app.py            # Streamlit dashboard
├── data/
│   └── README.md             # download instructions
├── scripts/
│   └── setup_data.sh         # automated data download
├── run_pipeline.py           # orchestrates full pipeline end to end
├── profiles.yml.example
├── .env.example
└── requirements.txt
```

---

## DBT model layers

**Staging** reads raw CSVs via DuckDB's `read_csv()`, casts types, and
filters bad rows. `stg_tmdb__metadata` joins through `links.csv` for an
exact MovieLens→TMDB ID match rather than fuzzy title matching.

**Intermediate** models enrich and aggregate. `int_movies_enriched` joins
MovieLens metadata to TMDB for genres, runtime, popularity, and plot
summaries. `int_user_activity_windows` computes rolling 12-month rating
activity per user, pre-aggregated to monthly buckets to keep window
functions tractable at scale.

**Marts** are the final analysis-ready tables Python and the dashboard
read from: `dim_movies`, `fact_ratings`, `mart_genre_correlation`,
`mart_movie_correlation`, and `mart_ab_test_results`.

---

## Why Python handles cooccurrence (not SQL)

A SQL self-join on movie pairs generates O(n²) intermediate rows before
any aggregation filter runs. At 32M ratings across thousands of popular
movies, this exhausts available RAM on any reasonable development machine.

`python/analysis/cooccurrence.py` uses `scipy.sparse` CSR matrices and
computes adjusted cosine similarity in chunks of 200 movies, keeping peak
memory usage well within 16GB. Results are written back to DuckDB as
`mart_movie_correlation` — same interface as if dbt had built it.

---

## Recommendation model

SVD matrix factorization decomposes the user-movie rating matrix into
latent factors representing hidden taste dimensions. Before decomposition,
user mean ratings are subtracted (adjusted cosine) to correct for
individual rating bias — users who rate everything 4-5 won't appear
falsely similar to each other.

```python
# core decomposition
U, sigma, Vt = svds(matrix_demeaned, k=50)
predicted = U @ np.diag(sigma) @ Vt
predicted += user_means.reshape(-1, 1)
```

Trained on 50,000 most active users. Evaluated on 1,000 held-out users
with 20% of ratings withheld as a test set.

---

## A/B test design

5,000 users randomly split 50/50 into control and treatment:

- **Control**: top-N most popular unseen movies (popularity baseline)
- **Treatment**: SVD personalized recommendations

Test items removed from training matrix before generating recommendations
for either variant. Precision@10 measured against held-out items rated
≥ 4.0 stars. Two-sided independent samples t-test at α = 0.05.

**Result**: SVD outperforms popularity baseline by +27.6% on Precision@10
(p < 0.0001). The improvement is statistically significant and consistent
across user segments.

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/VisualBurrito/content-personalization.git
cd content-personalization
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Download data**
```bash
./scripts/setup_data.sh full   # downloads MovieLens 32M + TMDB
```
Or follow manual instructions in `data/README.md`.

**3. Configure DBT**
```bash
cp profiles.yml.example ~/.dbt/profiles.yml
# edit ~/.dbt/profiles.yml with your local paths
```

**4. Configure environment**
```bash
cp .env.example .env
# add your MotherDuck token if using cloud warehouse
```

**5. Run the full pipeline**
```bash
python run_pipeline.py
```

Or run steps individually:
```bash
cd dbt_code && dbt run && cd ..
python python/analysis/cooccurrence.py
python python/modeling/svd_model.py
python python/modeling/ab_test.py
python python/modeling/precompute.py
python python/modeling/push_to_motherduck.py
streamlit run python/dashboard/app.py
```

---

## Background

Senior data analyst with 7+ years of experience spanning healthcare analytics,
actuarial modeling, and data engineering and degrees in Physics and Music.

This project reflects an interest in applying those foundations to
recommendation systems and content analytics.
