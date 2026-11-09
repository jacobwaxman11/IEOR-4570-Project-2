# NYC Rental Agent

A chat agent over a local NYC rental apartment listing database. Ask about neighborhoods, rents, buildings, and model deals. The page streams each tool call, then shows the answer with listing cards you can save.

The agent is Gemini on Vertex AI (`gemini-3.8-flash`), reached through the OpenAI Agents SDK and LiteLLM. Listing data lives in SQLite. SQL the model writes is read-only.

## What you can ask

- Filtered search: neighborhood, beds, baths, square feet, rent.
- The cheapest active listings in a neighborhood.
- Amenities for one building.
- A short neighborhood profile, or a rent comparison across neighborhoods.
- Deals from the rent model: predicted rent, listed rent, percent discount, and a 0–10 deal score.

Anything those tools do not cover goes through the schema tools: `read_mappings`, then `get_enum_values` for a text column, then one `SELECT`.

## Setup

Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
cd IEOR-4570-Project-2
uv sync
cp .env.example .env
```

The chat and the photo scorer both use Vertex AI. In `.env`, set:

- `GOOGLE_CLOUD_PROJECT` — your GCP project id.
- `GOOGLE_APPLICATION_CREDENTIALS` — path to a service-account JSON key.

Or authenticate with Application Default Credentials instead of a key file:

```bash
gcloud auth application-default login
```

`GOOGLE_CLOUD_LOCATION` defaults to `global` and is read by the photo scorer. The chat model is pinned to Vertex location `global`. `GOOGLE_MAPS_API_KEY` is only used by Places and Routes helpers in `tools.py`; those helpers are not registered on the agent.

Listing data also lives in BigQuery, in dataset `IEOR_4570_Project_2` (dataset ids cannot contain hyphens) in location `US`. `BQ_LOCATION` overrides that location. `GOOGLE_CLOUD_LOCATION` is not a BigQuery location.

`APP_ENV` chooses which database the chat uses. `local` (the default) reads listings from SQLite and writes chats and saved listings to SQLite. `prod` reads listings from BigQuery and writes chats and saved listings only there. Cloud Run sets `K_SERVICE`, and an unset `APP_ENV` there uses BigQuery, because `rentals.db` is not in the image. The runtime service account needs `roles/bigquery.jobUser` and `roles/bigquery.dataEditor`. `GOOGLE_CLOUD_PROJECT` can be omitted on Cloud Run; the metadata server supplies the project id.

`scripts/rentals.db` is not in git. A local run exits when a tool first opens the database if that file is missing. Build it with the pipeline below, or copy in an existing database. Cloud Run starts from the `Procfile`, which binds uvicorn to `0.0.0.0:$PORT`.

## Run the chat

From the project root:

```bash
uv run app.py
```

Open [http://localhost:8001](http://localhost:8001). The port is `PORT`, default `8001`.

The sidebar keeps past chats (`conversations.db`) and saved listings (`ui.db`). Both files are local and gitignored. With `APP_ENV=prod` those rows are in BigQuery instead, in `agent_sessions`, `agent_messages`, and `favorites`. Tool steps stream in as they run. When a turn returns listings, the page shows a card grid, one card, or a pair, depending on the question.

## Build the database

Run these from `scripts/`. `uv` still uses the project environment, and each script writes `rentals.db` in the current directory. The same writes are mirrored to BigQuery. Set `BQ_MIRROR=0` to keep a run local. Re-runs skip work that is already stored.

Load the current SQLite files into BigQuery once, before the first prod deploy. This creates the dataset and replaces every table, including chat history and saved listings:

```bash
uv run scripts/migrate_to_bigquery.py
```

Build or refresh the local database from `scripts/`. Each of these writes SQLite and BigQuery:

```bash
cd scripts

# 1. Search pages: rent, status, photos. Defaults to Manhattan, $3,000–$7,000, 250 new listings.
uv run scrape_listings.py

# 2. Detail pages: amenities, description, availability, floor plan.
uv run backfill_details.py

# 3. Gemini scores for listing photos, then per-listing aggregates.
uv run score_listings.py

# 4. Out-of-fold ridge on log rent. Writes regression_features and regression_results.
uv run features.py
```

`update_listings.py` refreshes rent and status for rows whose `last_updated_date` is older than a week. `first_scraped_at` is left alone. A few StreetEasy area ids: 113 Chelsea, 115 Gramercy, 140 Upper East Side, 146 Hudson Yards, 158 Flatiron.

Photo scoring needs the same Vertex credentials as the chat. It sends local images to `gemini-3.8-flash`, caches one score per image URL, and rolls those up into `listing_scores`.

`features.py` fits a ridge on standardized predictors of `log_rent`. Alpha is chosen by out-of-fold RMSE. A listing is never scored by a model that trained on it. `percent_discount` is `100 * (predicted - actual) / predicted`, so a positive number means the unit is listed below the model. Deal score is that discount mapped onto 0–10: 5 is fairly priced, 10 is at least 25% under the model, and 0 is at least 25% over. Pass `--output features.csv` to also dump the frame, `--include-unscored` to keep listings that have no photo scores, or `--raw` to skip the amenity-tier and image-composite transforms.

## Layout


| Path                             | Role                                                       |
| -------------------------------- | ---------------------------------------------------------- |
| `app.py`                         | FastAPI chat server, agent, and SSE stream of tool calls   |
| `tools.py`                       | Read-only SQL tools and the analysis tools the agent calls |
| `present.py`                     | Listing cards, chat history, and saved listings            |
| `index.html`, `static/`          | Chat page                                                  |
| `scripts/db.py`                  | SQLite schema and write helpers, mirrored to BigQuery      |
| `scripts/bq.py`                  | BigQuery dataset, loads, and merges                        |
| `scripts/migrate_to_bigquery.py` | One-time load of the local SQLite files                    |
| `session_store.py`               | Chat history: SQLite locally, BigQuery in prod             |
| `scripts/scrape_listings.py`     | StreetEasy search scrape and photo download                |
| `scripts/backfill_details.py`    | Detail-page backfill                                       |
| `scripts/score_listings.py`      | Gemini photo scoring                                       |
| `scripts/features.py`            | Feature frame and deal-score regression                    |
| `scripts/update_listings.py`     | Refresh stale rent and status                              |
| `scripts/rentals.db`             | Listings, amenities, images, scores, deal results          |
| `scripts/images/`                | Downloaded photos, used only by the local scorer           |




## Database

`listing_id` (building slug plus unit) joins the listing tables. `image_url` joins `listing_images` to `image_scores`; a shared building photo is scored once.


| Table                 | Contents                                                                                                                                                                  |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `listings`            | One row per StreetEasy listing                                                                                                                                            |
| `listing_amenities`   | Normalized amenity tags                                                                                                                                                   |
| `listing_images`      | Photo URLs and local paths, in listing order                                                                                                                              |
| `image_scores`        | Gemini scores for one image                                                                                                                                               |
| `listing_scores`      | Photo aggregates for one listing                                                                                                                                          |
| `regression_features` | Numeric frame the rent model trained on. In BigQuery, a column name keeps only letters, numbers, and underscores, so `nh_Hell's Kitchen` is stored as `nh_Hells_Kitchen`. |
| `regression_results`  | Predicted rent, percent discount, deal score                                                                                                                              |


`run_sql` allows a single `SELECT` or `WITH`. Writes, `PRAGMA`, and multiple statements are rejected before they run. Locally the connection is opened read-only. In prod the same guard runs against BigQuery Standard SQL. Results are capped at 100 rows.

Prod tables live in `IEOR_4570_Project_2`: the seven rental tables above, plus `favorites`, `agent_sessions`, and `agent_messages`. The agent catalog is still only the rental tables. Pipeline scripts write both SQLite and BigQuery. The page writes chats and saved listings only to the database for `APP_ENV`.