# NYC Rental Agent

A chat agent for someone comparing Manhattan rentals. The target user is a renter, or a friend helping one, who wants a unit in Manhattan. They ask which listings match a neighborhood, bedroom count, and budget, what a building includes, how two neighborhoods compare, and which units a rent model marks as underpriced.

The page streams each tool call, then shows the answer with listing cards the user can save. The model is Gemini on Vertex AI (`gemini-3.8-flash`), reached through the OpenAI Agents SDK and LiteLLM. Listing data lives in SQLite. SQL the model writes is read-only.

## Data

Listings come from [StreetEasy](https://streeteasy.com) for-rent search pages. `scripts/scrape_listings.py` reads the schema.org Apartment objects on those pages, keeps rent, status, and photos, and writes `scripts/rentals.db`. `scripts/backfill_details.py` then opens each listing's detail page for amenities, the description, availability, and the floor plan. `scripts/score_listings.py` sends the downloaded photos to the same Gemini model and stores one score per image. `scripts/features.py` fits an out-of-fold ridge on `log_rent` and writes predicted rent, percent discount, and a 0–10 deal score. A positive percent discount means the unit is listed below the model. Deal score maps that discount onto 0–10: 5 is the model's price, 10 is at least 25% under it, and 0 is at least 25% over it.

The file used for the answers below was pulled on November 8, 2026. It has 318 active listings in 36 Manhattan neighborhoods, with rents from $3,100 to $7,764. Photo scores and the ridge outputs are in the same file: 236 listings have a deal score. `scripts/rentals.db` is not in git. A local chat exits when a tool first opens the database if that file is missing.

To build a database from scratch, from `scripts/`:

```bash
cd scripts

# Search pages. The example below is Manhattan, $3,000–$7,000, 250 new listings.
uv run scrape_listings.py

# Detail pages: amenities, description, availability, floor plan.
uv run backfill_details.py

# Gemini scores for listing photos, then per-listing aggregates.
uv run score_listings.py

# Out-of-fold ridge on log rent. Writes regression_features and regression_results.
uv run features.py
```

`uv` uses the project environment, and each script writes `rentals.db` in the current directory. The same writes can be mirrored to BigQuery. Set `BQ_MIRROR=0` to keep a run local. Re-runs skip work that is already stored. Area ids for the search URL are documented at the top of `scrape_listings.py` (113 Chelsea, 115 Gramercy, 140 Upper East Side, 146 Hudson Yards, 158 Flatiron, and others). A grader can answer the questions below from the description here without rebuilding the file. Re-running the scrape changes rents and which units are present.

`uv run scripts/migrate_to_bigquery.py` loads the current SQLite files into BigQuery dataset `IEOR_4570_Project_2` in location `US`.

## Sample queries

These three were run against `scripts/rentals.db` on November 8, 2026. "This agent" is the app in `app.py`: the system prompt, `read_mappings`, `get_enum_values`, `run_sql`, and the analysis tools. "No tools" is the same model, `gemini-3.8-flash`, with one user message, no system prompt, and no tools. Paste a question into the chat at [http://localhost:8001](http://localhost:8001).

### 1. Cheapest 1-bedroom in Chelsea

Ask: "What is the cheapest active 1-bedroom in Chelsea? Give the address, square feet, and monthly rent."

Expected answer: **100 West 26th Street #6H**, 635 sq ft, **$5,595**.

Source: `listings`. Filter `neighborhood = 'Chelsea'`, `beds = 1`, and `listing_status = 'ACTIVE'`, then take the lowest `rent`. The next listing in that neighborhood is 777 6th Avenue #4C at $5,863. West Chelsea is a separate value in `neighborhood`. Its cheapest 1-bedroom is 311 11th Avenue #1119 at $5,495, with no square footage stored.

This agent: 100 West 26th Street #6H (listing id `chelsea-tower_6h`), 635 sq ft, $5,595. It also noted the cheaper West Chelsea unit and kept it out of the Chelsea answer. Tools: `read_mappings`, then `run_sql`.

Same model, no tools and no system prompt: it said it cannot see live StreetEasy inventory, told the user to sort StreetEasy by price, and guessed that the cheapest market-rate Chelsea 1-bedrooms run about $2,800–$3,400 and 400–550 sq ft.

### 2. Best model deal in Hell's Kitchen

Ask: "What is the single best model deal in Hell's Kitchen? Give the address, number of bedrooms, listed rent, predicted rent, percent discount, and deal score."

Expected answer: **605 West 42nd Street #28K**, a studio (`beds` is 0), listed **$3,895**, predicted **$4,843**, percent discount **19.57**, deal score **8.91**.

Source: `listings` joined to `regression_results` on `listing_id`, filtered to `neighborhood = 'Hell''s Kitchen'`, ordered by `deal_score` and then `percent_discount`.

This agent: the same unit, studio, $3,895 listed, $4,843 predicted, 19.57% under the model, deal score 8.91. Tool: `best_deals_in_neighborhood` with Hell's Kitchen and `limit` 1.

Same model, no tools and no system prompt: it said predicted rent and deal score belong to a specific model it does not have, and asked which dataset to use. It named no listing.

### 3. Most common amenities in Chelsea

Ask: "What are the five most common amenities among Chelsea listings, and how many Chelsea listings have each?"

Expected answer: Chelsea has 24 listings. Three amenities are tied at 23 listings: `elevator`, `fios_available`, and `live_in_super`. The next three are tied at 22: `concierge`, `doorman`, and `gym`. There is no unique set of five.

Source: `listing_amenities` joined to `listings` on `listing_id`, filtered to `neighborhood = 'Chelsea'`, counted with `COUNT(DISTINCT listing_id)` and grouped by `amenity`.

This agent: elevator, Verizon Fios (`fios_available`), and live-in super at 23, then concierge, doorman, and gym at 22, and it said the top of the list is a tie. Tools: `summarize_neighborhood`, then `read_mappings` and `run_sql`.

Same model, no tools and no system prompt: it treated the question as an Inside Airbnb New York extract and named Wifi, heating, air conditioning, kitchen, and essentials. None of those strings are amenity values in this database. It then gave Pandas and SQL for a `listings.csv` it does not have.

## Where plain `run_sql` gets a different answer

The plain agent is the same model with one tool, `run_sql`, and a short prompt that only names the tables and columns. The app agent adds `read_mappings`, `get_enum_values`, the analysis tools, the example queries, and the area aliases. Both runs below used those two agents on November 8, 2026, against the same database.

StreetEasy stores Lincoln Square, Manhattan Valley, and Morningside Heights as their own neighborhoods. `get_deals` treats "Upper West Side" the way a renter does and includes those three along with the rows labeled Upper West Side. A query that matches the string `Upper West Side`, or `LIKE '%West%'`, never sees Lincoln Square or Manhattan Valley, because those names do not contain "West".

### Best 1-bedroom deal on the Upper West Side, at or under $5,000

Ask: "What is the best 1-bedroom deal on the Upper West Side with rent at or below $5000? Give the address, the neighborhood stored on the listing, the listed rent, the predicted rent, and the deal score."

Expected answer: **75 West End Avenue #S9F**, neighborhood **Lincoln Square**, listed **$4,850**, predicted **$5,547**, deal score **7.51** (percent discount 12.57).

Source: `listings` joined to `regression_results`. Among 1-bedrooms at or under $5,000 in Upper West Side, Lincoln Square, Manhattan Valley, and Morningside Heights, this row has the highest `deal_score`. The best row whose `neighborhood` is exactly `Upper West Side` is 150 West 73rd Street #7, listed $4,500, predicted $4,754, deal score 6.07.

Plain `run_sql` answered 150 West 73rd Street #7, Upper West Side, $4,500, predicted $4,754, deal score 6.07. Trace:

1. `run_sql` — `WHERE beds = 1 AND rent <= 5000 AND neighborhood LIKE '%Upper West Side%'` ordered by `deal_score`. One row: 150 West 73rd Street #7, predicted rent 4754, deal score 6.07.
2. `run_sql` — `SELECT DISTINCT neighborhood WHERE neighborhood LIKE '%West%' OR neighborhood LIKE '%Upper%'`. Rows: Upper West Side, West Chelsea, West Village. Lincoln Square and Manhattan Valley are not in that result.
3. `run_sql` — every 1-bedroom with `neighborhood = 'Upper West Side'`. The highest deal score in that set is still 6.07, on 150 West 73rd Street #7.
4. `run_sql` — the full `listings` row for `150-west-73-street-new_york_7`.
5. `run_sql` — the `regression_results` row for that id: predicted rent 4754, percent discount 5.34, deal score 6.07.
6. `run_sql` — `LOWER(neighborhood) LIKE '%west side%'` with the same bed and rent limits. The same single row comes back.

This agent answered 75 West End Avenue #S9F, Lincoln Square, $4,850, predicted $5,547, deal score 7.51. It also reported the literal Upper West Side row as a separate case. Trace:

1. `get_deals` — `neighborhood` "Upper West Side", `min_beds` 1, `max_beds` 1, `max_rent` 5000. The tool resolved the area to Upper West Side, Lincoln Square, Manhattan Valley, and Morningside Heights, ranked by deal score, and returned 3 listings. The first is 75 West End Avenue #S9F, Lincoln Square, actual rent 4850, predicted rent 5547, percent discount 12.57, deal score 7.51. The third is 150 West 73rd Street #7.



### Best 2-bedroom deal on the Upper West Side

Ask: "What is the best 2-bedroom deal on the Upper West Side? Give the address, the neighborhood stored on the listing, the listed rent, the predicted rent, and the deal score."

Expected answer: **63 West 104th Street #101**, neighborhood **Manhattan Valley**, listed **$5,000**, predicted **$5,283**, deal score **6.07** (percent discount 5.36).

Source: the same join and the same four neighborhoods, with `beds = 2` and no rent cap. The only 2-bedroom whose `neighborhood` is exactly `Upper West Side` and that has a deal score is 101 W. 90th Street #17E, listed $6,950, predicted $7,159, deal score 5.58.

Plain `run_sql` answered 101 W. 90th Street #17E, Upper West Side, $6,950, predicted $7,159, deal score 5.58. Trace:

1. `run_sql` — `DISTINCT neighborhood` with `LIKE '%West Side%' OR LIKE '%UWS%'`. One row: Upper West Side.
2. `run_sql` — `neighborhood = 'Upper West Side' AND beds = 2`, ordered by `deal_score`. One row: 101 W. 90th Street #17E, predicted rent 7159, percent discount 2.92, deal score 5.58.
3. `run_sql` — a sample of Upper West Side listings and their deal scores. No higher 2-bedroom score appears.
4. `run_sql` — `neighborhood LIKE '%West%' AND beds = 2`. The same 101 W. 90th Street row is the only result.
5. `run_sql` — `COUNT(*)` of 2-bedrooms in the whole file: 35.
6. `run_sql` — every 2-bedroom, with neighborhood, rent, predicted rent, and deal score. This result includes **63 West 104th Street #101, Manhattan Valley, rent 5000, predicted rent 5283, deal score 6.07**, which is higher than 5.58. The plain agent left that row out of the answer because the neighborhood string is not Upper West Side.
7. `run_sql` — the joined `listings` and `regression_results` row for `james-marquis_17e`, confirming 101 W. 90th Street #17E.

This agent answered 63 West 104th Street #101, Manhattan Valley, $5,000, predicted $5,283, deal score 6.07, and mentioned 101 W. 90th Street #17E as the best row stored under the literal neighborhood. Trace:

1. `get_deals` — `neighborhood` "Upper West Side", `min_beds` 2, `max_beds` 2. The tool searched the same four neighborhoods and returned 4 listings. The first is 63 West 104th Street #101, Manhattan Valley, actual rent 5000, predicted rent 5283, percent discount 5.36, deal score 6.07. The second is 101 W. 90th Street #17E, Upper West Side, deal score 5.58.



## What you can ask

- Filtered search: neighborhood, beds, baths, square feet, rent.
- The cheapest active listings in a neighborhood.
- Amenities for one building.
- A short neighborhood profile, or a rent comparison across neighborhoods.
- Deals from the rent model: predicted rent, listed rent, percent discount, and a 0–10 deal score.

`get_deals` maps a few names renters use onto the stored neighborhoods. "UES" and "Upper East Side" cover Carnegie Hill, Lenox Hill, Yorkville, and Sutton Place. "UWS" and "Upper West Side" cover Upper West Side, Lincoln Square, Manhattan Valley, and Morningside Heights. There is no `listings.neighborhood` value equal to "Upper East Side".

Anything those tools do not cover goes through the schema tools: `read_mappings`, then `get_enum_values` for a text column whose values were not inlined, then one `SELECT`.

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

## Run the chat

From the project root:

```bash
uv run app.py
```

Open [http://localhost:8001](http://localhost:8001). The port is `PORT`, default `8001`.

The sidebar keeps past chats (`conversations.db`) and saved listings (`ui.db`). Both files are local and gitignored. With `APP_ENV=prod` those rows are in BigQuery instead, in `agent_sessions`, `agent_messages`, and `favorites`. Tool steps stream in as they run. When a turn returns listings, the page shows a card grid, one card, or a pair, depending on the question.

## Build the database

The load commands are in [Data](#data). A few more notes:

`update_listings.py` refreshes rent and status for rows whose `last_updated_date` is older than a week. `first_scraped_at` is left alone.

Photo scoring needs the same Vertex credentials as the chat. It sends local images to `gemini-3.8-flash`, caches one score per image URL, and rolls those up into `listing_scores`.

`features.py` fits a ridge on standardized predictors of `log_rent`. Alpha is chosen by out-of-fold RMSE. A listing is never scored by a model that trained on it. Pass `--output features.csv` to also dump the frame, `--include-unscored` to keep listings that have no photo scores, or `--raw` to skip the amenity-tier and image-composite transforms.

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