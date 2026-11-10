"""Build the training frame from rentals.db.

Joins ``listings`` + ``listing_scores`` + ``listing_amenities`` into one row
per listing and applies the engineering decisions documented inline. The
output is a fully-numeric DataFrame ready to split into X / y for any
sklearn estimator. No scaling is applied — leave that for a Pipeline so it
can be fit inside CV.

``prepare_features()`` returns a *base* frame with individual amenity
one-hot columns and all per-image Gemini score columns intact. Two
optional transforms can be applied afterwards in a notebook to evaluate
their effect:

    collapse_amenity_tiers(df)   — replaces amen_* one-hots with 3 tier counts
    collapse_image_scores(df)    — replaces correlated image columns with
                                   apt_quality / common_quality composites

Usage from a notebook:

    from features import prepare_features, collapse_amenity_tiers, collapse_image_scores
    df = prepare_features()
    y = df["log_rent"]
    X = df.drop(columns=["listing_id", "log_rent"])
    # ... run models on the raw feature set, then:
    df2 = collapse_amenity_tiers(collapse_image_scores(df))

Usage from the CLI (applies all transforms, fits an out-of-fold ridge on
``log_rent``, and writes both tables into the database):

    uv run features.py
    uv run features.py --output features.csv

Tables written to ``--db``:

    regression_features  — one row per scored listing: listing_id, log_rent,
                           and every predictor the model saw
    regression_results   — actual_rent, predicted_rent, percent_discount,
                           deal_score (0–10)

Only listings with ``listing_status = 'ACTIVE'`` are loaded. Inactive
listings are left out of the feature frame, the ridge fit, and the deal
scores. ``save_regression`` replaces both tables on every run. A listing
that was scored while active keeps its ``regression_results`` row after it
moves to another status, until the next run. That run omits it, and the
replace deletes the old score. Changing ``listing_status`` does not update
``regression_results`` by itself.

``percent_discount`` is ``100 * (predicted - actual) / predicted``. Positive
means the unit is listed below the model. ``deal_score`` maps that onto
0–10 with 5 = fairly priced: each 5 percentage points of discount moves the
score by 1, clipped at 0 (at least 25% over the model) and 10 (at least 25%
under). Predictions are out-of-fold so a listing is never scored by a model
that trained on it.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import db as dbm


DEFAULT_DB_PATH = "rentals.db"

# Out-of-fold ridge on standardized predictors. Alpha is chosen from this
# grid by out-of-fold RMSE on log_rent; the intercept is not penalized.
_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
_RIDGE_FOLDS = 5
_RIDGE_SEED = 0

# deal_score = clip(5 + percent_discount / _DEAL_POINTS_PER_SCORE, 0, 10)
# so ±25 percentage points of discount lands on 10 and 0.
_DEAL_POINTS_PER_SCORE = 5.0

# Raw image-score columns loaded from listing_scores. These are collapsed
# into two composites (apt_quality, common_quality) during engineering.
_APT_QUALITY_COLS = (
    "avg_natural_light", "avg_finish_quality", "avg_space_feeling",
    "avg_condition",
)
_COMMON_QUALITY_COLS = (
    "common_avg_finish_quality", "common_avg_condition", "common_avg_appeal",
)

# Union used for NaN-detection (deciding whether a listing has been scored).
_ALL_RAW_IMAGE_COLS = _APT_QUALITY_COLS + _COMMON_QUALITY_COLS

# ── Amenity tiers ─────────────────────────────────────────────────────────────
# Tier A: premium amenities that meaningfully drive rent up.
# Tier B: standard/expected amenities in managed NYC buildings.
# Tier C: low-signal tags (view descriptors, marketing fluff, near-universal).
# Amenities not listed in any tier are silently ignored.

AMENITY_TIER_A = frozenset({
    "pool", "hot_tub", "gym", "roof_deck", "private_roof_deck",
    "washer_dryer", "central_ac", "parking", "garage",
    "patio", "balcony", "terrace", "private_outdoor_space",
    "concierge", "media_room", "childrens_playroom",
    "hardwood_floors", "fireplace", "loft",
    "doorman", "valet_service", "valet",
})

AMENITY_TIER_B = frozenset({
    "elevator", "dishwasher", "laundry", "package_room",
    "storage_space", "bike_room", "shared_outdoor_space", "garden",
    "courtyard", "cold_storage", "locker_cage", "deck",
})

AMENITY_TIER_C = frozenset({
    "fios_available", "virtual", "city", "skyline", "water",
    "park", "view", "decorative", "part_time", "wheelchair_access",
    "live_in_super", "full_time", "assigned", "furnished",
})


# ── Loaders ──────────────────────────────────────────────────────────────────


def _load_listings(conn: sqlite3.Connection) -> pd.DataFrame:
    """Active listings plus listing_scores fields. Image-score columns are
    NaN for listings the scorer hasn't reached yet. Inactive listings are
    excluded from training and from deal scores.
    """
    return pd.read_sql(
        """
        SELECT
            l.listing_id,
            l.rent,
            l.beds, l.baths, l.sqft,
            l.neighborhood,
            l.building_type,
            (l.floor_plan_url IS NOT NULL) AS has_floor_plan,
            ls.avg_natural_light,
            ls.avg_finish_quality,
            ls.avg_space_feeling,
            ls.avg_condition,
            ls.common_avg_finish_quality,
            ls.common_avg_condition,
            ls.common_avg_appeal
        FROM listings l
        LEFT JOIN listing_scores ls ON l.listing_id = ls.listing_id
        WHERE l.listing_status = 'ACTIVE'
        """,
        conn,
    )


def _load_amenities_one_hot(
    conn: sqlite3.Connection, min_count: int = 10
) -> pd.DataFrame:
    """Pivot listing_amenities into a one-hot matrix (one column per amenity).

    Amenities present in fewer than ``min_count`` active listings are dropped
    to avoid fitting noise on the long tail of rare tags. Tags that appear
    only on inactive listings are ignored.
    """
    raw = pd.read_sql(
        """
        SELECT a.listing_id, a.amenity
        FROM listing_amenities a
        JOIN listings l ON l.listing_id = a.listing_id
        WHERE l.listing_status = 'ACTIVE'
        """,
        conn,
    )
    if raw.empty:
        return pd.DataFrame(columns=["listing_id"])

    counts = raw["amenity"].value_counts()
    keep = counts[counts >= min_count].index
    raw = raw[raw["amenity"].isin(keep)]

    one_hot = (
        pd.crosstab(raw["listing_id"], raw["amenity"])
        .clip(upper=1)
        .add_prefix("amen_")
        .astype(int)
        .reset_index()
    )
    return one_hot


# ── Engineering ──────────────────────────────────────────────────────────────


def engineer_features(
    df: pd.DataFrame,
    require_scores: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply target transform, sqft imputation, categorical encoding, and
    NaN handling. Returns a frame with individual amenity one-hots and raw
    image-score columns preserved. Use ``collapse_amenity_tiers`` and
    ``collapse_image_scores`` afterwards for optional compression.

    Args:
        df: raw frame from the SQL join (listings ⋈ scores ⋈ amenities).
        require_scores: if True (default), drop listings whose images haven't
            been scored yet. If False, mean-impute the score columns and add
            a ``has_image_scores`` flag.
        verbose: print row-count deltas for diagnostics.
    """
    df = df.copy()

    # ── Target ───────────────────────────────────────────────────────────────
    df["log_rent"] = np.log(df["rent"])

    # ── Drop or flag unscored rows ───────────────────────────────────────────
    if require_scores:
        before = len(df)
        df = df.dropna(subset=["avg_natural_light"]).reset_index(drop=True)
        if verbose:
            print(f"  Dropped {before - len(df)} listings without image scores "
                  f"({len(df)} remaining)")
    else:
        df["has_image_scores"] = df["avg_natural_light"].notna().astype(int)

    # ── Mean-impute remaining NaN in image-score columns ─────────────────────
    for c in _ALL_RAW_IMAGE_COLS:
        if c in df.columns and df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mean())

    # ── Sqft imputation (median by beds) + missingness flag ──────────────────
    df["has_sqft"] = df["sqft"].notna().astype(int)
    bed_medians = df.groupby("beds")["sqft"].transform("median")
    df["sqft"] = df["sqft"].fillna(bed_medians)
    df["sqft"] = df["sqft"].fillna(df["sqft"].median())

    # ── Bool/int normalization ───────────────────────────────────────────────
    for col in ("has_floor_plan"):
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ── One-hot encode categoricals ──────────────────────────────────────────
    df = pd.get_dummies(
        df,
        columns=["neighborhood", "building_type"],
        prefix=["nh", "bt"],
        drop_first=True,
        dtype=int,
    )

    # ── Drop the raw target column (log_rent is the canonical y) ─────────────
    df = df.drop(columns=["rent"])

    return _reorder_columns(df)


# ── Optional post-hoc transforms ─────────────────────────────────────────────


def collapse_image_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Replace correlated per-dimension image columns with two composites.

    ``apt_quality``    = mean(avg_natural_light, avg_finish_quality,
                              avg_space_feeling, avg_condition)
    ``common_quality`` = mean(common_avg_finish_quality,
                              common_avg_condition, common_avg_appeal)
    """
    df = df.copy()
    apt_present = [c for c in _APT_QUALITY_COLS if c in df.columns]
    com_present = [c for c in _COMMON_QUALITY_COLS if c in df.columns]

    if apt_present:
        df["apt_quality"] = df[apt_present].mean(axis=1)
        df = df.drop(columns=apt_present)
    if com_present:
        df["common_quality"] = df[com_present].mean(axis=1)
        df = df.drop(columns=com_present)

    return _reorder_columns(df)


def collapse_amenity_tiers(df: pd.DataFrame) -> pd.DataFrame:
    """Replace individual ``amen_*`` one-hot columns with three tier counts.

    Each listing gets ``amen_tier_a``, ``amen_tier_b``, ``amen_tier_c``
    equal to the number of its amenities falling into that tier.
    """
    df = df.copy()
    amen_cols = [c for c in df.columns if c.startswith("amen_")]
    if not amen_cols:
        return df

    a_cols = [c for c in amen_cols if c.removeprefix("amen_") in AMENITY_TIER_A]
    b_cols = [c for c in amen_cols if c.removeprefix("amen_") in AMENITY_TIER_B]
    c_cols = [c for c in amen_cols if c.removeprefix("amen_") in AMENITY_TIER_C]

    df["amen_tier_a"] = df[a_cols].sum(axis=1).astype(int) if a_cols else 0
    df["amen_tier_b"] = df[b_cols].sum(axis=1).astype(int) if b_cols else 0
    df["amen_tier_c"] = df[c_cols].sum(axis=1).astype(int) if c_cols else 0

    df = df.drop(columns=amen_cols)
    return _reorder_columns(df)


def _reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Stable, inspection-friendly column ordering: id, target, then numeric
    listing features, then one-hot groups, then amenity columns last."""
    front = [c for c in ("listing_id", "log_rent") if c in df.columns]

    amen = sorted(c for c in df.columns if c.startswith("amen_"))
    nh   = sorted(c for c in df.columns if c.startswith("nh_"))
    bt   = sorted(c for c in df.columns if c.startswith("bt_"))

    rest = [c for c in df.columns if c not in front + amen + nh + bt]
    return df[front + sorted(rest) + bt + nh + amen]


# ── Public entry point ───────────────────────────────────────────────────────


def prepare_features(
    db_path: str = DEFAULT_DB_PATH,
    amenity_min_count: int = 10,
    require_scores: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load + engineer in one call. Returns a DataFrame with individual
    amenity one-hot columns and raw image-score columns. Apply
    ``collapse_amenity_tiers`` / ``collapse_image_scores`` afterwards to
    compress these feature groups."""
    conn = sqlite3.connect(db_path)
    try:
        listings = _load_listings(conn)
        amen = _load_amenities_one_hot(conn, amenity_min_count)
    finally:
        conn.close()

    df = listings.merge(amen, on="listing_id", how="left")

    amen_cols = [c for c in df.columns if c.startswith("amen_")]
    if amen_cols:
        df[amen_cols] = df[amen_cols].fillna(0).astype(int)

    return engineer_features(df, require_scores=require_scores, verbose=verbose)


# ── Regression ───────────────────────────────────────────────────────────────


def _feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predictors and log_rent, in the frame's row order."""
    cols = [c for c in df.columns if c not in ("listing_id", "log_rent")]
    X = df[cols].to_numpy(dtype=float)
    y = df["log_rent"].to_numpy(dtype=float)
    return X, y


def _ridge_predict_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Standardize on the training fold, then ridge with a free intercept."""
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    Z_train = (X_train - mu) / sigma
    Z_test = (X_test - mu) / sigma

    n, p = Z_train.shape
    A = np.empty((p + 1, p + 1), dtype=float)
    A[0, 0] = n
    A[0, 1:] = Z_train.sum(axis=0)
    A[1:, 0] = A[0, 1:]
    A[1:, 1:] = Z_train.T @ Z_train + alpha * np.eye(p)

    rhs = np.empty(p + 1, dtype=float)
    rhs[0] = y_train.sum()
    rhs[1:] = Z_train.T @ y_train
    beta = np.linalg.solve(A, rhs)
    return beta[0] + Z_test @ beta[1:]


def _oof_log_predictions(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    n_splits: int,
) -> np.ndarray:
    """Log-rent predictions where each row is scored by a model fit without it."""
    n = len(y)
    n_splits = min(n_splits, n)
    if n_splits < 2:
        return _ridge_predict_fold(X, y, X, alpha)

    order = np.random.default_rng(_RIDGE_SEED).permutation(n)
    folds = np.array_split(order, n_splits)
    pred = np.empty(n, dtype=float)
    for i, test in enumerate(folds):
        train = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        pred[test] = _ridge_predict_fold(X[train], y[train], X[test], alpha)
    return pred


def _select_ridge(
    X: np.ndarray, y: np.ndarray, n_splits: int
) -> tuple[float, np.ndarray, int]:
    """Pick the alpha with the lowest out-of-fold RMSE. Returns that alpha,
    its out-of-fold log predictions, and the fold count actually used."""
    folds = min(n_splits, len(y))
    best_alpha = _RIDGE_ALPHAS[0]
    best_pred = _oof_log_predictions(X, y, best_alpha, folds)
    best_rmse = float(np.sqrt(np.mean((best_pred - y) ** 2)))
    for alpha in _RIDGE_ALPHAS[1:]:
        pred = _oof_log_predictions(X, y, alpha, folds)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        if rmse < best_rmse:
            best_alpha, best_pred, best_rmse = alpha, pred, rmse
    return best_alpha, best_pred, folds


def deal_score(percent_discount: np.ndarray) -> np.ndarray:
    """Map percent discount onto 0–10. Fairly priced (0%) scores 5."""
    score = 5.0 + np.asarray(percent_discount, dtype=float) / _DEAL_POINTS_PER_SCORE
    return np.clip(np.round(score, 2), 0.0, 10.0)


def predict_deals(df: pd.DataFrame, rents: pd.Series) -> pd.DataFrame:
    """Predicted rent, percent discount, and deal score for each feature row.

    ``rents`` is actual rent indexed by ``listing_id``. Predictions come from
    an out-of-fold ridge on ``log_rent`` and are rounded to the nearest dollar
    before the discount is computed, so the stored rents and the discount agree.
    """
    if "log_rent" not in df.columns or "listing_id" not in df.columns:
        raise ValueError("feature frame must include listing_id and log_rent")

    missing = set(df["listing_id"]) - set(rents.index)
    if missing:
        raise ValueError(f"{len(missing)} listings have features but no rent")

    X, y = _feature_matrix(df)
    alpha, log_pred, folds = _select_ridge(X, y, _RIDGE_FOLDS)

    actual = rents.loc[df["listing_id"]].to_numpy(dtype=float)
    predicted = np.maximum(np.rint(np.exp(log_pred)), 1.0)
    percent_discount = np.round(100.0 * (predicted - actual) / predicted, 2)

    model = f"ridge_log_rent_oof_k{folds}_alpha{alpha:g}"
    scored_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return pd.DataFrame({
        "listing_id": df["listing_id"].to_numpy(),
        "actual_rent": actual.astype(int),
        "predicted_rent": predicted.astype(int),
        "percent_discount": percent_discount,
        "deal_score": deal_score(percent_discount),
        "model": model,
        "scored_at": scored_at,
    })


def _load_rents(conn: sqlite3.Connection) -> pd.Series:
    rents = pd.read_sql(
        "SELECT listing_id, rent FROM listings WHERE listing_status = 'ACTIVE'",
        conn,
    )
    return rents.set_index("listing_id")["rent"]


def save_regression(
    conn: sqlite3.Connection,
    features: pd.DataFrame,
    results: pd.DataFrame,
) -> None:
    """Replace ``regression_features`` and ``regression_results`` with this run.

    Both tables are rewritten from the active listings in this fit. A listing
    that left ACTIVE since the previous run is absent from ``results``, so the
    delete drops its deal score. Status changes between runs leave the old
    row in place until this replace.
    """
    features.to_sql("regression_features", conn, if_exists="replace", index=False)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_regression_features_listing "
        "ON regression_features(listing_id)"
    )
    conn.execute("DELETE FROM regression_results")
    conn.executemany(
        """
        INSERT INTO regression_results (
            listing_id, actual_rent, predicted_rent, percent_discount,
            deal_score, model, scored_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        list(results.itertuples(index=False, name=None)),
    )
    import bq

    bq.replace_regression(features, results)


def run_regression(
    db_path: str = DEFAULT_DB_PATH,
    *,
    require_scores: bool = True,
    raw: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the feature frame from active listings, score those deals, and write both tables.

    Inactive listings are not predictors and do not receive a deal score.
    Returns ``(features, results)``.
    """
    features = prepare_features(
        db_path=db_path,
        require_scores=require_scores,
        verbose=verbose,
    )
    if not raw:
        features = collapse_image_scores(features)
        features = collapse_amenity_tiers(features)

    with dbm.open_db(db_path) as conn:
        results = predict_deals(features, _load_rents(conn))
        save_regression(conn, features, results)
    return features, results


# ── CLI ──────────────────────────────────────────────────────────────────────


def _summary(df: pd.DataFrame) -> None:
    n_amen = sum(1 for c in df.columns if c.startswith("amen_"))
    n_nh   = sum(1 for c in df.columns if c.startswith("nh_"))
    n_bt   = sum(1 for c in df.columns if c.startswith("bt_"))
    n_features = len(df.columns) - 2  # excl. listing_id + log_rent

    print(f"\nTraining frame: {len(df)} rows × {len(df.columns)} columns "
          f"({n_features} features)")
    print(f"  target log_rent: mean={df['log_rent'].mean():.3f}, "
          f"std={df['log_rent'].std():.3f}")
    print(f"  amenity columns:                     {n_amen}")
    print(f"  neighborhood (one-hot, drop_first):  {n_nh}")
    print(f"  building_type (one-hot, drop_first): {n_bt}")
    print(f"  numeric listing features:            "
          f"{n_features - n_amen - n_nh - n_bt}")


def _save(df: pd.DataFrame, path: Path) -> None:
    if path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    print(f"Saved to {path}")


def _summary_results(results: pd.DataFrame, db_path: str) -> None:
    err = (results["predicted_rent"] - results["actual_rent"]).abs()
    print(f"\nWrote regression_features and regression_results to {db_path}")
    print(f"  model: {results['model'].iloc[0]}")
    print(f"  listings scored: {len(results)}")
    print(f"  MAE ${err.mean():,.0f}   median |error| ${err.median():,.0f}")
    print("  best deals (deal_score, percent_discount, actual → predicted):")
    top = results.sort_values(
        ["deal_score", "percent_discount"], ascending=False
    ).head(5)
    for row in top.itertuples(index=False):
        print(
            f"    {row.deal_score:4.1f}   {row.percent_discount:+6.1f}%   "
            f"${row.actual_rent:,.0f} → ${row.predicted_rent:,.0f}   {row.listing_id}"
        )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--db", default=DEFAULT_DB_PATH, help="SQLite database path")
    p.add_argument("--output", default=None,
                   help="Optional path to dump the frame (.csv or .parquet)")
    p.add_argument("--include-unscored", action="store_true",
                   help="Keep listings missing image scores (mean-impute + flag)")
    p.add_argument("--raw", action="store_true",
                   help="Skip amenity-tier and image-composite transforms")
    args = p.parse_args()

    df, results = run_regression(
        db_path=args.db,
        require_scores=not args.include_unscored,
        raw=args.raw,
    )

    _summary(df)
    _summary_results(results, args.db)

    if args.output:
        _save(df, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())
