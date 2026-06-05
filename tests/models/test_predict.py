import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pytest

from geoai.models.predict import run_predictions, _load_model


def _make_fake_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("""
        CREATE TABLE listings (
            id BIGINT PRIMARY KEY, latitude DOUBLE, longitude DOUBLE,
            price DOUBLE, accommodates INTEGER, bedrooms DOUBLE, beds DOUBLE,
            room_type VARCHAR, minimum_nights INTEGER, review_scores_rating DOUBLE,
            host_is_superhost BOOLEAN, number_of_reviews INTEGER
        )
    """)
    con.execute("""
        INSERT INTO listings VALUES
        (1, 41.14, -8.61, 100.0, 2, 1.0, 1.0, 'Entire home/apt', 2, 4.8, true, 50),
        (2, 41.15, -8.62, 80.0,  1, 1.0, 1.0, 'Private room',    1, 4.5, false, 30)
    """)
    con.execute("""
        CREATE TABLE listing_features (
            listing_id BIGINT PRIMARY KEY,
            dist_city_center_km DOUBLE, dist_nearest_metro_km DOUBLE,
            dist_nearest_station_km DOUBLE, dist_nearest_supermarket_km DOUBLE,
            dist_airport_km DOUBLE, travel_time_airport_min DOUBLE,
            restaurants_250m INTEGER, restaurants_500m INTEGER, bars_500m INTEGER,
            cafes_500m INTEGER, supermarkets_1km INTEGER, attractions_1km INTEGER,
            museums_2km INTEGER, parks_500m INTEGER, amenity_density_1km INTEGER,
            restaurant_density DOUBLE, listings_500m INTEGER, listings_1km INTEGER,
            avg_price_500m DOUBLE, median_price_neighbourhood DOUBLE,
            walkability_score DOUBLE, dist_livraria_lello_km DOUBLE,
            dist_torre_clerigos_km DOUBLE, dist_ribeira_km DOUBLE,
            dist_ponte_luis_km DOUBLE, dist_mercado_bolhao_km DOUBLE,
            dist_jardins_cristal_km DOUBLE, h3_cell_r8 VARCHAR,
            occupancy_rate_30d DOUBLE, occupancy_rate_90d DOUBLE,
            occupancy_rate_365d DOUBLE
        )
    """)
    con.execute("""
        INSERT INTO listing_features VALUES
        (1, 0.5, 0.3, 0.4, 0.2, 8.0, 20.0, 10, 20, 15, 8, 3, 5, 2, 4, 30, 0.8, 5, 12, 95.0, 90.0, 88.0, 0.3, 0.2, 0.1, 0.4, 0.5, 0.6, '88abc123ffffff', 0.8, 0.75, 0.72),
        (2, 1.2, 0.8, 0.9, 0.6, 9.0, 22.0,  5, 10,  8, 4, 2, 3, 1, 2, 18, 0.5, 3,  8, 80.0, 75.0, 70.0, 0.8, 0.7, 0.6, 0.9, 1.0, 1.1, '88abc456ffffff', 0.6, 0.58, 0.55)
    """)
    con.close()
    return db_path


def _make_fake_model(n_features: int, tmp_path: Path, name: str):
    model = MagicMock()
    model.predict.return_value = np.full(2, 0.5)
    artifact = {"model": model, "feature_names": [f"f{i}" for i in range(n_features)]}
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    return path


def _make_artifacts(db_path):
    """Build artifact dicts with MagicMock models (no pickling needed — patched via _load_model)."""
    from geoai.models.features import build_feature_matrix, prepare_X_y_price, prepare_X_y_occupancy
    df = build_feature_matrix(db_path)
    X_p, _, pf = prepare_X_y_price(df)
    X_o, _, of = prepare_X_y_occupancy(df)

    price_model = MagicMock()
    price_model.predict.return_value = np.log(np.full(len(df), 90.0))
    occ_model = MagicMock()
    occ_model.predict.return_value = np.full(len(df), 0.70)

    price_artifact = {"model": price_model, "feature_names": list(pf)}
    occ_artifact = {"model": occ_model, "feature_names": list(of)}
    return price_artifact, occ_artifact, df, pf, of


class TestLoadModel:
    def test_raises_if_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Run `python -m geoai.models.runner`"):
            _load_model(tmp_path / "missing.pkl")

    def test_loads_artifact(self, tmp_path):
        path = tmp_path / "model.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": "ok"}, f)
        assert _load_model(path) == {"model": "ok"}


class TestRunPredictions:
    def test_creates_all_four_tables(self, tmp_path):
        db_path = _make_fake_db(tmp_path)
        price_artifact, occ_artifact, df, pf, of = _make_artifacts(db_path)

        def fake_load_model(path):
            if "price" in str(path):
                return price_artifact
            return occ_artifact

        with patch("geoai.models.predict.PRICE_MODEL_PATH", tmp_path / "price_model.pkl"), \
             patch("geoai.models.predict.OCCUPANCY_MODEL_PATH", tmp_path / "occupancy_model.pkl"), \
             patch("geoai.models.predict.DB_PATH", db_path), \
             patch("geoai.models.predict._load_model", side_effect=fake_load_model), \
             patch("geoai.models.predict.shap_lib") as mock_shap:

            shap_calls = [np.zeros((len(df), len(pf))), np.zeros((len(df), len(of)))]
            mock_shap.TreeExplainer.return_value.shap_values.side_effect = shap_calls
            mock_shap.TreeExplainer.return_value.expected_value = 0.0
            mock_shap.sample.side_effect = lambda x, n: x[:n]

            run_predictions(db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        tables = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()}
        con.close()

        assert "listing_predictions" in tables
        assert "hex_aggregates" in tables
        assert "shap_global" in tables
        assert "hex_shap" in tables

    def test_listing_predictions_row_count(self, tmp_path):
        db_path = _make_fake_db(tmp_path)
        price_artifact, occ_artifact, df, pf, of = _make_artifacts(db_path)

        def fake_load_model(path):
            if "price" in str(path):
                return price_artifact
            return occ_artifact

        with patch("geoai.models.predict.PRICE_MODEL_PATH", tmp_path / "price_model.pkl"), \
             patch("geoai.models.predict.OCCUPANCY_MODEL_PATH", tmp_path / "occupancy_model.pkl"), \
             patch("geoai.models.predict.DB_PATH", db_path), \
             patch("geoai.models.predict._load_model", side_effect=fake_load_model), \
             patch("geoai.models.predict.shap_lib") as mock_shap:

            shap_calls = [np.zeros((len(df), len(pf))), np.zeros((len(df), len(of)))]
            mock_shap.TreeExplainer.return_value.shap_values.side_effect = shap_calls
            mock_shap.TreeExplainer.return_value.expected_value = 0.0
            mock_shap.sample.side_effect = lambda x, n: x[:n]

            run_predictions(db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        count = con.execute("SELECT COUNT(*) FROM listing_predictions").fetchone()[0]
        con.close()
        assert count == 2

    def test_shap_global_has_both_models(self, tmp_path):
        db_path = _make_fake_db(tmp_path)
        price_artifact, occ_artifact, df, pf, of = _make_artifacts(db_path)

        def fake_load_model(path):
            if "price" in str(path):
                return price_artifact
            return occ_artifact

        with patch("geoai.models.predict.PRICE_MODEL_PATH", tmp_path / "price_model.pkl"), \
             patch("geoai.models.predict.OCCUPANCY_MODEL_PATH", tmp_path / "occupancy_model.pkl"), \
             patch("geoai.models.predict.DB_PATH", db_path), \
             patch("geoai.models.predict._load_model", side_effect=fake_load_model), \
             patch("geoai.models.predict.shap_lib") as mock_shap:

            shap_calls = [np.zeros((len(df), len(pf))), np.zeros((len(df), len(of)))]
            mock_shap.TreeExplainer.return_value.shap_values.side_effect = shap_calls
            mock_shap.TreeExplainer.return_value.expected_value = 0.0
            mock_shap.sample.side_effect = lambda x, n: x[:n]

            run_predictions(db_path)

        con = duckdb.connect(str(db_path), read_only=True)
        models = {r[0] for r in con.execute("SELECT DISTINCT model FROM shap_global").fetchall()}
        con.close()
        assert models == {"price", "occupancy"}
