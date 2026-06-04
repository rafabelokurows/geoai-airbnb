# GeoAI Airbnb Intelligence Platform — Feature Specifications

---

## Phase 1: Data Warehouse

### 1.1 Airbnb Listings Ingestion
- **Input:** `data/raw/airbnb/listings.csv.gz`
- **Process:** Parse with Polars; clean price (`$1,200.00` → `1200.0`); parse superhost flag; drop rows missing coordinates
- **Output:** `listings` table in DuckDB
- **Key columns:** `id`, `latitude`, `longitude`, `price`, `room_type`, `property_type`, `accommodates`, `bedrooms`, `amenities`, `host_is_superhost`, `review_scores_rating`

### 1.2 Calendar Ingestion
- **Input:** `data/raw/airbnb/calendar.csv.gz`
- **Process:** Parse availability (`t`/`f`), price per night, date range
- **Output:** `calendar` table in DuckDB
- **Key columns:** `listing_id`, `date`, `available`, `price`, `minimum_nights`, `maximum_nights`
- **Note:** Calendar data enables **real occupancy estimation** — far more accurate than review-count proxies

### 1.3 Reviews Ingestion
- **Input:** `data/raw/airbnb/reviews.csv.gz`
- **Output:** `reviews` table in DuckDB
- **Key columns:** `listing_id`, `date`, `reviewer_id`, `comments`

### 1.4 Neighbourhood Boundaries
- **Input:** `data/raw/airbnb/neighbourhoods.geojson`
- **Output:** `neighbourhoods` table in DuckDB (WKT geometry)
- **Use:** Spatial joins, neighbourhood-level aggregations

### 1.5 OSM POI Ingestion
- **Input:** OSMnx query for Porto, Portugal
- **POI categories:** restaurants, bars, cafes, supermarkets, pharmacies, museums, attractions, parks, railway stations, subway entrances
- **Output:** `poi_features` table with `osm_id`, `poi_type`, `poi_subtype`, `latitude`, `longitude`, `geometry_wkt`

---

## Phase 2: Geospatial Feature Engineering

### 2.1 Accessibility Features
- `distance_to_nearest_metro` — Haversine distance to closest subway entrance
- `distance_to_nearest_train_station` — distance to railway station
- `distance_to_city_center` — distance to Praça da Liberdade (Porto center)

### 2.2 POI Density Features (per listing, per radius)
Radii: 250m, 500m, 1km, 2km

| Feature | Description |
|---------|-------------|
| `restaurants_500m` | Restaurant count within 500m |
| `bars_500m` | Bar/pub count within 500m |
| `supermarkets_1km` | Supermarket count within 1km |
| `attractions_1km` | Tourist attraction count within 1km |
| `museums_2km` | Museum count within 2km |
| `parks_500m` | Park/garden count within 500m |

### 2.3 Competition Features
- `listings_500m` — competing Airbnb listings within 500m
- `listings_1km` — competing listings within 1km
- `avg_price_500m` — average listing price within 500m
- `median_price_neighbourhood` — neighbourhood-level price median

### 2.4 Walkability Score
- Custom composite: weighted sum of POI density + transit accessibility
- Range: 0–100

### 2.5 H3 Hexagon Aggregation
- Assign each listing an H3 hex cell (resolution 8 ≈ 460m diameter)
- Aggregate: avg price, avg occupancy, listing count per hex
- Used for: heatmap layers, neighbourhood comparison

### 2.6 Occupancy Estimation (from Calendar)
- `occupancy_rate_30d` — % of past 30 days booked
- `occupancy_rate_90d` — % of past 90 days booked
- `occupancy_rate_365d` — annual occupancy estimate
- Method: `available = f` on past dates → booked

---

## Phase 3: Machine Learning Models

### 3.1 Price Prediction Model
- **Target:** `price` (nightly rate, EUR)
- **Features:** geospatial features (Phase 2) + listing attributes
- **Models:** Linear Regression (baseline) → CatBoost (production)
- **Metrics:** RMSE, MAE, MAPE
- **Output:** `predicted_price` per listing

### 3.2 Occupancy Prediction Model
- **Target:** `occupancy_rate_365d` (from calendar data)
- **Features:** same as price model + price itself
- **Models:** LightGBM
- **Metrics:** RMSE, MAE, R²

### 3.3 Revenue Prediction
- **Formula:** `predicted_revenue = predicted_price × predicted_occupancy_rate × 365`
- **Output:** `predicted_annual_revenue` per listing

### 3.4 Investment Score
- **Formula:** `score = predicted_revenue - competition_penalty + accessibility_bonus + neighborhood_growth`
- **Output:** ranked listing/neighbourhood investment score
- **Use:** Investor Intelligence Module

---

## Phase 4: Explainable AI

### 4.1 Global Feature Importance (SHAP)
- TreeSHAP for CatBoost/LightGBM models
- Output: ranked feature importance chart
- Example output:
  1. distance_to_city_center
  2. accommodates
  3. review_scores_rating
  4. metro_accessibility
  5. competition density

### 4.2 Local Explanation (per listing)
- SHAP waterfall plot per listing
- API response format:
  ```json
  {
    "predicted_price": 142.0,
    "drivers": [
      {"feature": "city_center_proximity", "impact": +28.5},
      {"feature": "metro_nearby", "impact": +12.0},
      {"feature": "competition_density", "impact": -8.3}
    ]
  }
  ```

---

## Phase 5: Geospatial Visualization (Streamlit + PyDeck)

### 5.1 Dashboard Pages

| Page | Description |
|------|-------------|
| Overview | City-level KPIs, listing count, avg price, avg occupancy |
| Listings Explorer | Filter listings by room type, price, neighbourhood |
| Neighbourhood Ranking | Sort by revenue potential, occupancy, competition |
| Price Heatmap | H3 hex map colored by predicted price |
| Revenue Heatmap | H3 hex map colored by predicted annual revenue |
| Occupancy Heatmap | H3 hex map colored by occupancy rate |
| Competition Map | Listing density heatmap |
| Opportunity Map | `predicted_revenue - actual_revenue` — positive = underpriced |

### 5.2 Opportunity Map (Key Feature)
- **Formula:** `opportunity_score = predicted_annual_revenue - estimated_actual_revenue`
- Positive score → listing is underperforming its potential
- Useful for investors to find undervalued properties

---

## Phase 6: FastAPI

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict-price` | Predict nightly price for a listing spec |
| POST | `/predict-revenue` | Predict annual revenue |
| GET | `/neighbourhood/{id}` | Neighbourhood metrics and ranking |
| GET | `/investment-opportunities` | Top N opportunity listings |
| POST | `/causal-effect` | Estimate causal effect of a treatment |
| POST | `/generate-report` | LLM-generated investment report |
| GET | `/health` | Health check |

---

## Phase 7: Causal Inference + LLM Analyst

### 7.1 Causal Questions Supported

| Treatment | Outcome | Method |
|-----------|---------|--------|
| Superhost status | Occupancy rate | Propensity Score Matching |
| Pool availability | Annual revenue | CausalForestDML (EconML) |
| Self check-in | Bookings per month | DoWhy |
| Entire home (vs shared) | Price premium | DoWhy |
| Metro proximity (<300m) | Occupancy | Propensity Score Matching |

### 7.2 Heterogeneous Effects
- Estimate treatment effects by: property type, neighbourhood, room type
- "Does a pool matter more in tourist vs residential areas?"

### 7.3 LLM Property Analyst
- **Model:** OpenAI GPT-4o via LangGraph agent
- **Tool calls:** query DuckDB, retrieve SHAP values, retrieve causal estimates
- **Example query:** "Why is this listing underperforming its neighbourhood?"
- **Example query:** "Should I invest in Bonfim or Cedofeita?"

### 7.4 Investment Report Generator
- Input: neighbourhood name or property coordinates
- Output: structured markdown report with:
  - Market summary (avg price, occupancy, competition)
  - Top performing listings and why
  - Causal effects of key amenities
  - Investment recommendation with ROI estimate

---

## Stretch Goals

| Feature | Description |
|---------|-------------|
| Demand Forecasting | MLForecast/LightGBM to predict future occupancy by season |
| GeoAI Recommendation Engine | "Find best neighbourhood under €250K investment" |
| Multi-City Comparison | Porto vs Lisbon vs Barcelona |
| Satellite Imagery | Sentinel-2 nighttime lights for neighborhood growth prediction |
