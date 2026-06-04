# GeoAI Short-Term Rental Intelligence Platform
## Airbnb Intelligence Platform

### Tagline

A geospatial AI platform that helps investors, hosts, and analysts understand what drives Airbnb performance, predict revenue opportunities, estimate causal effects of amenities and location factors, and discover underpriced investment opportunities.

---

# Executive Summary

Most Airbnb projects predict listing prices.

This project answers deeper questions:

- What drives Airbnb pricing?
- What drives occupancy?
- Where should investors buy properties?
- Which amenities truly increase revenue?
- Which neighborhoods are undervalued?
- What is the causal impact of location and property features?

The platform combines:

- Geospatial Analytics
- Machine Learning
- Explainable AI
- Causal Inference
- LLM-Powered Insights
- Interactive Maps
- FastAPI

---

# Portfolio Positioning

This project demonstrates:

- Applied AI
- Machine Learning
- Geospatial Analysis
- Causal Inference
- Data Engineering
- Explainable AI
- LLM Applications
- FastAPI Development
- Product Thinking

Target Roles:

- Applied AI Engineer
- Data Scientist
- Senior Data Scientist
- Geospatial Data Scientist
- Analytics Engineer
- Machine Learning Engineer

---

# Product Vision

## User Types

### Airbnb Host

Questions:

- How much should I charge?
- Which amenities matter most?
- How can I improve occupancy?

---

### Investor

Questions:

- Where should I buy?
- Which neighborhoods are emerging?
- Which areas are undervalued?

---

### Analyst

Questions:

- What factors drive performance?
- What causes higher occupancy?
- What drives long-term revenue?

---

# System Architecture

```text
Inside Airbnb
OpenStreetMap
Census Data
Tourism Data
Weather Data
Event Data
      |
      v

Feature Engineering Layer
      |
      v

DuckDB Feature Store
      |
      +--------------------------+
      |                          |
      v                          v

Machine Learning         Geospatial Engine
      |                          |
      +-------------+------------+
                    |
                    v

Explainability Layer
                    |
                    v

LLM Insight Engine
                    |
                    v

FastAPI + Dashboard
```

---

# Data Sources

## Airbnb Listings

Primary Source:

https://insideairbnb.com

Data:

- listing_id
- latitude
- longitude
- room_type
- property_type
- amenities
- reviews
- availability
- host information
- pricing

---

## OpenStreetMap

Libraries:

- OSMnx
- GeoPandas

Extract:

- restaurants
- bars
- cafes
- supermarkets
- museums
- attractions
- parks
- metro stations
- train stations
- airports

---

## Demographics

Potential Sources:

- Census
- Eurostat
- National Statistics Offices

Variables:

- income
- population density
- age distribution
- education

---

## Tourism Data

Examples:

- visitor arrivals
- hotel occupancy
- seasonality
- tourism hotspots

---

## Weather Data

Optional Enhancement

Variables:

- temperature
- rainfall
- seasonality

---

# Data Engineering

## Pipeline Design

### Ingestion

Scheduled ingestion:

- Airbnb snapshots
- OSM updates
- tourism data

---

### Transformation

Use:

- Polars
- DuckDB

Tasks:

- cleaning
- geospatial joins
- feature creation

---

### Storage

DuckDB

Tables:

- listings
- amenities
- neighborhoods
- poi_features
- occupancy_features
- model_predictions

---

# Geospatial Feature Engineering

This is the project's competitive advantage.

---

## Accessibility Features

Calculate:

- distance_to_metro
- distance_to_train
- distance_to_airport

---

## Tourism Features

Count within radius:

- attractions
- museums
- landmarks

Radius examples:

- 500m
- 1km
- 2km

---

## Nightlife Features

Count:

- bars
- clubs
- restaurants

---

## Convenience Features

Count:

- supermarkets
- pharmacies
- convenience stores

---

## Competition Features

Nearby Airbnb listings:

- within 500m
- within 1km

Calculate:

- listing density
- average nearby price

---

## Walkability Score

Create custom score using:

- POI density
- transport accessibility

---

# Machine Learning Layer

## Model 1: Price Prediction

Target:

price

Models:

### Baseline

Linear Regression

### Production

CatBoost

Why:

- excellent tabular performance
- categorical features
- low preprocessing

Metrics:

- RMSE
- MAE
- MAPE

---

## Model 2: Occupancy Prediction

Target:

estimated occupancy

Potential proxies:

- reviews
- calendar availability

Models:

- CatBoost
- LightGBM

---

## Model 3: Revenue Prediction

Formula:

Revenue = Price × Occupancy

Target:

annual revenue

---

## Model 4: Investment Score

Custom score:

Expected Revenue
-
Competition Risk
+
Neighborhood Growth
+
Accessibility

Produces ranking.

---

# Explainable AI

## SHAP Analysis

Generate:

Global Feature Importance

Example:

1. Distance to city center
2. Number of reviews
3. Metro accessibility
4. Superhost status

---

## Local Explanations

Per listing:

Predicted Price: €160

Drivers:

+ City center proximity
+ Metro station nearby
+ High review score
- High competition density

---

# Geospatial Visualization

## Technology

Preferred:

- PyDeck
- Deck.gl

---

## Map Layers

### Price Heatmap

Visualize predicted prices.

---

### Revenue Heatmap

Visualize annual revenue.

---

### Occupancy Heatmap

Visualize demand.

---

### Competition Map

Visualize Airbnb density.

---

### Opportunity Map

Most interesting layer.

Formula:

Predicted Revenue
-
Actual Revenue

Positive values indicate opportunities.

---

# Investor Intelligence Module

## Neighborhood Ranking

Rank neighborhoods by:

- revenue potential
- growth potential
- competition
- accessibility

---

## Investment Simulator

User inputs:

Property Price
Bedrooms
Location

System estimates:

- annual revenue
- occupancy
- ROI

---

# Causal Inference Module

This is the differentiator.

Most Airbnb projects stop at prediction.

---

## Questions

Does Superhost status increase occupancy?

Does adding a pool increase revenue?

Does metro proximity matter?

Does self-check-in improve bookings?

---

## Treatment Examples

- Superhost
- Pool
- Self Check-in
- Entire Home
- Metro Access

---

## Outcomes

- Occupancy
- Revenue
- Price

---

## Methods

### Matching

Propensity Score Matching

---

### Double Machine Learning

EconML

CausalForestDML

---

### Heterogeneous Effects

Estimate impacts by:

- city
- neighborhood
- property type

---

# LLM Property Analyst

## User Query

"Why is this listing expensive?"

Agent generates:

"This property commands a premium due to proximity to major tourist attractions, excellent transport accessibility, and strong review performance."

---

## Investment Report Generator

User Query:

"Should I invest in Porto or Lisbon?"

Agent:

- retrieves metrics
- compares neighborhoods
- summarizes opportunities

---

# FastAPI Architecture

## Endpoints

### Prediction

POST /predict-price

---

### Revenue Estimate

POST /predict-revenue

---

### Neighborhood Analysis

GET /neighborhood/{id}

---

### Opportunity Search

GET /investment-opportunities

---

### Causal Analysis

POST /causal-effect

---

### AI Report

POST /generate-report

---

# Dashboard Design

## Streamlit MVP

Pages:

- Overview
- Listings Explorer
- Neighborhood Explorer
- Revenue Map
- Opportunity Map
- Causal Explorer

---

## Future React Frontend

Enhanced UX

Interactive Deck.gl maps

---

# Technology Stack

## Data

- DuckDB
- Polars

## Geospatial

- GeoPandas
- OSMnx
- H3
- Shapely

## ML

- CatBoost
- LightGBM
- Scikit-Learn

## Explainability

- SHAP

## Causal Inference

- DoWhy
- EconML

## AI

- OpenAI
- LangGraph

## Backend

- FastAPI

## Visualization

- PyDeck
- Deck.gl

---

# Deployment

## API

- Railway
- Render

## Storage

- DuckDB

## Frontend

- Streamlit Cloud

---

# Stretch Goals

## Forecast Future Airbnb Demand

Forecast:

- occupancy
- revenue

using:

- MLForecast
- LightGBM

---

## GeoAI Recommendation Engine

"Find the best neighborhood under €250,000."

---

## Satellite Data Integration

Add:

- Sentinel imagery
- Nighttime lights

Predict neighborhood growth.

---

## Multi-City Comparison

Compare:

- Porto
- Lisbon
- Madrid
- Barcelona
- Paris

---

# Development Roadmap

## Phase 1 (2 Weeks)

- Airbnb ingestion
- OSM ingestion
- DuckDB setup

Deliverable:

Data warehouse.

---

## Phase 2 (2 Weeks)

- Geospatial feature engineering
- Accessibility metrics
- Tourism metrics

Deliverable:

Feature store.

---

## Phase 3 (2 Weeks)

- Price prediction
- Revenue prediction
- Model evaluation

Deliverable:

ML pipeline.

---

## Phase 4 (2 Weeks)

- SHAP analysis
- Explainability reports

Deliverable:

Model interpretation layer.

---

## Phase 5 (2 Weeks)

- Interactive maps
- Streamlit dashboard

Deliverable:

Visual analytics platform.

---

## Phase 6 (2 Weeks)

- FastAPI
- Deployment

Deliverable:

Public application.

---

## Phase 7 (Optional)

- Causal inference
- LLM analyst
- Investment reports

Deliverable:

GeoAI Intelligence Platform.

---

# Success Criteria

A recruiter should immediately conclude:

"This is not a simple Airbnb pricing model.

This is a production-grade GeoAI platform combining geospatial analytics, machine learning, explainability, causal inference, APIs, and AI-generated insights."
