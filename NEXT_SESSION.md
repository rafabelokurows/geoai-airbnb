# Next Session TODOs

## 1. Feature Importance Analysis ✓ DONE
- Pruned 7 sparse amenity flags (<2% hit rate)
- Expanded remaining groups + added 8 description keyword features
- Price RMSE: 54.32 → 53.59 (-1.3%)

## 2. Per-Room-Type Models
Currently one shared LightGBM for all room types; `room_type` enters as one-hot dummies.
Private rooms and entire places have different occupancy dynamics — separate models could improve MAE (currently stuck at 0.164, target <0.15).

- Train one price model + one occupancy model per main room type:
  - `Entire home/apt` (~10k listings — enough data)
  - `Private room` (~1.5k listings)
  - `Hotel room` (~370 listings — may need to merge with Private room)
- Refactor `src/geoai/models/price.py` + `occupancy.py` to accept a filtered DataFrame
- Refactor `src/geoai/models/evaluate.py` to loop over room types and report per-type RMSE/MAE
- Update `src/geoai/models/predict.py` to route each listing through its room-type model
- Update SHAP pipeline in `src/geoai/explainability/shap_explainer.py` to hold one explainer per model
- Compare aggregate RMSE/MAE vs current single-model baseline

## 3. More Analytics + Narratives
- Add narrative text to AnalyticsSidebar: "Listings in this hex earn X% more than city average"
- Opportunity score: combine high revenue + low competition into a single hex score
- Top amenities by impact: "Having a pool adds ~€X/night on average" (from SHAP)
- Neighbourhood comparison table: rank all neighbourhoods by avg_revenue
- Occupancy vs price scatter per hex (are high-price hexes actually occupied?)
- Narrative on hex click: auto-generate 2-3 sentence summary of why that hex is/isn't a good opportunity
