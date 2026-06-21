import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query

from geoai.api.deps import AppState, get_state
from geoai.api.schemas import Driver, ExplainResponse, FeatureImportance
from geoai.explainability.shap_explainer import explain_listing

router = APIRouter()


@router.get("/explain/global", response_model=list[FeatureImportance])
def explain_global(
    top_n: int = Query(default=10, le=50),
    state: AppState = Depends(get_state),
) -> list[FeatureImportance]:
    sample = min(500, len(state.X_price))
    shap_vals = state.price_explainer.shap_values(state.X_price[:sample])
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:top_n]
    return [
        FeatureImportance(feature=state.price_features[i], importance=float(mean_abs[i]))
        for i in idx
    ]


@router.get("/listings/{listing_id}/explain", response_model=ExplainResponse)
def explain(listing_id: int, state: AppState = Depends(get_state)) -> ExplainResponse:
    idx = state.listing_id_to_idx.get(listing_id)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found or has no predictions")
    group = state.listing_id_to_group.get(listing_id, "Entire home/apt")
    explainer = state.price_explainers.get(group, state.price_explainer)
    result = explain_listing(
        explainer,
        state.X_price[idx],
        state.price_features,
        top_n=5,
    )
    return ExplainResponse(
        listing_id=listing_id,
        predicted_price=float(np.exp(result["predicted_value"])),
        base_value=float(np.exp(result["base_value"])),
        drivers=[Driver(feature=d["feature"], impact=float(d["impact"])) for d in result["drivers"]],
    )
