import numpy as np
from fastapi import APIRouter, Depends, HTTPException

from geoai.api.deps import AppState, get_state
from geoai.api.schemas import Driver, ExplainResponse
from geoai.explainability.shap_explainer import explain_listing

router = APIRouter()


@router.get("/listings/{listing_id}/explain", response_model=ExplainResponse)
def explain(listing_id: int, state: AppState = Depends(get_state)) -> ExplainResponse:
    idx = state.listing_id_to_idx.get(listing_id)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"Listing {listing_id} not found or has no predictions")
    result = explain_listing(
        state.price_explainer,
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
