from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class KpiResponse(BaseModel):
    listing_count: int
    listings_with_predictions: int
    avg_price: float
    avg_occupancy: float
    median_annual_revenue: float


class HexCell(BaseModel):
    h3_cell: str
    listing_count: int
    avg_price: float
    avg_occupancy: float
    avg_revenue: float


class ListingPoint(BaseModel):
    id: int
    latitude: float
    longitude: float
    price: float | None
    room_type: str | None
    predicted_price: float | None
    predicted_occupancy: float | None
    estimated_annual_revenue: float | None


class ListingsResponse(BaseModel):
    listings: list[ListingPoint]
    total: int


class OpportunityListing(BaseModel):
    listing_id: str
    latitude: float
    longitude: float
    actual_price: float
    predicted_price: float
    opportunity_gap: float
    estimated_uplift_annual: float


class Driver(BaseModel):
    feature: str
    impact: float


class ExplainResponse(BaseModel):
    listing_id: int
    predicted_price: float
    base_value: float
    drivers: list[Driver]


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class NeighbourhoodRank(BaseModel):
    neighbourhood: str
    listing_count: int
    avg_revenue: float


class HexListing(BaseModel):
    price: float | None
    predicted_occupancy: float | None
