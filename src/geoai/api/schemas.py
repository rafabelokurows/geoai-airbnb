from pydantic import BaseModel


class StatsResponse(BaseModel):
    avg_price: float
    avg_occupancy: float
    median_revenue: float
    listing_count: int


class HexSummary(BaseModel):
    hex_id: str
    value: float
    listing_count: int


class HexDetail(BaseModel):
    hex_id: str
    avg_price: float
    avg_occupancy: float
    avg_revenue: float
    listing_count: int
    avg_walkability_score: float
    avg_restaurant_density: float
    avg_dist_city_center_km: float
    avg_competition_score: float


class ListingPoint(BaseModel):
    id: int
    latitude: float
    longitude: float
    predicted_price: float
    predicted_occupancy: float


class ShapFeature(BaseModel):
    feature: str
    importance: float


class ShapDriver(BaseModel):
    feature: str
    avg_impact: float


class HexShapResponse(BaseModel):
    hex_id: str
    base_value: float
    drivers: list[ShapDriver]
