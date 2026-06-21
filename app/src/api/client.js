const BASE = '/api'

export async function fetchKpis() {
  const r = await fetch(`${BASE}/kpis`)
  if (!r.ok) throw new Error('Failed to fetch KPIs')
  return r.json()
}

export async function fetchHexAggregates() {
  const r = await fetch(`${BASE}/hex-aggregates`)
  if (!r.ok) throw new Error('Failed to fetch hex aggregates')
  return r.json()
}

export async function fetchListings(limit = 5000) {
  const r = await fetch(`${BASE}/listings?limit=${limit}`)
  if (!r.ok) throw new Error('Failed to fetch listings')
  return r.json()
}

export async function fetchOpportunities(topN = 100) {
  const r = await fetch(`${BASE}/opportunities?top_n=${topN}`)
  if (!r.ok) throw new Error('Failed to fetch opportunities')
  return r.json()
}

export async function fetchExplain(listingId) {
  const r = await fetch(`${BASE}/listings/${listingId}/explain`)
  if (r.status === 404) throw new Error('SHAP explanation not available for this listing')
  if (!r.ok) throw new Error(`Failed to explain listing ${listingId}`)
  return r.json()
}

export async function fetchGlobalExplain(topN = 10) {
  const r = await fetch(`${BASE}/explain/global?top_n=${topN}`)
  if (!r.ok) throw new Error('Failed to fetch global SHAP importance')
  return r.json()
}

export async function fetchNeighbourhoods() {
  const r = await fetch(`${BASE}/neighbourhoods`)
  if (!r.ok) throw new Error('Failed to fetch neighbourhoods')
  return r.json()
}

export async function fetchHexListings(h3Cell) {
  const r = await fetch(`${BASE}/hex/${encodeURIComponent(h3Cell)}/listings`)
  if (!r.ok) throw new Error(`Failed to fetch listings for hex ${h3Cell}`)
  return r.json()
}

export async function fetchPriceGap(topN = 50) {
  const r = await fetch(`${BASE}/price-gap?top_n=${topN}`)
  if (!r.ok) throw new Error('Failed to fetch price gap data')
  return r.json()
}

export async function fetchHexPriceGap(h3Cell) {
  const r = await fetch(`${BASE}/hex/${encodeURIComponent(h3Cell)}/price-gap`)
  if (!r.ok) throw new Error(`Failed to fetch price gap for hex ${h3Cell}`)
  return r.json()
}
