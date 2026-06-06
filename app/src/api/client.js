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
  if (!r.ok) throw new Error(`Failed to explain listing ${listingId}`)
  return r.json()
}
