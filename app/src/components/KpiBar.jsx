import { useEffect, useState } from 'react'
import { fetchKpis } from '../api/client'

export default function KpiBar() {
  const [kpis, setKpis] = useState(null)

  useEffect(() => {
    fetchKpis().then(setKpis).catch(console.error)
  }, [])

  if (!kpis) return <div className="kpi-bar"><span style={{ color: '#64748b' }}>Loading…</span></div>

  const fmt = (n, decimals = 0) =>
    new Intl.NumberFormat('en-GB', { maximumFractionDigits: decimals }).format(n)

  return (
    <div className="kpi-bar">
      <div className="kpi-item">
        <span className="kpi-label">Listings</span>
        <span className="kpi-value">{fmt(kpis.listing_count)}</span>
      </div>
      <div className="kpi-item">
        <span className="kpi-label">With Predictions</span>
        <span className="kpi-value">{fmt(kpis.listings_with_predictions)}</span>
      </div>
      <div className="kpi-item">
        <span className="kpi-label">Avg Price/Night</span>
        <span className="kpi-value">€{fmt(kpis.avg_price, 0)}</span>
      </div>
      <div className="kpi-item">
        <span className="kpi-label">Avg Occupancy</span>
        <span className="kpi-value">{fmt(kpis.avg_occupancy * 100, 1)}%</span>
      </div>
      <div className="kpi-item">
        <span className="kpi-label">Median Rev/Year</span>
        <span className="kpi-value">€{fmt(kpis.median_annual_revenue)}</span>
      </div>
    </div>
  )
}
