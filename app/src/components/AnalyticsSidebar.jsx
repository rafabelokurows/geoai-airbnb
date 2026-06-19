import { useMemo, useEffect, useState } from 'react'
import { fetchGlobalExplain } from '../api/client'

const LAYER_META = {
  price: { label: 'Price Heatmap', key: 'avg_price', unit: '€', suffix: '/night' },
  revenue: { label: 'Revenue Heatmap', key: 'avg_revenue', unit: '€', suffix: '/yr' },
  opportunity: { label: 'Opportunity Map', key: null, unit: '', suffix: '' },
}

function pct(arr, p) {
  if (!arr.length) return 0
  const sorted = [...arr].sort((a, b) => a - b)
  return sorted[Math.floor((p / 100) * (sorted.length - 1))]
}

function mean(arr) {
  return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0
}

function fmt(n, decimals = 0) {
  return new Intl.NumberFormat('en-GB', { maximumFractionDigits: decimals }).format(n)
}

export default function AnalyticsSidebar({ hexData, activeLayer }) {
  const meta = LAYER_META[activeLayer]
  const [shapImportance, setShapImportance] = useState([])

  useEffect(() => {
    fetchGlobalExplain(10).then(setShapImportance).catch(console.error)
  }, [])

  const stats = useMemo(() => {
    if (!meta.key || !hexData.length) return null
    const vals = hexData.map(d => d[meta.key]).filter(Number.isFinite)
    if (!vals.length) return null
    return {
      mean: mean(vals),
      median: pct(vals, 50),
      p25: pct(vals, 25),
      p75: pct(vals, 75),
    }
  }, [hexData, meta.key])

  const totalListings = useMemo(() => (
    hexData.reduce((s, d) => s + (d.listing_count || 0), 0)
  ), [hexData])

  const maxImportance = shapImportance.length
    ? shapImportance[0].importance
    : 1

  return (
    <div className="analytics-sidebar">
      <div className="sidebar-header">{meta.label}</div>

      <div className="sidebar-section">
        <div className="sidebar-stat-row">
          <span className="sidebar-stat-label">Hex cells</span>
          <span className="sidebar-stat-value">{fmt(hexData.length)}</span>
        </div>
        <div className="sidebar-stat-row">
          <span className="sidebar-stat-label">Total listings</span>
          <span className="sidebar-stat-value">{fmt(totalListings)}</span>
        </div>
      </div>

      {stats && (
        <div className="sidebar-section">
          <div className="sidebar-section-title">Distribution</div>
          <div className="sidebar-stat-row">
            <span className="sidebar-stat-label">Mean</span>
            <span className="sidebar-stat-value">{meta.unit}{fmt(stats.mean)}</span>
          </div>
          <div className="sidebar-stat-row">
            <span className="sidebar-stat-label">Median</span>
            <span className="sidebar-stat-value">{meta.unit}{fmt(stats.median)}</span>
          </div>
          <div className="sidebar-stat-row">
            <span className="sidebar-stat-label">P25 – P75</span>
            <span className="sidebar-stat-value">
              {meta.unit}{fmt(stats.p25)} – {meta.unit}{fmt(stats.p75)}
            </span>
          </div>
        </div>
      )}

      {activeLayer === 'opportunity' && (
        <div className="sidebar-section">
          <p className="sidebar-hint">
            Orange dots = underpriced listings (&gt;15% below predicted). Click a dot to see SHAP drivers.
          </p>
        </div>
      )}

      {shapImportance.length > 0 && (
        <div className="sidebar-section sidebar-shap">
          <div className="sidebar-section-title">SHAP · Top Price Drivers</div>
          {shapImportance.map(({ feature, importance }) => {
            const pct = maxImportance > 0 ? (importance / maxImportance) * 100 : 0
            return (
              <div key={feature} className="shap-row">
                <div className="shap-label-row">
                  <span className="shap-feat">{feature.replace(/_/g, ' ')}</span>
                  <span className="shap-val">+{importance.toFixed(3)}</span>
                </div>
                <div className="shap-bar-bg">
                  <div className="shap-bar-fill" style={{ width: `${pct}%` }} />
                </div>
              </div>
            )
          })}
          <p className="sidebar-hint" style={{ marginTop: 8 }}>
            Mean |SHAP| across 500 listings
          </p>
        </div>
      )}
    </div>
  )
}
