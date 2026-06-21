import { useMemo } from 'react'

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

export default function AnalyticsSidebar({ hexData, activeLayer, shapImportance, priceGapData }) {
  const meta = LAYER_META[activeLayer]

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

  const cityAvgPrice = useMemo(() => {
    const prices = hexData.map(d => d.avg_price).filter(Number.isFinite)
    return prices.length ? mean(prices) : 100
  }, [hexData])

  const maxImportance = shapImportance.length ? shapImportance[0].importance : 1

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

      {priceGapData && (
        <div className="sidebar-section">
          <div className="sidebar-section-title">Price Gap Analysis</div>
          <div className="sidebar-stat-row">
            <span className="sidebar-stat-label" style={{ color: '#22c55e' }}>Underpriced</span>
            <span className="sidebar-stat-value">{fmt(priceGapData.underpriced.length)} listings</span>
          </div>
          <div className="sidebar-stat-row">
            <span className="sidebar-stat-label" style={{ color: '#ef4444' }}>Overpriced</span>
            <span className="sidebar-stat-value">{fmt(priceGapData.overpriced.length)} listings</span>
          </div>
          {priceGapData.segment_summary.filter(s => s.segment_type === 'room_type').length > 0 && (
            <>
              <div className="sidebar-section-title" style={{ marginTop: 10 }}>By Room Type</div>
              <table style={{ width: '100%', fontSize: 11, borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ color: '#64748b' }}>
                    <th style={{ textAlign: 'left', paddingBottom: 4, fontWeight: 500 }}>Type</th>
                    <th style={{ textAlign: 'right', paddingBottom: 4, fontWeight: 500 }}>Under</th>
                    <th style={{ textAlign: 'right', paddingBottom: 4, fontWeight: 500 }}>Over</th>
                    <th style={{ textAlign: 'right', paddingBottom: 4, fontWeight: 500 }}>Med gap</th>
                  </tr>
                </thead>
                <tbody>
                  {priceGapData.segment_summary
                    .filter(s => s.segment_type === 'room_type')
                    .map(s => (
                      <tr key={s.segment_value} style={{ borderTop: '1px solid #1e2030' }}>
                        <td style={{ paddingTop: 4, paddingBottom: 4, color: '#cbd5e1', maxWidth: 90, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {s.segment_value}
                        </td>
                        <td style={{ textAlign: 'right', color: '#22c55e' }}>{s.underpriced_count}</td>
                        <td style={{ textAlign: 'right', color: '#ef4444' }}>{s.overpriced_count}</td>
                        <td style={{ textAlign: 'right', color: '#94a3b8' }}>
                          {(s.median_gap_pct * 100).toFixed(0)}%
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      )}

      {shapImportance.length > 0 && (
        <div className="sidebar-section sidebar-shap">
          <div className="sidebar-section-title">Top Price Drivers (SHAP)</div>
          {shapImportance.map(({ feature, importance }) => {
            const barPct = maxImportance > 0 ? (importance / maxImportance) * 100 : 0
            const eurImpact = cityAvgPrice * (Math.exp(importance) - 1)
            return (
              <div key={feature} className="shap-row">
                <div className="shap-label-row">
                  <span className="shap-feat">{feature.replace(/_/g, ' ')}</span>
                  <span className="shap-val">+€{fmt(eurImpact, 0)}/night</span>
                </div>
                <div className="shap-bar-bg">
                  <div className="shap-bar-fill" style={{ width: `${barPct}%` }} />
                </div>
              </div>
            )
          })}
          <p className="sidebar-hint" style={{ marginTop: 8 }}>
            Mean |SHAP| across 500 listings · EUR impact approx.
          </p>
        </div>
      )}

    </div>
  )
}
