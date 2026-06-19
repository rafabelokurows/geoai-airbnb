import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { fetchExplain } from '../api/client'

function fmt(n, decimals = 0) {
  if (n == null || !Number.isFinite(Number(n))) return '—'
  return new Intl.NumberFormat('en-GB', { maximumFractionDigits: decimals }).format(Number(n))
}

export default function ListingDetail({ opportunity, onClose }) {
  const [shap, setShap] = useState({ id: null, data: null, error: null })

  useEffect(() => {
    if (!opportunity) return
    let cancelled = false
    setShap({ id: opportunity.listing_id, data: null, error: null })

    fetchExplain(opportunity.listing_id)
      .then(data => { if (!cancelled) setShap({ id: opportunity.listing_id, data, error: null }) })
      .catch(e => { if (!cancelled) setShap({ id: opportunity.listing_id, data: null, error: e.message }) })

    return () => { cancelled = true }
  }, [opportunity?.listing_id])

  if (!opportunity) return null

  const shapReady = shap.id === opportunity.listing_id
  const shapData = shapReady ? shap.data : null
  const shapError = shapReady ? shap.error : null

  return (
    <div className="detail-panel">
      <button className="close-btn" onClick={onClose}>×</button>
      <h3>Opportunity</h3>
      <p className="subtitle" style={{ fontFamily: 'monospace', fontSize: 10 }}>
        ID {opportunity.listing_id}
      </p>

      <div className="hex-stat-grid" style={{ marginTop: 12 }}>
        <div className="hex-stat">
          <span className="hex-stat-label">Actual price</span>
          <span className="hex-stat-value">€{fmt(opportunity.actual_price)}</span>
        </div>
        <div className="hex-stat">
          <span className="hex-stat-label">Predicted price</span>
          <span className="hex-stat-value" style={{ color: '#22c55e' }}>
            €{fmt(opportunity.predicted_price)}
          </span>
        </div>
        <div className="hex-stat">
          <span className="hex-stat-label">Gap/night</span>
          <span className="hex-stat-value" style={{ color: '#fbbf24' }}>
            +€{fmt(opportunity.opportunity_gap)}
          </span>
        </div>
        <div className="hex-stat">
          <span className="hex-stat-label">Annual uplift</span>
          <span className="hex-stat-value" style={{ color: '#fbbf24' }}>
            €{fmt(opportunity.estimated_uplift_annual)}
          </span>
        </div>
      </div>

      <div style={{ marginTop: 20, marginBottom: 8, fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
        SHAP · Price Drivers
      </div>

      {!shapData && !shapError && (
        <p style={{ color: '#64748b', fontSize: 12 }}>Loading explanation…</p>
      )}

      {shapError && (
        <p style={{ color: '#94a3b8', fontSize: 12 }}>{shapError}</p>
      )}

      {shapData && (
        <>
          <p style={{ fontSize: 11, color: '#64748b', marginBottom: 10 }}>
            Base €{fmt(shapData.base_value)} → predicted €{fmt(shapData.predicted_price)}/night
          </p>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart
              layout="vertical"
              data={shapData.drivers}
              margin={{ top: 0, right: 16, left: 0, bottom: 0 }}
            >
              <XAxis type="number" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <YAxis
                type="category"
                dataKey="feature"
                width={155}
                tick={{ fontSize: 10, fill: '#94a3b8' }}
                tickFormatter={f => f.replace(/_/g, ' ')}
              />
              <Tooltip
                formatter={v => v.toFixed(4)}
                contentStyle={{ background: '#1a1d27', border: '1px solid #2d3148', fontSize: 11 }}
              />
              <ReferenceLine x={0} stroke="#3d4258" />
              <Bar dataKey="impact" radius={[0, 3, 3, 0]}>
                {shapData.drivers.map((d, i) => (
                  <Cell key={i} fill={d.impact >= 0 ? '#22c55e' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  )
}
