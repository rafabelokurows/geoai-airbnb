import { useEffect, useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, ReferenceLine
} from 'recharts'
import { fetchExplain } from '../api/client'

export default function ListingDetail({ listingId, onClose }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!listingId) return
    setData(null)
    setError(null)
    fetchExplain(listingId)
      .then(setData)
      .catch(e => setError(e.message))
  }, [listingId])

  if (!listingId) return null

  return (
    <div className="detail-panel">
      <button className="close-btn" onClick={onClose}>×</button>
      <h3>Listing #{listingId}</h3>

      {error && <p style={{ color: '#f87171', fontSize: 13 }}>{error}</p>}

      {!data && !error && (
        <p style={{ color: '#64748b', fontSize: 13 }}>Loading explanation…</p>
      )}

      {data && (
        <>
          <p className="subtitle">
            Predicted price: <strong>€{data.predicted_price.toFixed(0)}/night</strong>
          </p>

          <p style={{ fontSize: 12, color: '#64748b', marginBottom: 12 }}>
            Top 5 price drivers (SHAP values in log-price units)
          </p>

          <ResponsiveContainer width="100%" height={220}>
            <BarChart
              layout="vertical"
              data={data.drivers}
              margin={{ top: 0, right: 16, left: 0, bottom: 0 }}
            >
              <XAxis type="number" tick={{ fontSize: 10, fill: '#94a3b8' }} />
              <YAxis
                type="category"
                dataKey="feature"
                width={160}
                tick={{ fontSize: 10, fill: '#94a3b8' }}
                tickFormatter={f => f.replace(/_/g, ' ')}
              />
              <Tooltip
                formatter={v => v.toFixed(4)}
                contentStyle={{ background: '#1a1d27', border: '1px solid #2d3148', fontSize: 12 }}
              />
              <ReferenceLine x={0} stroke="#3d4258" />
              <Bar dataKey="impact" radius={[0, 3, 3, 0]}>
                {data.drivers.map((d, i) => (
                  <Cell key={i} fill={d.impact >= 0 ? '#22c55e' : '#ef4444'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>

          <div style={{ marginTop: 16, fontSize: 12, color: '#64748b' }}>
            Base value: €{data.base_value.toFixed(0)}/night (market average)
          </div>
        </>
      )}
    </div>
  )
}
