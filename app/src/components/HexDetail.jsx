import { useEffect, useState, useMemo } from 'react'
import { fetchHexListings, fetchHexPriceGap } from '../api/client'

function fmt(n, decimals = 0) {
  if (n == null || !Number.isFinite(n)) return '—'
  return new Intl.NumberFormat('en-GB', { maximumFractionDigits: decimals }).format(n)
}

function mean(arr) {
  return arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0
}

function hexNarrative(hex, cityMedianRevenue) {
  const rev = hex.avg_revenue
  const occ = (hex.avg_occupancy ?? 0) * 100
  const score = hex.opportunity_score ?? 0

  if (!Number.isFinite(rev)) return 'Insufficient data for this hex cell.'

  const pctDiff = ((rev - cityMedianRevenue) / cityMedianRevenue) * 100
  const direction = pctDiff >= 0 ? 'above' : 'below'
  const absPct = Math.abs(pctDiff).toFixed(0)
  const zone = score > 0.6
    ? 'Strong opportunity zone.'
    : score > 0.4
    ? 'Moderate opportunity.'
    : 'Competitive or low-yield area.'

  return `This area has ${hex.listing_count} listings earning ${absPct}% ${direction} the city median. Occupancy averages ${occ.toFixed(0)}%. ${zone}`
}

function ScatterPlot({ listings }) {
  const W = 220, H = 100, PAD = 16

  const valid = listings.filter(
    l => l.price != null && Number.isFinite(l.price) &&
         l.predicted_occupancy != null && Number.isFinite(l.predicted_occupancy)
  )

  if (!valid.length) return <p style={{ fontSize: 11, color: '#475569' }}>No listing data.</p>

  const prices = valid.map(l => l.price)
  const occs = valid.map(l => l.predicted_occupancy)
  const minP = Math.min(...prices), maxP = Math.max(...prices)
  const minO = Math.min(...occs), maxO = Math.max(...occs)

  const cx = (price) => PAD + ((price - minP) / (maxP - minP || 1)) * (W - PAD * 2)
  const cy = (occ) => H - PAD - ((occ - minO) / (maxO - minO || 1)) * (H - PAD * 2)

  return (
    <svg width={W} height={H} style={{ display: 'block', marginTop: 6 }}>
      <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#334155" strokeWidth={1} />
      <line x1={PAD} y1={PAD} x2={PAD} y2={H - PAD} stroke="#334155" strokeWidth={1} />
      <text x={PAD} y={H - 2} fontSize={9} fill="#475569">price →</text>
      <text x={2} y={PAD + 4} fontSize={9} fill="#475569" writingMode="vertical-rl">occ</text>
      {valid.map((l, i) => (
        <circle
          key={i}
          cx={cx(l.price)}
          cy={cy(l.predicted_occupancy)}
          r={3}
          fill="#38bdf8"
          fillOpacity={0.6}
        />
      ))}
    </svg>
  )
}

function PriceGapRow({ listing, direction }) {
  const gapPct = Math.abs(listing.price_gap_pct * 100).toFixed(0)
  const sign = direction === 'underpriced' ? '-' : '+'
  const color = direction === 'underpriced' ? '#22c55e' : '#ef4444'
  return (
    <div style={{ padding: '7px 0', borderBottom: '1px solid #1e2030', fontSize: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
        <span style={{ color: '#cbd5e1' }}>
          €{fmt(listing.actual_price)}/night
          <span style={{ color: '#64748b', margin: '0 4px' }}>→</span>
          <span style={{ color: '#94a3b8' }}>€{fmt(listing.predicted_price)} pred.</span>
        </span>
        <span style={{ color, fontWeight: 600 }}>{sign}{gapPct}%</span>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 10, color: '#475569' }}>
          {listing.room_type ?? '—'}
        </span>
        <a
          href={`https://www.airbnb.com/rooms/${listing.listing_id}`}
          target="_blank"
          rel="noopener noreferrer"
          style={{ fontSize: 10, color: '#38bdf8', textDecoration: 'none' }}
        >
          View listing ↗
        </a>
      </div>
    </div>
  )
}

function PriceGapTab({ priceGap }) {
  if (!priceGap) return <p style={{ fontSize: 12, color: '#64748b' }}>Loading…</p>

  const byGap = (a, b) => Math.abs(b.price_gap_pct) - Math.abs(a.price_gap_pct)
  const underpriced = [...priceGap.underpriced].sort(byGap)
  const overpriced = [...priceGap.overpriced].sort(byGap)
  if (!underpriced.length && !overpriced.length) {
    return <p style={{ fontSize: 12, color: '#64748b' }}>No significantly mispriced listings in this hex.</p>
  }

  return (
    <div>
      {underpriced.length > 0 && (
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, color: '#22c55e', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>
            Underpriced ({underpriced.length})
          </div>
          {underpriced.map(l => (
            <PriceGapRow key={l.listing_id} listing={l} direction="underpriced" />
          ))}
        </div>
      )}
      {overpriced.length > 0 && (
        <div>
          <div style={{ fontSize: 11, color: '#ef4444', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>
            Overpriced ({overpriced.length})
          </div>
          {overpriced.map(l => (
            <PriceGapRow key={l.listing_id} listing={l} direction="overpriced" />
          ))}
        </div>
      )}
    </div>
  )
}

export default function HexDetail({ hex, hexData, shapImportance, onClose }) {
  const [hexListings, setHexListings] = useState([])
  const [priceGap, setPriceGap] = useState(null)
  const [activeTab, setActiveTab] = useState('overview')

  useEffect(() => {
    if (!hex?.h3_cell) return
    setHexListings([])
    setPriceGap(null)
    setActiveTab('overview')
    fetchHexListings(hex.h3_cell).then(setHexListings).catch(console.error)
    fetchHexPriceGap(hex.h3_cell).then(setPriceGap).catch(console.error)
  }, [hex?.h3_cell])

  const cityMedianRevenue = useMemo(() => {
    if (!hexData?.length) return 0
    const revs = hexData.map(d => d.avg_revenue).filter(Number.isFinite).sort((a, b) => a - b)
    return revs.length ? revs[Math.floor(revs.length / 2)] : 0
  }, [hexData])

  const cityAvgPrice = useMemo(() => {
    if (!hexData?.length) return 100
    const prices = hexData.map(d => d.avg_price).filter(Number.isFinite)
    return prices.length ? mean(prices) : 100
  }, [hexData])

  const top10Shap = useMemo(() => shapImportance.slice(0, 10), [shapImportance])
  const maxImportance = top10Shap.length ? top10Shap[0].importance : 1

  if (!hex) return null

  const narrative = hexNarrative(hex, cityMedianRevenue)
  const score = hex.opportunity_score ?? 0

  const tabStyle = (tab) => ({
    flex: 1,
    padding: '5px 0',
    fontSize: 11,
    fontWeight: 500,
    background: 'none',
    border: 'none',
    borderBottom: activeTab === tab ? '2px solid #38bdf8' : '2px solid transparent',
    color: activeTab === tab ? '#e2e8f0' : '#64748b',
    cursor: 'pointer',
    textTransform: 'uppercase',
    letterSpacing: '0.06em',
  })

  return (
    <div className="detail-panel">
      <button className="close-btn" onClick={onClose}>×</button>
      <h3>Hex Cell</h3>
      <p className="subtitle" style={{ fontFamily: 'monospace', fontSize: 11 }}>
        {hex.h3_cell}
      </p>

      <div style={{ display: 'flex', marginTop: 10, marginBottom: 12, borderBottom: '1px solid #1e2030' }}>
        <button style={tabStyle('overview')} onClick={() => setActiveTab('overview')}>Overview</button>
        <button style={tabStyle('pricegap')} onClick={() => setActiveTab('pricegap')}>
          Price Gap{priceGap ? ` (${priceGap.underpriced.length + priceGap.overpriced.length})` : ''}
        </button>
      </div>

      {activeTab === 'overview' && (
        <>
          <p style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.5, margin: '0 0 12px' }}>
            {narrative}
          </p>

          <div className="hex-stat-grid">
            <div className="hex-stat">
              <span className="hex-stat-label">Listings</span>
              <span className="hex-stat-value">{fmt(hex.listing_count)}</span>
            </div>
            <div className="hex-stat">
              <span className="hex-stat-label">Avg price/night</span>
              <span className="hex-stat-value">€{fmt(hex.avg_price)}</span>
            </div>
            <div className="hex-stat">
              <span className="hex-stat-label">Avg occupancy</span>
              <span className="hex-stat-value">{fmt((hex.avg_occupancy ?? 0) * 100, 1)}%</span>
            </div>
            <div className="hex-stat">
              <span className="hex-stat-label">Avg revenue/yr</span>
              <span className="hex-stat-value">€{fmt(hex.avg_revenue)}</span>
            </div>
            <div className="hex-stat">
              <span className="hex-stat-label">Opportunity score</span>
              <span className="hex-stat-value" style={{ color: score > 0.6 ? '#34d399' : '#94a3b8' }}>
                {(score * 100).toFixed(0)}/100
              </span>
            </div>
          </div>

          {hexListings.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 2 }}>Price vs Occupancy</div>
              <ScatterPlot listings={hexListings} />
            </div>
          )}

          {top10Shap.length > 0 && (
            <div style={{ marginTop: 14 }}>
              <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 6 }}>Top Price Drivers (SHAP)</div>
              {top10Shap.map(({ feature, importance }) => {
                const barPct = maxImportance > 0 ? (importance / maxImportance) * 100 : 0
                const eurImpact = cityAvgPrice * (Math.exp(importance) - 1)
                return (
                  <div key={feature} className="shap-row">
                    <div className="shap-label-row">
                      <span className="shap-feat" style={{ fontSize: 10 }}>{feature.replace(/_/g, ' ')}</span>
                      <span className="shap-val" style={{ fontSize: 10 }}>+€{fmt(eurImpact, 0)}</span>
                    </div>
                    <div className="shap-bar-bg">
                      <div className="shap-bar-fill" style={{ width: `${barPct}%` }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}

          <p style={{ marginTop: 16, fontSize: 11, color: '#475569' }}>
            Click a listing dot on the Opportunity Map to see SHAP price drivers.
          </p>
        </>
      )}

      {activeTab === 'pricegap' && (
        <PriceGapTab priceGap={priceGap} />
      )}
    </div>
  )
}
