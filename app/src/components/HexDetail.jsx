function fmt(n, decimals = 0) {
  if (n == null || !Number.isFinite(n)) return '—'
  return new Intl.NumberFormat('en-GB', { maximumFractionDigits: decimals }).format(n)
}

export default function HexDetail({ hex, onClose }) {
  if (!hex) return null

  return (
    <div className="detail-panel">
      <button className="close-btn" onClick={onClose}>×</button>
      <h3>Hex Cell</h3>
      <p className="subtitle" style={{ fontFamily: 'monospace', fontSize: 11 }}>
        {hex.h3_cell}
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
      </div>

      <p style={{ marginTop: 20, fontSize: 11, color: '#475569' }}>
        Click a listing dot on the Opportunity Map to see SHAP price drivers.
      </p>
    </div>
  )
}
