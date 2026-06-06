const LAYERS = [
  { id: 'price', label: 'Price Heatmap' },
  { id: 'revenue', label: 'Revenue Heatmap' },
  { id: 'opportunity', label: 'Opportunity Map' },
]

export default function LayerControls({ active, onChange }) {
  return (
    <div className="layer-controls">
      {LAYERS.map(l => (
        <button
          key={l.id}
          className={`layer-btn${active === l.id ? ' active' : ''}`}
          onClick={() => onChange(l.id)}
        >
          {l.label}
        </button>
      ))}
    </div>
  )
}
