import { useEffect, useState, useCallback } from 'react'
import DeckGL from '@deck.gl/react'
import { H3HexagonLayer } from '@deck.gl/geo-layers'
import { ScatterplotLayer } from '@deck.gl/layers'
import Map from 'react-map-gl/maplibre'
import { fetchHexAggregates, fetchOpportunities } from '../api/client'
import LayerControls from './LayerControls'

const INITIAL_VIEW = {
  longitude: -8.6099,
  latitude: 41.1496,
  zoom: 12,
  pitch: 0,
  bearing: 0,
}

function lerpColor(t, low, high) {
  return low.map((c, i) => Math.round(c + t * (high[i] - c)))
}

function hexColor(value, min, max, colorMode) {
  const t = max === min ? 0 : Math.min(1, Math.max(0, (value - min) / (max - min)))
  const alpha = 180
  if (colorMode === 'price') return [...lerpColor(t, [68, 1, 84], [253, 231, 37]), alpha]
  if (colorMode === 'revenue') return [...lerpColor(t, [0, 63, 92], [255, 166, 0]), alpha]
  return [...lerpColor(t, [254, 240, 217], [215, 48, 31]), alpha]
}

export default function MapView({ onListingClick }) {
  const [hexData, setHexData] = useState([])
  const [opportunities, setOpportunities] = useState([])
  const [activeLayer, setActiveLayer] = useState('price')

  useEffect(() => {
    fetchHexAggregates().then(setHexData).catch(console.error)
    fetchOpportunities(200).then(setOpportunities).catch(console.error)
  }, [])

  const buildLayers = useCallback(() => {
    if (activeLayer === 'opportunity') {
      return [
        new ScatterplotLayer({
          id: 'opportunity-scatter',
          data: opportunities,
          getPosition: d => [d.longitude, d.latitude],
          getRadius: 60,
          getFillColor: d => {
            const maxGap = Math.max(...opportunities.map(o => o.opportunity_gap), 1)
            const t = d.opportunity_gap / maxGap
            return [...lerpColor(t, [255, 200, 0], [200, 0, 0]), 200]
          },
          pickable: true,
          onClick: ({ object }) => object && onListingClick(object.listing_id),
          radiusMinPixels: 4,
          radiusMaxPixels: 20,
        }),
      ]
    }

    if (!hexData.length) return []

    const valueKey = activeLayer === 'price' ? 'avg_price' : 'avg_revenue'
    const vals = hexData.map(d => d[valueKey]).filter(Boolean)
    const min = Math.min(...vals)
    const max = Math.max(...vals)

    return [
      new H3HexagonLayer({
        id: `hex-${activeLayer}`,
        data: hexData,
        getHexagon: d => d.h3_cell,
        getFillColor: d => hexColor(d[valueKey] ?? min, min, max, activeLayer),
        getElevation: 0,
        extruded: false,
        filled: true,
        stroked: false,
        pickable: true,
        highPrecision: false,
        coverage: 0.92,
        onClick: ({ object }) => object && onListingClick(null),
      }),
    ]
  }, [activeLayer, hexData, opportunities, onListingClick])

  return (
    <div className="map-container">
      <LayerControls active={activeLayer} onChange={setActiveLayer} />
      <DeckGL
        initialViewState={INITIAL_VIEW}
        controller={true}
        layers={buildLayers()}
        getTooltip={({ object }) => {
          if (!object) return null
          if (activeLayer === 'opportunity') {
            return `Gap: €${object.opportunity_gap?.toFixed(0)}/night\nUplift: €${object.estimated_uplift_annual?.toFixed(0)}/year`
          }
          return `Listings: ${object.listing_count}\nAvg Price: €${object.avg_price?.toFixed(0)}\nAvg Revenue: €${object.avg_revenue?.toFixed(0)}/yr`
        }}
      >
        <Map
          mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        />
      </DeckGL>
    </div>
  )
}
