import { useEffect, useState, useCallback, useMemo } from 'react'
import DeckGL from '@deck.gl/react'
import { PolygonLayer, ScatterplotLayer } from '@deck.gl/layers'
import { cellToBoundary } from 'h3-js'
import Map from 'react-map-gl/maplibre'
import { fetchOpportunities } from '../api/client'
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

function percentile(arr, p) {
  const sorted = [...arr].sort((a, b) => a - b)
  return sorted[Math.floor((p / 100) * (sorted.length - 1))]
}

function hexColor(value, min, max, colorMode) {
  const t = max === min ? 0 : Math.min(1, Math.max(0, (value - min) / (max - min)))
  const alpha = 180
  if (colorMode === 'price') return [...lerpColor(t, [68, 1, 84], [253, 231, 37]), alpha]
  if (colorMode === 'revenue') return [...lerpColor(t, [0, 63, 92], [255, 166, 0]), alpha]
  return [...lerpColor(t, [254, 240, 217], [215, 48, 31]), alpha]
}

export default function MapView({ hexData, activeLayer, onLayerChange, onHexClick, onListingClick }) {
  const [opportunities, setOpportunities] = useState([])
  const [viewState, setViewState] = useState(INITIAL_VIEW)

  useEffect(() => {
    fetchOpportunities(200).then(setOpportunities).catch(console.error)
  }, [])

  const polygonData = useMemo(() => (
    hexData
      .map(d => {
        try {
          return { ...d, polygon: cellToBoundary(d.h3_cell, true) }
        } catch {
          return null
        }
      })
      .filter(d => d?.polygon?.length)
  ), [hexData])

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
          onClick: ({ object }) => object && onListingClick(object),
          radiusMinPixels: 4,
          radiusMaxPixels: 20,
        }),
      ]
    }

    if (!polygonData.length) return []

    const valueKey = activeLayer === 'price' ? 'avg_price' : 'avg_revenue'
    const vals = polygonData.map(d => d[valueKey]).filter(Number.isFinite)
    if (!vals.length) return []
    const min = percentile(vals, 5)
    const max = percentile(vals, 95)

    return [
      new PolygonLayer({
        id: `hex-${activeLayer}`,
        data: polygonData,
        getPolygon: d => d.polygon,
        getFillColor: d => hexColor(d[valueKey] ?? min, min, max, activeLayer),
        getElevation: 0,
        extruded: false,
        filled: true,
        stroked: false,
        pickable: true,
        highPrecision: false,
        coverage: 0.92,
        onClick: ({ object }) => object && onHexClick(object),
      }),
    ]
  }, [activeLayer, polygonData, opportunities, onHexClick, onListingClick])

  return (
    <div className="map-container">
      <LayerControls active={activeLayer} onChange={onLayerChange} />
      <DeckGL
        style={{ position: 'absolute', inset: 0 }}
        viewState={viewState}
        onViewStateChange={({ viewState: vs }) => setViewState(vs)}
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
          style={{ width: '100%', height: '100%' }}
          mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        />
      </DeckGL>
    </div>
  )
}
