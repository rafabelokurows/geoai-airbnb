import { useState, useCallback, useEffect, useMemo } from 'react'
import KpiBar from './components/KpiBar'
import MapView from './components/MapView'
import ListingDetail from './components/ListingDetail'
import HexDetail from './components/HexDetail'
import AnalyticsSidebar from './components/AnalyticsSidebar'
import { fetchHexAggregates, fetchGlobalExplain } from './api/client'

function normalize(val, min, max) {
  if (max === min) return 0
  return (val - min) / (max - min)
}

export default function App() {
  const [hexData, setHexData] = useState([])
  const [activeLayer, setActiveLayer] = useState('price')
  const [selectedOpportunity, setSelectedOpportunity] = useState(null)
  const [selectedHex, setSelectedHex] = useState(null)
  const [shapImportance, setShapImportance] = useState([])

  useEffect(() => {
    fetchHexAggregates().then(setHexData).catch(console.error)
    fetchGlobalExplain(15).then(setShapImportance).catch(console.error)
  }, [])

  const enrichedHexData = useMemo(() => {
    if (!hexData.length) return hexData
    const revenues = hexData.map(d => d.avg_revenue).filter(Number.isFinite)
    const counts = hexData.map(d => d.listing_count).filter(Number.isFinite)
    const minRev = Math.min(...revenues), maxRev = Math.max(...revenues)
    const minCount = Math.min(...counts), maxCount = Math.max(...counts)
    return hexData.map(d => ({
      ...d,
      opportunity_score:
        0.5 * normalize(d.avg_revenue, minRev, maxRev) +
        0.5 * (1 - normalize(d.listing_count, minCount, maxCount)),
    }))
  }, [hexData])

  const handleLayerChange = useCallback((layer) => {
    setActiveLayer(layer)
    setSelectedHex(null)
    setSelectedOpportunity(null)
  }, [])

  const handleHexClick = useCallback((hex) => {
    setSelectedHex(hex)
    setSelectedOpportunity(null)
  }, [])

  const handleListingClick = useCallback((opportunity) => {
    setSelectedOpportunity(opportunity)
    setSelectedHex(null)
  }, [])

  const handleClose = useCallback(() => {
    setSelectedOpportunity(null)
    setSelectedHex(null)
  }, [])

  return (
    <>
      <KpiBar />
      <div className="app-main">
        <AnalyticsSidebar
          hexData={enrichedHexData}
          activeLayer={activeLayer}
          shapImportance={shapImportance}
        />
        <div className="map-area">
          <MapView
            hexData={enrichedHexData}
            activeLayer={activeLayer}
            onLayerChange={handleLayerChange}
            onHexClick={handleHexClick}
            onListingClick={handleListingClick}
          />
          {selectedHex && (
            <HexDetail
              hex={selectedHex}
              hexData={enrichedHexData}
              shapImportance={shapImportance}
              onClose={handleClose}
            />
          )}
          {selectedOpportunity && (
            <ListingDetail opportunity={selectedOpportunity} onClose={handleClose} />
          )}
        </div>
      </div>
    </>
  )
}
