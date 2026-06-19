import { useState, useCallback, useEffect } from 'react'
import KpiBar from './components/KpiBar'
import MapView from './components/MapView'
import ListingDetail from './components/ListingDetail'
import HexDetail from './components/HexDetail'
import AnalyticsSidebar from './components/AnalyticsSidebar'
import { fetchHexAggregates } from './api/client'

export default function App() {
  const [hexData, setHexData] = useState([])
  const [activeLayer, setActiveLayer] = useState('price')
  const [selectedOpportunity, setSelectedOpportunity] = useState(null)
  const [selectedHex, setSelectedHex] = useState(null)

  useEffect(() => {
    fetchHexAggregates().then(setHexData).catch(console.error)
  }, [])

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
        <AnalyticsSidebar hexData={hexData} activeLayer={activeLayer} />
        <div className="map-area">
          <MapView
            hexData={hexData}
            activeLayer={activeLayer}
            onLayerChange={handleLayerChange}
            onHexClick={handleHexClick}
            onListingClick={handleListingClick}
          />
          {selectedHex && <HexDetail hex={selectedHex} onClose={handleClose} />}
          {selectedOpportunity && (
            <ListingDetail opportunity={selectedOpportunity} onClose={handleClose} />
          )}
        </div>
      </div>
    </>
  )
}
