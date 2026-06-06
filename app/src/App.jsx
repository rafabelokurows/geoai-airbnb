import { useState, useCallback } from 'react'
import KpiBar from './components/KpiBar'
import MapView from './components/MapView'
import ListingDetail from './components/ListingDetail'

export default function App() {
  const [selectedListingId, setSelectedListingId] = useState(null)

  const handleListingClick = useCallback((id) => {
    setSelectedListingId(id)
  }, [])

  const handleClose = useCallback(() => {
    setSelectedListingId(null)
  }, [])

  return (
    <>
      <KpiBar />
      <div style={{ flex: 1, position: 'relative' }}>
        <MapView onListingClick={handleListingClick} />
        <ListingDetail listingId={selectedListingId} onClose={handleClose} />
      </div>
    </>
  )
}
