import { useState } from 'react'
import ControlPanel from './components/ControlPanel'
import MapPanel from './components/MapPanel'

const MIN_W = 320
const MAX_W = 680

export default function App() {
  // Resizable control panel: drag the divider to widen or narrow it.
  const [width, setWidth] = useState(400)

  const startResize = (e: React.PointerEvent) => {
    e.preventDefault()
    const startX = e.clientX
    const startW = width
    const onMove = (ev: PointerEvent) => {
      setWidth(Math.min(MAX_W, Math.max(MIN_W, startW + (ev.clientX - startX))))
    }
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  return (
    <div className="flex h-full w-full">
      <div style={{ width }} className="h-full shrink-0 border-r border-zinc-200">
        <ControlPanel />
      </div>
      <div
        role="separator"
        aria-orientation="vertical"
        onPointerDown={startResize}
        title="Drag to resize"
        className="group relative z-10 -ml-1 w-2 shrink-0 cursor-col-resize"
      >
        <div className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-transparent transition-colors group-hover:bg-accent" />
      </div>
      <main className="relative h-full flex-1">
        <MapPanel />
      </main>
    </div>
  )
}
