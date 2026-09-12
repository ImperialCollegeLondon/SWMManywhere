import { useEffect, useMemo, useRef, useState } from 'react'
import Map, {
  Layer,
  NavigationControl,
  ScaleControl,
  Source,
  type MapLayerMouseEvent,
  type MapRef,
} from 'react-map-gl/maplibre'
import type { FilterSpecification, LngLatBoundsLike } from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { Feature, FeatureCollection, Polygon } from 'geojson'
import { Eye, EyeOff, LocateFixed, X } from 'lucide-react'
import { useStore, type LayerKey } from '../store'
import type { Bbox } from '../types'
import { BTN_SECONDARY } from '../ui'
import TimeseriesChart from './TimeseriesChart'

// Keyless vector basemap (OpenFreeMap). Its glyph endpoint provides the font the
// flow-direction arrows need.
const MAP_STYLE = 'https://tiles.openfreemap.org/styles/positron'

const EMPTY: FeatureCollection = { type: 'FeatureCollection', features: [] }
const kindIs = (k: string): FilterSpecification => ['==', ['get', 'kind'], k] as FilterSpecification
const vis = (on: boolean) => ({ visibility: (on ? 'visible' : 'none') as 'visible' | 'none' })

// Outfalls red as in swmmanywhere.utilities.plot_map. Pipes are coloured and
// widened by diameter class, manholes are hollow circles so they never merge with
// the pipes, and the accent (the docs' primary colour) marks the bounding box and
// the hovered or selected element.
const ACCENT = '#00bcd4'
const NODE_STROKE = '#0b2a52'
const OUTFALL = '#e5484d'
const SUB_FILL = '#4c9a6a'
const SUB_LINE = '#2f6e4a'

// Diameter classes (m) shared by the paint expressions and the legend.
const DIAMETER_CLASSES = [
  { min: 0, label: 'under 0.3 m', color: '#9db3c8', width: 1.2 },
  { min: 0.3, label: '0.3 to 0.5 m', color: '#6f8fb0', width: 2.2 },
  { min: 0.5, label: '0.5 to 0.8 m', color: '#41689a', width: 3.4 },
  { min: 0.8, label: '0.8 to 1.2 m', color: '#1f4578', width: 5 },
  { min: 1.2, label: '1.2 m and over', color: NODE_STROKE, width: 7 },
]
const DIAMETER = ['coalesce', ['get', 'diameter'], 0.15]
const byClass = (key: 'color' | 'width') => [
  'step', DIAMETER, DIAMETER_CLASSES[0][key],
  ...DIAMETER_CLASSES.slice(1).flatMap((c) => [c.min, c[key]]),
]
// Hovered OR selected features are emphasised via feature-state.
const hoverCase = (on: unknown, off: unknown) => [
  'case',
  ['any', ['boolean', ['feature-state', 'hover'], false], ['boolean', ['feature-state', 'selected'], false]],
  on,
  off,
]

// Sizes grow with zoom so the classes stay apart when zoomed in. MapLibre only
// accepts ["zoom"] as the outermost interpolate input, so the diameter class and
// the hover emphasis sit inside each zoom stop's output.
const zoomScaled = (base: unknown, hoverExtra: number) => [
  'interpolate', ['linear'], ['zoom'],
  13, ['*', ['+', base, hoverCase(hoverExtra, 0)], 0.6],
  15, ['+', base, hoverCase(hoverExtra, 0)],
  17, ['*', ['+', base, hoverCase(hoverExtra, 0)], 1.8],
]
const CONDUIT_COLOR = byClass('color')
const CONDUIT_WIDTH = zoomScaled(byClass('width'), 1.5)
const NODE_RADIUS = zoomScaled(3.4, 1.5)
const NODE_HIT_RADIUS = ['interpolate', ['linear'], ['zoom'], 13, 7, 15, 9, 17, 12]

// MapLibre expression arrays vs react-map-gl's typed paint props.
const expr = (e: unknown) => e as unknown as number
const exprColor = (e: unknown) => e as unknown as string

// Click targets, topmost first: outfall > manhole > pipe > subcatchment.
const INSPECT_LAYERS = ['m-outfall-hit', 'm-junction-hit', 'm-conduit-hit', 'm-sub-fill']

// Attributes shown per element; the keys are SWMManywhere's own node / edge /
// subcatchment attributes (see the graphfcns' adds_node_attributes etc.).
const POPUP_ROWS: Record<string, [string, string, string?][]> = {
  junction: [
    ['Surface elevation', 'surface_elevation', 'm'],
    ['Chamber floor', 'chamber_floor_elevation', 'm'],
    ['Contributing area', 'contributing_area', 'm²'],
    ['Drains to outfall', 'outfall'],
  ],
  outfall: [
    ['Surface elevation', 'surface_elevation', 'm'],
    ['Chamber floor', 'chamber_floor_elevation', 'm'],
    ['Contributing area', 'contributing_area', 'm²'],
  ],
  conduit: [
    ['Diameter', 'diameter', 'm'],
    ['Length', 'length', 'm'],
    ['Surface slope', 'surface_slope', 'm/m'],
    ['Cost', 'cost_usd', 'USD'],
    ['From node', 'u'],
    ['To node', 'v'],
  ],
  subcatchment: [
    ['Area', 'area', 'm²'],
    ['Impervious (rc)', 'rc', '%'],
    ['Width', 'width', 'm'],
    ['Slope', 'slope', 'm/m'],
    ['Outlet node', 'outlet'],
  ],
}
// Variables swmmanywhere.run can store; the .inp template sets FLOW_UNITS LPS.
const SERIES_UNITS: Record<string, string> = { flow: 'L/s', flooding: 'L/s', depth: 'm', runoff: 'L/s' }
const KIND_LABEL: Record<string, string> = {
  junction: 'Manhole',
  outfall: 'Outfall',
  conduit: 'Pipe',
  subcatchment: 'Subcatchment',
}
const KIND_COLOR: Record<string, string> = {
  junction: NODE_STROKE,
  outfall: OUTFALL,
  conduit: DIAMETER_CLASSES[2].color,
  subcatchment: SUB_FILL,
}

// Swatch styles for the layer list: manholes are shown hollow, like on the map.
const LAYER_ROWS: [LayerKey, string, React.CSSProperties][] = [
  ['subcatchments', 'Subcatchments', { background: SUB_FILL }],
  ['conduits', 'Pipes', { background: DIAMETER_CLASSES[2].color }],
  ['junctions', 'Manholes', { background: '#ffffff', boxShadow: `inset 0 0 0 1.5px ${NODE_STROKE}` }],
  ['outfalls', 'Outfalls', { background: OUTFALL }],
]

function rectangle(a: [number, number], b: [number, number]): Feature<Polygon> {
  const [x0, y0, x1, y1] = [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])]
  return {
    type: 'Feature',
    properties: {},
    geometry: { type: 'Polygon', coordinates: [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]] },
  }
}
const bboxBounds = (b: Bbox): LngLatBoundsLike => [[b[0], b[1]], [b[2], b[3]]]

function fmtValue(v: unknown): string {
  if (typeof v === 'number') return Number.isInteger(v) ? String(v) : v.toPrecision(4)
  return String(v)
}

export default function MapPanel() {
  const mapRef = useRef<MapRef>(null)
  const drawing = useStore((s) => s.drawing)
  const anchor = useStore((s) => s.anchor)
  const bbox = useStore((s) => s.bbox)
  const fitToken = useStore((s) => s.fitToken)
  const mapClick = useStore((s) => s.mapClick)
  const preview = useStore((s) => s.preview)
  const layers = useStore((s) => s.layers)
  const toggleLayer = useStore((s) => s.toggleLayer)
  const selected = useStore((s) => s.selected)
  const select = useStore((s) => s.select)
  const timeseries = useStore((s) => s.timeseries)
  const [cursor, setCursor] = useState<[number, number] | null>(null)
  const [hovering, setHovering] = useState(false)
  const [showAll, setShowAll] = useState(false)
  const hoverId = useRef<number | string | null>(null)
  const selectedId = useRef<number | string | null>(null)

  const layerCounts = useMemo(() => {
    const c: Record<LayerKey, number> = { subcatchments: 0, conduits: 0, junctions: 0, outfalls: 0 }
    preview?.features.forEach((f) => {
      const kind = (f.properties as { kind?: string } | null)?.kind
      if (kind === 'subcatchment') c.subcatchments++
      else if (kind === 'conduit') c.conduits++
      else if (kind === 'junction') c.junctions++
      else if (kind === 'outfall') c.outfalls++
    })
    return c
  }, [preview])

  const setFeatureState = (id: number | string | null, key: 'hover' | 'selected', on: boolean) => {
    if (id !== null && mapRef.current) mapRef.current.setFeatureState({ source: 'model', id }, { [key]: on })
  }
  const setHover = (id: number | string | undefined) => {
    if (hoverId.current !== null && hoverId.current !== id) setFeatureState(hoverId.current, 'hover', false)
    hoverId.current = id ?? null
    if (id !== undefined) setFeatureState(id, 'hover', true)
  }
  const clearSelection = () => {
    setFeatureState(selectedId.current, 'selected', false)
    selectedId.current = null
    void select(null)
  }

  // Fit to the model when a preview arrives; a new build voids the old selection.
  useEffect(() => {
    clearSelection()
    if (preview && mapRef.current) {
      const b = boundsOf(preview)
      if (b) mapRef.current.fitBounds(b, { padding: 50, duration: 800 })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preview])

  // "Zoom to box" from the panel.
  useEffect(() => {
    if (fitToken && bbox && mapRef.current) mapRef.current.fitBounds(bboxBounds(bbox), { padding: 60, duration: 600 })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fitToken])

  useEffect(() => {
    if (drawing) {
      clearSelection()
      setCursor(null)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawing])

  const bboxFc: FeatureCollection = bbox ? { type: 'FeatureCollection', features: [rectangle([bbox[0], bbox[1]], [bbox[2], bbox[3]])] } : EMPTY
  const draftFc: FeatureCollection = anchor
    ? {
        type: 'FeatureCollection',
        features: [
          { type: 'Feature', properties: {}, geometry: { type: 'Point', coordinates: anchor } },
          ...(cursor ? [rectangle(anchor, cursor)] : []),
        ],
      }
    : EMPTY

  const pick = (e: MapLayerMouseEvent) => {
    const f = e.features?.[0]
    setFeatureState(selectedId.current, 'selected', false)
    selectedId.current = null
    if (f?.properties) {
      selectedId.current = f.id ?? null
      setFeatureState(f.id ?? null, 'selected', true)
      setShowAll(false)
      void select({
        id: String(f.properties.id ?? ''),
        kind: String(f.properties.kind ?? ''),
        props: f.properties as Record<string, unknown>,
        lng: e.lngLat.lng,
        lat: e.lngLat.lat,
      })
    } else {
      void select(null)
    }
  }

  const rows = showAll
    ? Object.keys(selected?.props ?? {})
        .filter((k) => k !== 'kind' && k !== 'id')
        .sort()
        .map((k) => [k, k] as [string, string, string?])
    : (POPUP_ROWS[selected?.kind ?? ''] ?? [])

  return (
    <div className="relative h-full w-full">
      <Map
        ref={mapRef}
        // The SWMManywhere docs' demo area (Andorra la Vella).
        initialViewState={{ longitude: 1.535, latitude: 42.509, zoom: 14.5 }}
        mapStyle={MAP_STYLE}
        style={{ width: '100%', height: '100%' }}
        cursor={drawing ? 'crosshair' : hovering ? 'pointer' : ''}
        interactiveLayerIds={drawing ? [] : INSPECT_LAYERS}
        // If the container had no size when the map was created (styles not yet
        // applied), MapLibre falls back to 400x300; re-measure once the style is in.
        onLoad={(e) => {
          e.target.resize()
          if (import.meta.env.DEV) (window as unknown as { __map?: unknown }).__map = e.target
        }}
        onClick={(e: MapLayerMouseEvent) => {
          if (drawing) mapClick(e.lngLat.lng, e.lngLat.lat)
          else pick(e)
        }}
        onMouseMove={(e: MapLayerMouseEvent) => {
          if (drawing) {
            if (anchor) setCursor([e.lngLat.lng, e.lngLat.lat])
            return
          }
          const f = e.features?.[0]
          setHover(f?.id as number | string | undefined)
          setHovering(!!f)
        }}
        onMouseLeave={() => {
          setHover(undefined)
          setHovering(false)
        }}
      >
        <NavigationControl position="top-right" showCompass={false} />
        <ScaleControl position="bottom-right" />

        {/* Synthesised model: subcatchments, pipes, manholes, outfalls. generateId powers feature-state. */}
        <Source id="model" type="geojson" data={preview ?? EMPTY} generateId>
          <Layer id="m-sub-fill" type="fill" filter={kindIs('subcatchment')} layout={vis(layers.subcatchments)}
            paint={{ 'fill-color': exprColor(hoverCase(ACCENT, SUB_FILL)), 'fill-opacity': expr(hoverCase(0.35, 0.16)) }} />
          <Layer id="m-sub-line" type="line" filter={kindIs('subcatchment')} layout={vis(layers.subcatchments)}
            paint={{ 'line-color': SUB_LINE, 'line-width': 0.5, 'line-opacity': 0.5 }} />
          <Layer id="m-conduit" type="line" filter={kindIs('conduit')} layout={{ 'line-cap': 'round', ...vis(layers.conduits) }}
            paint={{ 'line-color': exprColor(hoverCase(ACCENT, CONDUIT_COLOR)),
                     'line-width': expr(CONDUIT_WIDTH) }} />
          {/* Flow direction (u -> v): white chevrons with a halo in the pipe's colour. */}
          <Layer id="m-flow" type="symbol" filter={kindIs('conduit')} minzoom={15}
            layout={{ ...vis(layers.conduits), 'symbol-placement': 'line', 'symbol-spacing': 60,
                      'text-field': '>', 'text-size': 12, 'text-font': ['Noto Sans Bold'],
                      'text-keep-upright': false, 'text-allow-overlap': true, 'text-rotation-alignment': 'map' }}
            paint={{ 'text-color': '#ffffff', 'text-halo-color': exprColor(CONDUIT_COLOR), 'text-halo-width': 1.4 }} />
          <Layer id="m-junction" type="circle" filter={kindIs('junction')} layout={vis(layers.junctions)}
            paint={{ 'circle-radius': expr(NODE_RADIUS),
                     'circle-color': exprColor(hoverCase(ACCENT, '#ffffff')),
                     'circle-stroke-width': 1.6, 'circle-stroke-color': NODE_STROKE }} />
          <Layer id="m-outfall" type="circle" filter={kindIs('outfall')} layout={vis(layers.outfalls)}
            paint={{ 'circle-radius': expr(hoverCase(8.5, 7)), 'circle-color': OUTFALL,
                     'circle-stroke-width': expr(hoverCase(3, 2)), 'circle-stroke-color': exprColor(hoverCase(ACCENT, '#ffffff')) }} />

          {/* Invisible fat hit targets; declared last = queried first. */}
          <Layer id="m-conduit-hit" type="line" filter={kindIs('conduit')} layout={vis(layers.conduits)}
            paint={{ 'line-color': '#000000', 'line-width': 12, 'line-opacity': 0 }} />
          <Layer id="m-junction-hit" type="circle" filter={kindIs('junction')} layout={vis(layers.junctions)}
            paint={{ 'circle-radius': expr(NODE_HIT_RADIUS), 'circle-color': '#000000', 'circle-opacity': 0 }} />
          <Layer id="m-outfall-hit" type="circle" filter={kindIs('outfall')} layout={vis(layers.outfalls)}
            paint={{ 'circle-radius': 12, 'circle-color': '#000000', 'circle-opacity': 0 }} />
        </Source>

        {/* Committed bounding box */}
        <Source id="bbox" type="geojson" data={bboxFc}>
          <Layer id="bbox-fill" type="fill" paint={{ 'fill-color': ACCENT, 'fill-opacity': 0.05 }} />
          <Layer id="bbox-line" type="line" paint={{ 'line-color': ACCENT, 'line-width': 2 }} />
        </Source>

        {/* In-progress box while drawing */}
        <Source id="draft" type="geojson" data={draftFc}>
          <Layer id="draft-line" type="line" paint={{ 'line-color': ACCENT, 'line-width': 2 }} />
          <Layer id="draft-pt" type="circle"
            paint={{ 'circle-radius': 4, 'circle-color': ACCENT, 'circle-stroke-width': 1.5, 'circle-stroke-color': '#ffffff' }} />
        </Source>
      </Map>

      {/* Layer toggles (top-left) */}
      {preview && (
        <div className="absolute left-3 top-3 z-10 w-56 rounded-xl border border-zinc-200 bg-white/95 shadow-sm backdrop-blur">
          <div className="px-3.5 pb-1 pt-3 text-[12px] font-semibold text-zinc-800">Layers</div>
          <div className="px-1.5 pb-1.5">
            {LAYER_ROWS.map(([key, label, swatch]) => (
              <button
                key={key}
                onClick={() => toggleLayer(key)}
                className="flex w-full items-center gap-2.5 rounded-md px-2 py-1.5 text-left text-[13px] transition-colors hover:bg-zinc-50"
              >
                <span className={`h-2.5 w-2.5 shrink-0 rounded-full transition-opacity ${layers[key] ? '' : 'opacity-25'}`} style={swatch} />
                <span className={`flex-1 ${layers[key] ? 'text-zinc-800' : 'text-zinc-400'}`}>{label}</span>
                <span className="font-mono text-[11px] text-zinc-400">{layerCounts[key]}</span>
                {layers[key] ? <Eye size={14} className="text-zinc-400" /> : <EyeOff size={14} className="text-zinc-300" />}
              </button>
            ))}
          </div>
          <div className="border-t border-zinc-100 px-3.5 py-2">
            <div className="mb-1.5 text-[11px] font-medium text-zinc-500">Pipe diameter</div>
            {DIAMETER_CLASSES.map((c) => (
              <div key={c.label} className="flex items-center gap-2.5 py-0.5 text-[11px] text-zinc-600">
                <span className="inline-block w-7 shrink-0 rounded-full" style={{ height: c.width, background: c.color }} />
                {c.label}
              </div>
            ))}
          </div>
          <p className="border-t border-zinc-100 px-3.5 py-2 text-[11px] leading-snug text-zinc-400">
            Arrows show flow direction. Click an element.
          </p>
        </div>
      )}

      {/* Element card (bottom-left): attributes and, when the model was run, time series */}
      {selected && (
        <div className="absolute bottom-5 left-3 z-10 max-h-[78%] w-80 overflow-y-auto rounded-xl border border-zinc-200 bg-white shadow-md">
          <div className="flex items-center gap-2 px-3.5 py-2.5">
            <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: KIND_COLOR[selected.kind] ?? NODE_STROKE }} />
            <span className="text-[12px] font-medium text-zinc-500">{KIND_LABEL[selected.kind] ?? selected.kind}</span>
            <span className="ml-auto truncate font-mono text-[13px] font-semibold text-zinc-900">{selected.id}</span>
            <button onClick={clearSelection} className="rounded p-1 text-zinc-300 transition-colors hover:bg-zinc-100 hover:text-zinc-600">
              <X size={14} />
            </button>
          </div>
          <div className="border-t border-zinc-100 px-3.5 py-2">
            {rows
              .filter(([, key]) => selected.props[key] !== undefined && selected.props[key] !== null)
              .map(([label, key, unit]) => (
                <div key={key} className="flex items-baseline justify-between gap-4 py-1 text-[12.5px]">
                  <span className="truncate text-zinc-500">{label}</span>
                  <span className="truncate font-mono text-zinc-900">
                    {fmtValue(selected.props[key])}
                    {unit ? <span className="ml-1 font-sans text-[11px] text-zinc-400">{unit}</span> : null}
                  </span>
                </div>
              ))}
            <button
              onClick={() => setShowAll((v) => !v)}
              className="mt-1 text-[11px] text-zinc-400 transition-colors hover:text-zinc-700"
            >
              {showAll ? 'Summary' : 'All attributes'}
            </button>
          </div>
          {timeseries && Object.keys(timeseries).length > 0 && (
            <div className="space-y-3 border-t border-zinc-100 px-3.5 py-3">
              {Object.entries(timeseries).map(([variable, s]) => (
                <TimeseriesChart key={variable} label={variable} unit={SERIES_UNITS[variable] ?? ''} dates={s.dates} values={s.values} />
              ))}
            </div>
          )}
          <div className="border-t border-zinc-100 p-3">
            <button
              onClick={() =>
                mapRef.current?.flyTo({
                  center: [selected.lng, selected.lat],
                  zoom: Math.max(mapRef.current.getZoom(), 17),
                  duration: 700,
                })
              }
              className={`${BTN_SECONDARY} h-8 w-full`}
            >
              <LocateFixed size={14} /> Fly to
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function boundsOf(fc: FeatureCollection): LngLatBoundsLike | null {
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity, found = false
  const walk = (c: unknown): void => {
    const arr = c as number[]
    if (typeof arr[0] === 'number') {
      const x = arr[0], y = arr[1]
      if (x < minx) minx = x
      if (y < miny) miny = y
      if (x > maxx) maxx = x
      if (y > maxy) maxy = y
      found = true
    } else {
      ;(c as unknown[]).forEach(walk)
    }
  }
  for (const f of fc.features) {
    if (f.geometry && 'coordinates' in f.geometry) walk((f.geometry as { coordinates: unknown }).coordinates)
  }
  return found ? [[minx, miny], [maxx, maxy]] : null
}
