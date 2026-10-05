import { create } from 'zustand'
import type { FeatureCollection } from 'geojson'
import type { Bbox, Defaults, JobProgress, Overrides, RunSettings, Timeseries } from './types'
import { fetchConfigYaml, fetchDefaults, fetchPreview, fetchTimeseries, pollTask, submitTask } from './lib/api'

export type LayerKey = 'subcatchments' | 'conduits' | 'junctions' | 'outfalls'

export interface Selected {
  id: string
  kind: string
  props: Record<string, unknown>
  lng: number
  lat: number
}

const TERMINAL = new Set(['succeeded', 'failed'])
const DEFAULT_LAYERS: Record<LayerKey, boolean> = {
  subcatchments: true,
  conduits: true,
  junctions: true,
  outfalls: true,
}
const IDLE: JobProgress = { status: 'idle' }

// Two corners in any order -> SWMManywhere's [min lon, min lat, max lon, max lat].
export function normaliseBbox(a: [number, number], b: [number, number]): Bbox {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])]
}

// Approximate area of a lon/lat box; enough for the size guard and the label.
export function bboxAreaKm2([minLon, minLat, maxLon, maxLat]: Bbox): number {
  const midLat = ((minLat + maxLat) / 2) * (Math.PI / 180)
  return (maxLon - minLon) * 111.32 * Math.cos(midLat) * (maxLat - minLat) * 110.57
}

// The graph-function textarea: one name per line, blank lines and # comments ignored.
export function graphfcnLines(text: string): string[] {
  return text
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'))
}

// Everything that is stale once the area changes.
const CLEARED = { job: IDLE, taskId: null, preview: null, selected: null, timeseries: null, configYaml: '' }

interface AppState {
  defaults: Defaults | null
  defaultsError: string | null
  bbox: Bbox | null
  drawing: boolean
  anchor: [number, number] | null // first corner while drawing
  fitToken: number // bumped to ask the map to fit the bbox
  project: string
  overrides: Overrides
  graphfcnText: string
  runModel: boolean
  runSettings: RunSettings
  job: JobProgress
  taskId: string | null
  preview: FeatureCollection | null
  configYaml: string
  layers: Record<LayerKey, boolean>
  selected: Selected | null
  timeseries: Timeseries | null

  loadDefaults: () => Promise<void>
  startDraw: () => void
  cancelDraw: () => void
  mapClick: (lng: number, lat: number) => void
  setBbox: (bbox: Bbox | null) => void
  requestFit: () => void
  setProject: (project: string) => void
  setOverride: (group: string, field: string, value: unknown | undefined) => void
  resetGroup: (group: string) => void
  setGraphfcnText: (text: string) => void
  setRunModel: (on: boolean) => void
  setRunSettings: (patch: Partial<RunSettings>) => void
  toggleLayer: (key: LayerKey) => void
  select: (selected: Selected | null) => Promise<void>
  submit: () => Promise<void>
}

export const useStore = create<AppState>((set, get) => ({
  defaults: null,
  defaultsError: null,
  bbox: null,
  drawing: false,
  anchor: null,
  fitToken: 0,
  project: 'my_first_swmm',
  overrides: {},
  graphfcnText: '',
  runModel: true,
  runSettings: { duration: 86400, reporting_iters: 100, storevars: ['flooding', 'flow'] },
  job: IDLE,
  taskId: null,
  preview: null,
  configYaml: '',
  layers: DEFAULT_LAYERS,
  selected: null,
  timeseries: null,

  loadDefaults: async () => {
    try {
      const d = await fetchDefaults()
      set({
        defaults: d,
        defaultsError: null,
        graphfcnText: d.graphfcn_list.join('\n'),
        runSettings: { ...d.run_settings },
      })
    } catch (err) {
      set({
        defaultsError:
          `Server not reachable (${err instanceof Error ? err.message : err}). ` +
          'Start it from the frontend folder with: uvicorn server.app:app --port 8000',
      })
    }
  },

  startDraw: () => set({ drawing: true, anchor: null, bbox: null, ...CLEARED }),
  cancelDraw: () => set({ drawing: false, anchor: null }),
  mapClick: (lng, lat) => {
    const { drawing, anchor } = get()
    if (!drawing) return
    if (!anchor) {
      set({ anchor: [lng, lat] })
      return
    }
    set({ bbox: normaliseBbox(anchor, [lng, lat]), drawing: false, anchor: null })
  },
  setBbox: (bbox) => set({ bbox, drawing: false, anchor: null, ...CLEARED }),
  requestFit: () => set((s) => ({ fitToken: s.fitToken + 1 })),
  setProject: (project) => set({ project }),

  setOverride: (group, field, value) =>
    set((s) => {
      const g = { ...(s.overrides[group] ?? {}) }
      if (value === undefined) delete g[field]
      else g[field] = value
      const overrides = { ...s.overrides, [group]: g }
      if (Object.keys(g).length === 0) delete overrides[group]
      return { overrides }
    }),
  resetGroup: (group) =>
    set((s) => {
      const overrides = { ...s.overrides }
      delete overrides[group]
      return { overrides }
    }),
  setGraphfcnText: (graphfcnText) => set({ graphfcnText }),
  setRunModel: (runModel) => set({ runModel }),
  setRunSettings: (patch) => set((s) => ({ runSettings: { ...s.runSettings, ...patch } })),
  toggleLayer: (key) => set((s) => ({ layers: { ...s.layers, [key]: !s.layers[key] } })),

  select: async (selected) => {
    set({ selected, timeseries: null })
    const { taskId, runModel } = get()
    if (!selected || !taskId || !runModel) return
    try {
      const ts = await fetchTimeseries(taskId, selected.id)
      if (get().selected?.id === selected.id) set({ timeseries: ts })
    } catch {
      // no simulation results for this element
    }
  },

  submit: async () => {
    const { bbox, defaults, project, overrides, graphfcnText, runModel, runSettings } = get()
    if (!bbox || !defaults) return
    const lines = graphfcnLines(graphfcnText)
    const graphfcnList = lines.join('\n') === defaults.graphfcn_list.join('\n') ? null : lines
    set({ ...CLEARED, job: { status: 'queued' } })
    try {
      const taskId = await submitTask({ bbox, project, parameterOverrides: overrides, graphfcnList, runModel, runSettings })
      set({ taskId })
      // The exact config.yml the server validated: lets users reproduce the run from the CLI.
      fetchConfigYaml(taskId)
        .then((configYaml) => set({ configYaml }))
        .catch(() => {})
      for (;;) {
        const p = await pollTask(taskId)
        set({ job: p })
        if (TERMINAL.has(p.status)) {
          if (p.status === 'succeeded') set({ preview: await fetchPreview(taskId) })
          break
        }
        await new Promise((r) => setTimeout(r, 1500))
      }
    } catch (err) {
      set({ job: { status: 'failed', message: err instanceof Error ? err.message : `${err}` } })
    }
  },
}))

// Dev-only hook so automated checks can drive the real flow without synthetic map
// events. No effect in production builds.
if (typeof window !== 'undefined' && import.meta.env?.DEV) {
  ;(window as unknown as { __store?: typeof useStore }).__store = useStore
}
