// [min lon, min lat, max lon, max lat] — the order SWMManywhere's config `bbox` uses.
export type Bbox = [number, number, number, number]

export type JobStatus = 'idle' | 'queued' | 'running' | 'succeeded' | 'failed'

export interface JobProgress {
  status: JobStatus
  stage?: string
  progressPct?: number
  message?: string
  log?: string[]
  resultUrl?: string
}

// One field of a swmmanywhere parameter group, straight from the pydantic JSON
// schema of swmmanywhere.parameters (served by GET /api/v1/defaults).
export interface ParamField {
  name: string
  type: 'number' | 'integer' | 'string' | 'array' | 'boolean' | 'unknown'
  default: unknown
  unit?: string | null
  description?: string | null
  minimum?: number | null
  maximum?: number | null
  exclusiveMaximum?: number | null
}

export interface ParamGroup {
  name: string
  doc: string
  fields: ParamField[]
}

export interface RunSettings {
  duration: number
  reporting_iters: number
  storevars: string[]
}

export interface Defaults {
  version: string
  parameters: ParamGroup[]
  graphfcn_list: string[]
  graphfcns: Record<string, string>
  run_settings: RunSettings
  storevars: string[]
  max_area_km2: number
}

// parameter_overrides as SWMManywhere's config expects: {group: {field: value}}.
export type Overrides = Record<string, Record<string, unknown>>

// Simulation results for one element: {variable: {dates, values}}.
export type Timeseries = Record<string, { dates: string[]; values: number[] }>
