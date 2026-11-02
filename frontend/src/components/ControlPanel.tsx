import { useEffect, useMemo, useState } from 'react'
import { ChevronRight, Download, Loader2, LocateFixed, Play, RotateCcw, SquareDashed, Trash2 } from 'lucide-react'
import { bboxAreaKm2, graphfcnLines, useStore } from '../store'
import type { Bbox } from '../types'
import { BTN_PRIMARY, ICON_BTN, INPUT, LABEL } from '../ui'
import ParameterGroup from './ParameterGroup'

const BBOX_LABELS = ['min lon', 'min lat', 'max lon', 'max lat']

// metric_evaluation only applies when a real network is supplied for comparison
// (config `real:`), which this form does not do.
const HIDDEN_GROUPS = new Set(['metric_evaluation'])

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-t border-zinc-100 px-5 py-4">
      <h2 className="mb-3 text-[13px] font-semibold text-zinc-900">{title}</h2>
      {children}
    </section>
  )
}

export default function ControlPanel() {
  const defaults = useStore((s) => s.defaults)
  const defaultsError = useStore((s) => s.defaultsError)
  const loadDefaults = useStore((s) => s.loadDefaults)
  const bbox = useStore((s) => s.bbox)
  const drawing = useStore((s) => s.drawing)
  const anchor = useStore((s) => s.anchor)
  const startDraw = useStore((s) => s.startDraw)
  const cancelDraw = useStore((s) => s.cancelDraw)
  const setBbox = useStore((s) => s.setBbox)
  const requestFit = useStore((s) => s.requestFit)
  const project = useStore((s) => s.project)
  const setProject = useStore((s) => s.setProject)
  const overrides = useStore((s) => s.overrides)
  const setOverride = useStore((s) => s.setOverride)
  const resetGroup = useStore((s) => s.resetGroup)
  const graphfcnText = useStore((s) => s.graphfcnText)
  const setGraphfcnText = useStore((s) => s.setGraphfcnText)
  const runModel = useStore((s) => s.runModel)
  const setRunModel = useStore((s) => s.setRunModel)
  const runSettings = useStore((s) => s.runSettings)
  const setRunSettings = useStore((s) => s.setRunSettings)
  const job = useStore((s) => s.job)
  const submit = useStore((s) => s.submit)
  const preview = useStore((s) => s.preview)
  const configYaml = useStore((s) => s.configYaml)
  const [showLog, setShowLog] = useState(false)
  // Typed bbox: the four boxes are free text until all of them parse, so a CLI
  // user can paste numbers without drawing first.
  const [bboxDraft, setBboxDraft] = useState<string[]>(['', '', '', ''])

  useEffect(() => {
    void loadDefaults()
  }, [loadDefaults])

  useEffect(() => {
    if (!bbox) setBboxDraft(['', '', '', ''])
    else if (bboxDraft.some((t, i) => Number(t) !== bbox[i])) setBboxDraft(bbox.map(String))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bbox])

  const counts = useMemo(() => {
    const c = { subcatchment: 0, junction: 0, outfall: 0, conduit: 0 }
    preview?.features.forEach((f) => {
      const kind = (f.properties as { kind?: string } | null)?.kind
      if (kind && kind in c) c[kind as keyof typeof c]++
    })
    return c
  }, [preview])

  const busy = job.status === 'queued' || job.status === 'running'
  const areaKm2 = bbox ? bboxAreaKm2(bbox) : 0
  const tooLarge = !!defaults && areaKm2 > defaults.max_area_km2
  const graphfcnChanged = !!defaults && graphfcnText.trim() !== defaults.graphfcn_list.join('\n')
  const storevarChoices = defaults?.storevars ?? runSettings.storevars

  const editBbox = (i: number, text: string) => {
    const next = [...bboxDraft]
    next[i] = text
    setBboxDraft(next)
    const nums = next.map(Number)
    if (next.every((t) => t.trim() !== '') && nums.every(Number.isFinite)) setBbox(nums as Bbox)
  }

  const toggleStorevar = (v: string) =>
    setRunSettings({
      storevars: runSettings.storevars.includes(v)
        ? runSettings.storevars.filter((s) => s !== v)
        : [...runSettings.storevars, v],
    })

  return (
    <aside className="flex h-full w-full flex-col overflow-y-auto bg-white">
      <header className="px-5 pb-4 pt-5">
        <div className="flex items-center gap-2.5">
          <img src={`${import.meta.env.BASE_URL}logo.svg`} alt="" className="h-7 w-7 shrink-0" />
          <a
            href="https://github.com/ImperialCollegeLondon/SWMManywhere"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[20px] font-semibold tracking-tight text-zinc-900 transition-colors hover:text-accent-deep"
          >
            SWMManywhere
          </a>
          {defaults && <span className="ml-auto font-mono text-[11px] text-zinc-400">{defaults.version}</span>}
        </div>
        <p className="mt-2 text-[13px] leading-snug text-zinc-500">
          Synthesise a drainage network model for any bounding box and run it in SWMM. Each build is a plain{' '}
          <code>config.yml</code>.
        </p>
        {defaultsError && (
          <p className="mt-3 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-[12px] leading-snug text-red-700">
            {defaultsError}
          </p>
        )}
      </header>

      <Section title="Area">
        {!drawing ? (
          <button onClick={startDraw} className={`${BTN_PRIMARY} w-full`}>
            <SquareDashed size={15} /> Draw bounding box
          </button>
        ) : (
          <div className="flex items-center gap-2 rounded-md border border-accent/40 bg-accent/5 px-3 py-2 text-[12px] text-zinc-700">
            <span className="flex-1">{anchor ? 'Click the opposite corner' : 'Click the first corner on the map'}</span>
            <button onClick={cancelDraw} className="font-medium text-zinc-500 transition-colors hover:text-zinc-900">
              Cancel
            </button>
          </div>
        )}
        <div className="mt-3 grid grid-cols-2 gap-2">
          {BBOX_LABELS.map((label, i) => (
            <label key={label} className={LABEL}>
              {label}
              <input
                type="text"
                inputMode="decimal"
                value={bboxDraft[i]}
                placeholder="–"
                onChange={(e) => editBbox(i, e.target.value)}
                className={`${INPUT} mt-1 font-mono text-[12px]`}
              />
            </label>
          ))}
        </div>
        {bbox && (
          <div className="mt-2 flex items-center justify-between text-[12px] text-zinc-600">
            <span className={tooLarge ? 'font-medium text-red-600' : ''}>
              {areaKm2.toFixed(2)} km²
              {defaults && <span className="text-zinc-400"> · limit {defaults.max_area_km2} km²</span>}
            </span>
            <span className="flex items-center">
              <button onClick={requestFit} title="Zoom to box" className={ICON_BTN}>
                <LocateFixed size={14} />
              </button>
              <button onClick={() => setBbox(null)} title="Clear" className={ICON_BTN}>
                <Trash2 size={14} />
              </button>
            </span>
          </div>
        )}
        <label className={`${LABEL} mt-3`} title="Folder name under the server's base_dir">
          Project
          <input
            type="text"
            value={project}
            onChange={(e) => setProject(e.target.value)}
            pattern="[A-Za-z0-9_-]+"
            className={`${INPUT} mt-1 font-mono text-[12px]`}
          />
        </label>
      </Section>

      <Section title="Parameters">
        <p className="mb-2 text-[12px] text-zinc-500">
          Package defaults. Only changed values are sent as <code>parameter_overrides</code>.
        </p>
        {defaults && (
          <div className="divide-y divide-zinc-100 overflow-hidden rounded-lg border border-zinc-200">
            {defaults.parameters
              .filter((g) => !HIDDEN_GROUPS.has(g.name))
              .map((g) => (
                <ParameterGroup
                  key={g.name}
                  group={g}
                  values={overrides[g.name] ?? {}}
                  onChange={(field, value) => setOverride(g.name, field, value)}
                  onReset={() => resetGroup(g.name)}
                />
              ))}
          </div>
        )}
      </Section>

      <Section title="Graph functions">
        <details className="group overflow-hidden rounded-lg border border-zinc-200">
          <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-[13px] font-medium text-zinc-800 transition-colors hover:bg-zinc-50">
            <ChevronRight size={14} className="shrink-0 text-zinc-400 transition-transform duration-200 group-open:rotate-90" />
            <span className="font-mono text-[12px]">graphfcn_list</span>
            {graphfcnChanged && <span className="text-[11px] font-medium text-accent-deep">edited</span>}
            <span className="ml-auto font-mono text-[11px] text-zinc-400">{graphfcnLines(graphfcnText).length} steps</span>
          </summary>
          <div className="border-t border-zinc-100 bg-zinc-50/70 px-3 py-3">
            <textarea
              value={graphfcnText}
              onChange={(e) => setGraphfcnText(e.target.value)}
              rows={10}
              spellCheck={false}
              className={`${INPUT} h-auto py-2 font-mono text-[11.5px] leading-relaxed`}
            />
            <div className="mt-2 flex items-center justify-between text-[11px] text-zinc-500">
              <span>One per line, applied in order. Validated by the package before building.</span>
              {graphfcnChanged && (
                <button
                  onClick={() => setGraphfcnText(defaults!.graphfcn_list.join('\n'))}
                  className="flex shrink-0 items-center gap-1 transition-colors hover:text-zinc-900"
                >
                  <RotateCcw size={12} /> Reset
                </button>
              )}
            </div>
            {defaults && (
              <details className="group/list mt-2">
                <summary className="flex cursor-pointer items-center gap-1 text-[11px] text-zinc-500 transition-colors hover:text-zinc-900">
                  <ChevronRight size={12} className="transition-transform duration-200 group-open/list:rotate-90" />
                  Available functions
                </summary>
                <dl className="mt-1.5 space-y-1 text-[11px]">
                  {Object.entries(defaults.graphfcns).map(([name, doc]) => (
                    <div key={name} className="grid grid-cols-[11rem_1fr] gap-2">
                      <dt className="truncate font-mono text-zinc-700">{name}</dt>
                      <dd className="text-zinc-400">{doc}</dd>
                    </div>
                  ))}
                </dl>
              </details>
            )}
          </div>
        </details>
      </Section>

      <Section title="Simulation">
        <label className="flex items-center gap-2 text-[13px] text-zinc-800">
          <input
            type="checkbox"
            checked={runModel}
            onChange={(e) => setRunModel(e.target.checked)}
            className="h-3.5 w-3.5 accent-accent-deep"
          />
          Run in SWMM after synthesis
        </label>
        {runModel && (
          <div className="mt-3 space-y-3">
            <div className="grid grid-cols-2 gap-2">
              <label className={LABEL}>
                duration (s)
                <input
                  type="number"
                  min={60}
                  step={60}
                  value={runSettings.duration}
                  onChange={(e) => setRunSettings({ duration: Number(e.target.value) })}
                  className={`${INPUT} mt-1 font-mono text-[12px]`}
                />
              </label>
              <label className={LABEL}>
                reporting_iters
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={runSettings.reporting_iters}
                  onChange={(e) => setRunSettings({ reporting_iters: Number(e.target.value) })}
                  className={`${INPUT} mt-1 font-mono text-[12px]`}
                />
              </label>
            </div>
            <div>
              <span className={LABEL}>storevars</span>
              <div className="mt-1 flex flex-wrap gap-1.5">
                {storevarChoices.map((v) => {
                  const on = runSettings.storevars.includes(v)
                  return (
                    <button
                      key={v}
                      type="button"
                      aria-pressed={on}
                      onClick={() => toggleStorevar(v)}
                      className={`rounded-md border px-2 py-1 font-mono text-[12px] transition-colors ${
                        on
                          ? 'border-accent-deep bg-accent-deep text-white'
                          : 'border-zinc-200 bg-white text-zinc-600 hover:bg-zinc-50'
                      }`}
                    >
                      {v}
                    </button>
                  )
                })}
              </div>
            </div>
            <p className="text-[11px] text-zinc-400">
              Rain: the package&apos;s demo storm, <code>defs/storm.dat</code>, as with the CLI.
            </p>
          </div>
        )}
      </Section>

      <Section title="Build">
        <button onClick={submit} disabled={!bbox || !defaults || busy || tooLarge} className={`${BTN_PRIMARY} w-full`}>
          {busy ? <Loader2 size={15} className="animate-spin" /> : <Play size={15} />}
          {busy ? (job.stage ?? 'Building') : 'Build model'}
        </button>
        {tooLarge && <p className="mt-2 text-[12px] text-red-600">The box exceeds the server&apos;s size limit.</p>}
        {job.status !== 'idle' && (
          <div className="mt-3 rounded-lg bg-zinc-50 px-3 py-2.5 text-[12px] text-zinc-600">
            <div className="flex items-center justify-between">
              <span className="font-medium capitalize text-zinc-800">{job.status}</span>
              {typeof job.progressPct === 'number' && (
                <span className="font-mono text-zinc-400">{Math.round(job.progressPct)}%</span>
              )}
            </div>
            {typeof job.progressPct === 'number' && (
              <div className="mt-2 h-0.5 w-full overflow-hidden rounded bg-zinc-200">
                <div
                  className={`h-full transition-[width] duration-500 ${job.status === 'failed' ? 'bg-red-500' : 'bg-accent'}`}
                  style={{ width: `${Math.min(100, job.progressPct)}%` }}
                />
              </div>
            )}
            {job.stage && <div className="mt-1.5 text-zinc-500">{job.stage}</div>}
            {job.message && <div className="mt-1.5 whitespace-pre-wrap text-red-600">{job.message}</div>}
            {job.log && job.log.length > 0 && (
              <>
                <button
                  onClick={() => setShowLog((v) => !v)}
                  className="mt-1.5 text-[11px] text-zinc-400 transition-colors hover:text-zinc-700"
                >
                  {showLog ? 'Hide log' : 'Show log'}
                </button>
                {showLog && (
                  <pre className="mt-1.5 max-h-40 overflow-auto rounded-md border border-zinc-200 bg-white p-2 font-mono text-[10.5px] leading-snug text-zinc-500">
                    {job.log.join('\n')}
                  </pre>
                )}
              </>
            )}
            {job.status === 'succeeded' && job.resultUrl && (
              <a
                href={job.resultUrl}
                className="mt-2.5 inline-flex items-center gap-1.5 font-medium text-accent-deep transition-colors hover:text-accent-deeper"
              >
                <Download size={14} /> Download model package
              </a>
            )}
          </div>
        )}
        {configYaml && (
          <details className="group mt-2">
            <summary className="flex cursor-pointer items-center gap-1 text-[11px] text-zinc-500 transition-colors hover:text-zinc-900">
              <ChevronRight size={12} className="transition-transform duration-200 group-open:rotate-90" />
              config.yml for this build
            </summary>
            <pre className="mt-2 max-h-64 overflow-auto rounded-md border border-zinc-200 bg-zinc-50 p-3 font-mono text-[10.5px] leading-snug text-zinc-600">
              {configYaml}
            </pre>
          </details>
        )}
        {preview && (
          <p className="mt-3 text-[12px] leading-snug text-zinc-500">
            {counts.subcatchment} subcatchments, {counts.conduit} pipes, {counts.junction} manholes,{' '}
            {counts.outfall} outfall{counts.outfall === 1 ? '' : 's'}. Click any element on the map for its attributes
            {runModel ? ' and time series' : ''}.
          </p>
        )}
      </Section>

      <footer className="mt-auto border-t border-zinc-100 px-5 py-4 text-[11px] leading-relaxed text-zinc-400">
        SWMManywhere is developed at Imperial College London under BSD-3-Clause. Please cite{' '}
        <a href="https://doi.org/10.21105/joss.07729" target="_blank" rel="noopener noreferrer" className="underline decoration-zinc-300 underline-offset-2 hover:text-zinc-700">
          Dobson et al. 2025, JOSS
        </a>{' '}
        and{' '}
        <a href="https://doi.org/10.1016/j.envsoft.2025.106358" target="_blank" rel="noopener noreferrer" className="underline decoration-zinc-300 underline-offset-2 hover:text-zinc-700">
          Dobson et al. 2025, EM&amp;S
        </a>
        .{' '}
        <a href="https://imperialcollegelondon.github.io/SWMManywhere/" target="_blank" rel="noopener noreferrer" className="underline decoration-zinc-300 underline-offset-2 hover:text-zinc-700">
          Documentation
        </a>
      </footer>
    </aside>
  )
}
