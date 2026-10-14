interface Props {
  label: string
  unit: string
  dates: string[] // ISO timestamps from the server
  values: number[]
}

const W = 256
const H = 96
const PL = 36
const PR = 6
const PT = 8
const PB = 14

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toPrecision(2))
// "2000-01-01T06:15:00" -> "01-01 06:15" (day shown because runs can exceed 24 h)
const stamp = (iso: string) => iso.slice(5, 16).replace('T', ' ')

// Minimal SVG line chart for one simulated variable of one element.
export default function TimeseriesChart({ label, unit, dates, values }: Props) {
  if (values.length < 2) {
    return <p className="text-[11px] text-zinc-400">{label}: not enough data points</p>
  }
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  const x = (i: number) => PL + (i / (values.length - 1)) * (W - PL - PR)
  const y = (v: number) => PT + (1 - (v - min) / span) * (H - PT - PB)
  const path = values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ')

  return (
    <div>
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="font-medium text-zinc-700">
          {label} <span className="font-normal text-zinc-400">{unit}</span>
        </span>
        <span className="font-mono text-zinc-400">max {fmt(max)}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="h-24 w-full">
        <line x1={PL} y1={PT} x2={PL} y2={H - PB} stroke="#e4e4e7" strokeWidth={1} />
        <line x1={PL} y1={H - PB} x2={W - PR} y2={H - PB} stroke="#e4e4e7" strokeWidth={1} />
        <text x={PL - 3} y={PT + 3} fontSize={8} textAnchor="end" fill="#a1a1aa">
          {fmt(max)}
        </text>
        <text x={PL - 3} y={H - PB} fontSize={8} textAnchor="end" fill="#a1a1aa">
          {fmt(min)}
        </text>
        <text x={PL} y={H - 3} fontSize={8} fill="#a1a1aa">
          {stamp(dates[0])}
        </text>
        <text x={W - PR} y={H - 3} fontSize={8} textAnchor="end" fill="#a1a1aa">
          {stamp(dates[dates.length - 1])}
        </text>
        <path d={path} fill="none" stroke="#0e7c8b" strokeWidth={1.4} />
      </svg>
    </div>
  )
}
