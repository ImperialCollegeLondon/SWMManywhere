import { ChevronRight, RotateCcw } from 'lucide-react'
import type { ParamField, ParamGroup as Group } from '../types'
import { INPUT } from '../ui'

// Choices that swmmanywhere/parameters.py documents in the field descriptions but
// does not encode in the schema, so the UI can offer a select instead of free text.
const CHOICES: Record<string, string[]> = {
  'subcatchment_derivation.subbasin_clip_method': ['subbasin', 'community'],
  'outfall_derivation.method': ['separate', 'withtopo'],
}

const FIELD_INPUT = `${INPUT} h-7 text-[12px]`

const sameValue = (a: unknown, b: unknown) => JSON.stringify(a) === JSON.stringify(b)

// "0.15, 0.225, 0.3" -> numbers when every token is numeric, otherwise strings.
function parseList(text: string): unknown[] {
  const tokens = text
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean)
  return tokens.every((t) => !Number.isNaN(Number(t))) ? tokens.map(Number) : tokens
}

function meta(f: ParamField): string {
  const unit = f.unit && f.unit !== '-' ? f.unit : ''
  const hi = f.maximum ?? f.exclusiveMaximum
  const range = f.minimum == null && hi == null ? '' : `${f.minimum ?? ''}–${hi ?? ''}`
  return [unit, range].filter(Boolean).join(' · ')
}

interface Props {
  group: Group
  values: Record<string, unknown> // current overrides for this group
  onChange: (field: string, value: unknown | undefined) => void
  onReset: () => void
}

// One collapsible parameter group rendered from the package's own pydantic schema.
// Only values that differ from the package default become parameter_overrides.
export default function ParameterGroup({ group, values, onChange, onReset }: Props) {
  const changed = Object.keys(values).length
  return (
    <details className="group">
      <summary className="flex cursor-pointer select-none items-center gap-2 px-3 py-2 text-[13px] font-medium text-zinc-800 transition-colors hover:bg-zinc-50">
        <ChevronRight size={14} className="shrink-0 text-zinc-400 transition-transform duration-200 group-open:rotate-90" />
        <span className="flex-1">{group.name.replace(/_/g, ' ')}</span>
        {changed > 0 && <span className="text-[11px] font-medium text-accent-deep">{changed} changed</span>}
        <span className="font-mono text-[11px] text-zinc-400">{group.fields.length}</span>
      </summary>
      <div className="border-t border-zinc-100 bg-zinc-50/70 px-3 py-1.5">
        {group.doc && <p className="py-1 text-[11px] text-zinc-400">{group.doc}</p>}
        {group.fields.map((f) => (
          <FieldRow
            key={f.name}
            field={f}
            groupName={group.name}
            value={values[f.name]}
            onChange={(v) => onChange(f.name, v)}
          />
        ))}
        {changed > 0 && (
          <button
            onClick={onReset}
            className="my-1.5 flex items-center gap-1 text-[11px] text-zinc-500 transition-colors hover:text-zinc-900"
          >
            <RotateCcw size={12} /> Reset group
          </button>
        )}
      </div>
    </details>
  )
}

interface RowProps {
  field: ParamField
  groupName: string
  value: unknown // override, or undefined when the default applies
  onChange: (value: unknown | undefined) => void
}

function FieldRow({ field, groupName, value, onChange }: RowProps) {
  const overridden = value !== undefined
  const current = overridden ? value : field.default
  const choices = CHOICES[`${groupName}.${field.name}`]
  // Inputs are uncontrolled and commit on blur / Enter; the key remounts them when
  // the override is cleared so the default shows again.
  const key = `${field.name}:${overridden ? 'override' : 'default'}`
  const commit = (next: unknown) => onChange(sameValue(next, field.default) ? undefined : next)
  const blurOnEnter = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
  }

  const name = (
    <span className="min-w-0">
      <span className="flex items-center gap-1.5 font-mono text-[12px] text-zinc-800">
        {overridden && <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent" aria-hidden />}
        <span className="truncate">{field.name}</span>
      </span>
      {meta(field) && <span className="block text-[10.5px] text-zinc-400">{meta(field)}</span>}
    </span>
  )

  if (field.type === 'array') {
    const list = Array.isArray(current) ? current : []
    return (
      <label className="block py-1.5" title={field.description ?? undefined}>
        {name}
        <input
          key={key}
          type="text"
          defaultValue={list.join(', ')}
          onBlur={(e) => commit(e.target.value.trim() === '' ? field.default : parseList(e.target.value))}
          onKeyDown={blurOnEnter}
          className={`${FIELD_INPUT} mt-1 font-mono text-[11px]`}
        />
      </label>
    )
  }

  let input: React.ReactNode
  if (choices) {
    input = (
      <select value={String(current)} onChange={(e) => commit(e.target.value)} className={FIELD_INPUT}>
        {choices.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    )
  } else if (field.type === 'number' || field.type === 'integer') {
    input = (
      <input
        key={key}
        type="number"
        step={field.type === 'integer' ? 1 : 'any'}
        min={field.minimum ?? undefined}
        max={field.maximum ?? field.exclusiveMaximum ?? undefined}
        defaultValue={current == null ? '' : String(current)}
        placeholder={field.default == null ? 'auto' : undefined}
        onBlur={(e) => commit(e.target.value === '' ? field.default : Number(e.target.value))}
        onKeyDown={blurOnEnter}
        className={`${FIELD_INPUT} font-mono`}
      />
    )
  } else {
    input = (
      <input
        key={key}
        type="text"
        defaultValue={current == null ? '' : String(current)}
        onBlur={(e) => commit(e.target.value)}
        onKeyDown={blurOnEnter}
        className={FIELD_INPUT}
      />
    )
  }

  return (
    <label className="grid grid-cols-[1fr_7.5rem] items-center gap-3 py-1.5" title={field.description ?? undefined}>
      {name}
      {input}
    </label>
  )
}
