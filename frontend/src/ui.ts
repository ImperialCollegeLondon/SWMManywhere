// Shared class strings so every control looks the same.
export const INPUT =
  'h-8 w-full min-w-0 rounded-md border border-zinc-200 bg-white px-2 text-[13px] text-zinc-800 outline-none transition-colors placeholder:text-zinc-300 focus:border-accent focus:ring-2 focus:ring-accent/25'

const BTN =
  'inline-flex h-9 items-center justify-center gap-2 rounded-md px-3 text-[13px] font-medium transition-all duration-200 active:scale-[0.98] disabled:pointer-events-none disabled:opacity-40'
export const BTN_PRIMARY = `${BTN} bg-accent-deep text-white hover:bg-accent-deeper`
export const BTN_SECONDARY = `${BTN} border border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-50`

export const LABEL = 'block text-[11px] font-medium text-zinc-500'
export const ICON_BTN = 'rounded p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700'
