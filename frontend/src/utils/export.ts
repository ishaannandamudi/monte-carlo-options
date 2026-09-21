import type { JsonValue } from '../api/types'

export async function copyText(value: string): Promise<void> {
  await navigator.clipboard.writeText(value)
}

export function downloadJson(filename: string, value: JsonValue): void {
  const blob = new Blob([JSON.stringify(value, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}
