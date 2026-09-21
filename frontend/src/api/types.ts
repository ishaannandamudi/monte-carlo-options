export type JsonPrimitive = boolean | number | string | null
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[]

export interface JsonObject {
  [key: string]: JsonValue
}

export function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function parseJsonObject(source: string): JsonObject {
  const parsed: unknown = JSON.parse(source)
  if (!isJsonObject(parsed)) throw new Error('Request payload must be a JSON object.')
  return parsed
}

export function prettyJson(value: JsonValue): string {
  return JSON.stringify(value, null, 2)
}
