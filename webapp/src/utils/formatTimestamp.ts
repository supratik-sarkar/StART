/**
 * Authoritative Canonical Timestamp Formatter for StART Frontend.
 *
 * Guarantees:
 * 1. Zero "Invalid Date" strings rendered in UI.
 * 2. Zero "NaN" values.
 * 3. Handles Unix epoch seconds, epoch milliseconds, and ISO date strings.
 * 4. Safe fallback for undefined/null/unparseable values.
 */

export function parseTimestamp(ts: number | string | null | undefined): Date | null {
  if (ts == null || ts === '' || ts === 'undefined' || ts === 'null') {
    return null
  }

  if (typeof ts === 'number') {
    if (isNaN(ts) || !isFinite(ts) || ts <= 0) return null
    // If timestamp is in seconds (< 1e11), convert to milliseconds
    const ms = ts < 1e11 ? ts * 1000 : ts
    const d = new Date(ms)
    return isNaN(d.getTime()) ? null : d
  }

  // String parsing
  const numeric = Number(ts)
  if (!isNaN(numeric) && isFinite(numeric) && numeric > 0 && /^\d+$/.test(ts.trim())) {
    const ms = numeric < 1e11 ? numeric * 1000 : numeric
    const d = new Date(ms)
    return isNaN(d.getTime()) ? null : d
  }

  const d = new Date(ts)
  return isNaN(d.getTime()) ? null : d
}

export function formatTimeOnly(
  ts: number | string | null | undefined,
  fallback: string = '—'
): string {
  const d = parseTimestamp(ts)
  if (!d) return fallback
  try {
    return d.toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return fallback
  }
}

export function formatDateTime(
  ts: number | string | null | undefined,
  fallback: string = '—'
): string {
  const d = parseTimestamp(ts)
  if (!d) return fallback
  try {
    return d.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return fallback
  }
}
