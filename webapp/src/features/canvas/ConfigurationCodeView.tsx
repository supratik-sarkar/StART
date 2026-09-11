import { useState } from 'react'
import { CopyValue, Empty } from '../investigation/Science'

export function pythonLiteral(value: unknown, depth = 0): string {
  const indent = '    '.repeat(depth), next = '    '.repeat(depth+1)
  if (value == null) return 'None'
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'string') return JSON.stringify(value)
  if (typeof value === 'number') return String(value)
  if (Array.isArray(value)) return `[${value.map(v => pythonLiteral(v,depth)).join(', ')}]`
  const pairs = Object.entries(value as object)
  return pairs.length ? `{\n${pairs.map(([k,v]) => `${next}${JSON.stringify(k)}: ${pythonLiteral(v,depth+1)},`).join('\n')}\n${indent}}` : '{}'
}
export function ConfigurationCodeView({ rawConfig }: { workflowId?: string; contextId?: string; runId?: string; parameters?: Record<string, any>; rawConfig?: unknown }) {
  const [tab,setTab] = useState('python')
  if (!rawConfig || !Object.keys(rawConfig).length) return <Empty title="Resolved configuration not supplied"/>
  const code = tab === 'python' ? `review_config = ${pythonLiteral(rawConfig)}` : JSON.stringify(rawConfig,null,2)
  const tokens = code.split(/("(?:[^"\\]|\\.)*"|\b(?:True|False|None)\b|\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)/g)
  return <div className="configuration-code-view"><div className="code-toolbar"><div><button aria-pressed={tab==='python'} onClick={() => setTab('python')}>Python</button><button aria-pressed={tab==='raw'} onClick={() => setTab('raw')}>Raw</button></div><CopyValue value={code} truncate/></div><pre className="scientific-code"><code>{tokens.map((t,i) => <span key={i} className={t.startsWith('"') ? 'syntax-string' : /^(True|False|None)$/.test(t) ? 'syntax-keyword' : /^\d/.test(t) ? 'syntax-number' : ''}>{t}</span>)}</code></pre></div>
}
