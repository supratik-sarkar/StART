import { describe, expect, it } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import { RunCompareView } from './RunCompareView'
import type { StartBackend } from '../../contracts/backend'

describe('run comparison identity', () => {
  it('does not offer a comparison of a run against itself', () => {
    const markup = renderToStaticMarkup(<RunCompareView isOpen onClose={() => {}} runAId="RUN-1" runBId="RUN-1" compareResult={null} loading={false} onSelectRunA={() => {}} onSelectRunB={() => {}} onExecuteCompare={() => {}} onLoadRun={() => {}} backend={{} as StartBackend}/> )
    expect(markup).toContain('Choose a different candidate run')
    expect(markup).toMatch(/<button[^>]*disabled=""[^>]*>Compare<\/button>/)
  })
})
