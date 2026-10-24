import { afterEach, describe, expect, it, vi } from 'vitest'
import { PublicStARTBackend } from '../adapters/public/PublicStARTBackend'
import type { RunRequest } from '../contracts/types'
import { TurnstileClient, type TurnstileApi } from './TurnstileClient'

const request: RunRequest = {
  workflowId: 'predictive_ml',
  contextId: 'institutional_credit_v1',
  goal: 'Protected deterministic run',
  parameters: {},
}

function successfulRun(runId = 'RUN-1') {
  return new Response(JSON.stringify({success:true,data:{run:{run_id:runId,workflow:'predictive_ml',phase:'running',plan:[]}}}), {
    status: 200,
    headers: {'content-type':'application/json'},
  })
}

function mockedTurnstile(tokens: string[]) {
  let callbacks: Parameters<TurnstileApi['render']>[1] | null = null
  const reset = vi.fn()
  const api: TurnstileApi = {
    render: vi.fn((_container, options) => { callbacks = options; return 'widget-1' }),
    execute: vi.fn(() => callbacks?.callback(tokens.shift() ?? '')),
    reset,
  }
  vi.stubGlobal('window', {turnstile:api})
  const client = new TurnstileClient('SITE-KEY', {
    loadScript: async () => {},
    getApi: () => window.turnstile,
    createContainer: () => ({}) as HTMLElement,
  })
  return {api, client, reset, callbacks: () => callbacks!}
}

afterEach(() => vi.unstubAllGlobals())

describe('Turnstile protected run lifecycle', () => {
  it('attaches a fresh token and never reuses it across sequential protected operations', async () => {
    const turnstile = mockedTurnstile(['TOKEN-1', 'TOKEN-2'])
    const payloads: Record<string, unknown>[] = []
    vi.stubGlobal('fetch', vi.fn(async (_url, init) => {
      payloads.push(JSON.parse(String(init?.body)))
      return successfulRun(`RUN-${payloads.length}`)
    }))
    const backend = new PublicStARTBackend('', turnstile.client)

    await backend.createRun({...request, turnstile_token:'STALE-TOKEN'})
    await backend.createRun(request)

    expect(payloads.map(payload => payload.turnstile_token)).toEqual(['TOKEN-1', 'TOKEN-2'])
    expect(payloads[0].turnstile_token).not.toBe(payloads[1].turnstile_token)
    expect(turnstile.reset).toHaveBeenCalledTimes(2)
  })

  it('resets after backend success and backend failure', async () => {
    const turnstile = mockedTurnstile(['TOKEN-1', 'TOKEN-2'])
    const fetch = vi.fn()
      .mockResolvedValueOnce(successfulRun())
      .mockResolvedValueOnce(new Response(JSON.stringify({success:false,error_code:'TURNSTILE_FAILED',error:'rejected'}), {status:400}))
    vi.stubGlobal('fetch', fetch)
    const backend = new PublicStARTBackend('', turnstile.client)

    await backend.createRun(request)
    await expect(backend.createRun(request)).rejects.toThrow('Please retry execution')
    expect(turnstile.reset).toHaveBeenCalledTimes(2)
  })

  it('clears an expired token and waits for a replacement before submission', async () => {
    const turnstile = mockedTurnstile([])
    const submit = vi.fn(async () => 'submitted')
    vi.mocked(turnstile.api.execute).mockImplementationOnce(() => turnstile.callbacks()['expired-callback']())

    await expect(turnstile.client.withFreshToken(submit)).rejects.toThrow('expired')
    expect(submit).not.toHaveBeenCalled()

    vi.mocked(turnstile.api.execute).mockImplementationOnce(() => turnstile.callbacks().callback('TOKEN-REPLACEMENT'))
    await expect(turnstile.client.withFreshToken(submit)).resolves.toBe('submitted')
    expect(submit).toHaveBeenCalledWith('TOKEN-REPLACEMENT')
  })

  it('surfaces challenge errors as recoverable and permits a later retry', async () => {
    const turnstile = mockedTurnstile([])
    const submit = vi.fn(async () => 'submitted')
    vi.mocked(turnstile.api.execute).mockImplementationOnce(() => turnstile.callbacks()['error-callback']('internal-error'))
    await expect(turnstile.client.withFreshToken(submit)).rejects.toThrow('Please retry')
    expect(submit).not.toHaveBeenCalled()

    vi.mocked(turnstile.api.execute).mockImplementationOnce(() => turnstile.callbacks().callback('TOKEN-AFTER-ERROR'))
    await expect(turnstile.client.withFreshToken(submit)).resolves.toBe('submitted')
  })

  it('fails closed when the production site key is missing', async () => {
    const api: TurnstileApi = {render:vi.fn(),execute:vi.fn(),reset:vi.fn()}
    const client = new TurnstileClient('', {
      loadScript: async () => {},
      getApi: () => api,
      createContainer: () => ({}) as HTMLElement,
    })
    await expect(client.withFreshToken(async () => 'submitted')).rejects.toThrow('site key is not configured')
    expect(api.execute).not.toHaveBeenCalled()
  })

  it('reports script loading failure without submitting an unprotected request', async () => {
    const submit = vi.fn(async () => 'submitted')
    const client = new TurnstileClient('SITE-KEY', {
      loadScript: async () => { throw new Error('offline') },
      getApi: () => undefined,
      createContainer: () => ({}) as HTMLElement,
    })
    await expect(client.withFreshToken(submit)).rejects.toThrow('could not be loaded')
    expect(submit).not.toHaveBeenCalled()
  })
})
