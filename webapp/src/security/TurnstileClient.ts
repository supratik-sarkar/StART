const TURNSTILE_SCRIPT_URL = 'https://challenges.cloudflare.com/turnstile/v0/api.js?render=explicit'

export interface TurnstileApi {
  render(container: HTMLElement, options: {
    sitekey: string
    action?: string
    execution?: 'render' | 'execute'
    appearance?: 'always' | 'execute' | 'interaction-only'
    retry?: 'auto' | 'never'
    callback: (token: string) => void
    'expired-callback': () => void
    'error-callback': (code?: string) => void
  }): string
  execute(widgetId: string): void
  reset(widgetId: string): void
}

export interface TurnstileProtection {
  withFreshToken<T>(submit: (token: string) => Promise<T>): Promise<T>
}

export interface TurnstileClientDependencies {
  loadScript: () => Promise<void>
  getApi: () => TurnstileApi | undefined
  createContainer: () => HTMLElement
}

export class TurnstileClient implements TurnstileProtection {
  private widgetId: string | null = null
  private operation: Promise<unknown> | null = null
  private resolveToken: ((token: string) => void) | null = null
  private rejectToken: ((error: Error) => void) | null = null

  constructor(
    private readonly siteKey: string,
    private readonly dependencies: TurnstileClientDependencies = browserDependencies,
  ) {}

  async withFreshToken<T>(submit: (token: string) => Promise<T>): Promise<T> {
    if (this.operation) {
      throw new Error('Execution verification is already in progress. Please wait and retry.')
    }
    if (!this.siteKey.trim()) {
      throw new Error('Execution verification is unavailable because the Turnstile site key is not configured.')
    }

    const operation = this.runProtected(submit)
    this.operation = operation
    try {
      return await operation
    } finally {
      this.operation = null
    }
  }

  private async runProtected<T>(submit: (token: string) => Promise<T>): Promise<T> {
    try {
      const token = await this.acquireToken()
      // Once this callback starts, the token is spent even when the request fails.
      return await submit(token)
    } finally {
      this.clearPendingToken()
      this.resetWidget()
    }
  }

  private async acquireToken(): Promise<string> {
    await this.initialize()
    const api = this.dependencies.getApi()
    if (!api || !this.widgetId) {
      throw new Error('Execution verification did not initialize. Please retry.')
    }

    return new Promise<string>((resolve, reject) => {
      this.resolveToken = resolve
      this.rejectToken = reject
      try {
        api.execute(this.widgetId!)
      } catch {
        this.clearPendingToken()
        reject(new Error('Execution verification could not start. Please retry.'))
      }
    })
  }

  private async initialize(): Promise<void> {
    if (this.widgetId) return
    try {
      await this.dependencies.loadScript()
    } catch {
      throw new Error('Execution verification could not be loaded. Check your connection and retry.')
    }
    const api = this.dependencies.getApi()
    if (!api) throw new Error('Execution verification is unavailable. Please retry.')

    try {
      this.widgetId = api.render(this.dependencies.createContainer(), {
        sitekey: this.siteKey,
        action: 'workflow_run',
        execution: 'execute',
        appearance: 'interaction-only',
        retry: 'never',
        callback: token => {
          const resolve = this.resolveToken
          const reject = this.rejectToken
          this.clearPendingToken()
          if (token && resolve) resolve(token)
          else reject?.(new Error('Execution verification returned no token. Please retry.'))
        },
        'expired-callback': () => this.rejectCurrent(new Error('Execution verification expired. Please retry.')),
        'error-callback': () => this.rejectCurrent(new Error('Execution verification failed. Please retry.')),
      })
    } catch {
      throw new Error('Execution verification could not initialize. Please retry.')
    }
  }

  private rejectCurrent(error: Error): void {
    const reject = this.rejectToken
    this.clearPendingToken()
    reject?.(error)
  }

  private clearPendingToken(): void {
    this.resolveToken = null
    this.rejectToken = null
  }

  private resetWidget(): void {
    const api = this.dependencies.getApi()
    if (api && this.widgetId) {
      try { api.reset(this.widgetId) } catch {}
    }
  }
}

function loadTurnstileScript(): Promise<void> {
  if (window.turnstile) return Promise.resolve()
  const existing = document.querySelector<HTMLScriptElement>('script[data-start-turnstile]')
  if (existing) {
    if (existing.dataset.loadState === 'loaded') {
      existing.remove()
      return loadTurnstileScript()
    }
    return new Promise((resolve, reject) => {
      existing.addEventListener('load', () => resolve(), {once:true})
      existing.addEventListener('error', () => {
        existing.remove()
        reject(new Error('Turnstile script failed'))
      }, {once:true})
    })
  }
  return new Promise((resolve, reject) => {
    const script = document.createElement('script')
    script.src = TURNSTILE_SCRIPT_URL
    script.async = true
    script.defer = true
    script.dataset.startTurnstile = 'true'
    script.dataset.loadState = 'loading'
    script.addEventListener('load', () => {
      script.dataset.loadState = 'loaded'
      resolve()
    }, {once:true})
    script.addEventListener('error', () => {
      script.remove()
      reject(new Error('Turnstile script failed'))
    }, {once:true})
    document.head.appendChild(script)
  })
}

function createTurnstileContainer(): HTMLElement {
  const container = document.createElement('div')
  container.setAttribute('aria-label', 'Execution verification')
  container.style.position = 'fixed'
  container.style.right = '16px'
  container.style.bottom = '16px'
  container.style.zIndex = '1000'
  document.body.appendChild(container)
  return container
}

const browserDependencies: TurnstileClientDependencies = {
  loadScript: loadTurnstileScript,
  getApi: () => window.turnstile,
  createContainer: createTurnstileContainer,
}

export function createBrowserTurnstileClient(siteKey: string): TurnstileProtection {
  return new TurnstileClient(siteKey)
}
