# StART Greenfield Webapp

A portable, browser-native agentic engineering workbench for StART.

This package is intentionally **not wired to the existing public backend by default**. It starts with a clearly labeled deterministic preview adapter so the visual product can be reviewed independently. Production integration binds `PublicStARTBackend` to canonical StART routes.

## Run locally

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Adapter mode

Default: preview adapter.

For public backend integration:

```bash
VITE_START_ADAPTER=public
VITE_START_API_BASE=https://your-start-gateway.example
```

Read `docs/BACKEND_INTEGRATION_GUIDE.md` and `docs/BACKEND_WIRING_GUIDE.md` before integration.
