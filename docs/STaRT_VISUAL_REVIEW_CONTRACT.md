# StART Visual Review Contract

> Presentation semantics for terminal CLI, generated artifacts, and future web workstation.

## 1. Titles

- Every analytical output must have a descriptive title reflecting the analysis performed.
- Titles use sentence case.
- Titles must NOT include raw internal IDs unless the ID is the primary identifier.

## 2. Units

- All numerical results display units explicitly (e.g., "annualised %", "periodic", "USD").
- Percentages are displayed with their `%` symbol and appropriate decimal precision.
- Weight values are displayed as proportions (0–1) or percentages, consistently within a view.

## 3. Horizons

- Any time-dependent result (returns, volatility, VaR) displays the observation horizon.
- Annualisation convention is stated where applied (e.g., "252 trading days / year").

## 4. Evidence IDs

- Evidence IDs (EvidenceRecord `record_id`) are available in detail views.
- Summary views show abbreviated IDs (first 8 hex chars) to avoid visual clutter.
- Evidence IDs must never dominate the visual layout.

## 5. Semantic Payloads

- Diagnostic outputs carry structured `metrics` dictionaries.
- Narrative interpretation is separated from deterministic numerical evidence.
- Challenge → diagnostic → new evidence chains are traceable.

## 6. Legends

- Charts with multiple series include readable legends.
- Legends do not obscure data.
- Color assignments are deterministic and accessible.

## 7. Status Language

- `PASS` / `FAIL` / `RECORDED` / `SKIPPED` / `WARN` / `ERROR` — consistent across all surfaces.
- No pseudo-precision in status (e.g., "PASS with caveats" is not a status; use `RECORDED` + interpretation).
- Positive/negative/neutral status uses consistent color coding:
  - Green: PASS / favorable
  - Red: FAIL / unfavorable  
  - Amber/Yellow: WARN / caution
  - Gray: RECORDED / SKIPPED / neutral

## 8. Number Formatting

- Financial values: appropriate decimal places (typically 4–6 for ratios, 2 for percentages).
- No pseudo-precision: do not display 15 decimal places for a quantity with 4 significant figures.
- Use `round()` at the evidence boundary, not the display boundary.
- Large numbers use comma separators where readable.

## 9. Uncertainty and Limitations

- Every diagnostic that carries assumptions displays them.
- Bootstrap intervals display confidence level and method.
- Non-rejection caveats appear where statistically appropriate.
- Limitations are accessible but do not overwhelm the primary result.

## 10. Human-Review Readability

- Terminal output uses Rich panels, tables, and progress indicators.
- No giant walls of unformatted text.
- No raw developer dumps (tracebacks, debug logs) in public-facing output.
- Consistent typography and spacing within a session.
- Artifact panels on secondary displays use responsive layout.

## 11. Terminal Presentation

- Rich console: 256-color, Unicode box drawing.
- Panel headers: bold cyan for sections, bold green for success, bold red for failure.
- Tables: bordered, with header row styling.
- Progress: spinner or progress bar for long operations.

## 12. Generated Artifacts

- SVG preferred for vector graphics (charts, diagrams).
- HTML artifacts for interactive or complex visualizations.
- Deterministic content hash for reproducibility verification.
- Semantic data payload (JSON) accompanies every visual artifact.

## 13. Web Workstation Parity

- Future web frontend must implement the same semantic payload rendering.
- Same status colors, same number formatting, same evidence ID treatment.
- Responsive grid layout matching the terminal artifact board semantics.
- No information loss when transitioning from terminal to web surface.
