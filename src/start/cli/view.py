import asyncio
import json

from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.text import Text

from start.telemetry.bus import TelemetryBus, TelemetryEvent

# Unified Design Tokens for the StART Agent Committee
AGENT_COLOR_REGISTRY = {
    "DatasetDiscoveryAgent": "cyan",
    "TaskInferenceAgent": "orange1",
    "FeatureEngineeringAgent": "green",
    "ArchitectureReviewAgent": "magenta",
    "HyperparameterTuningAgent": "gold1",
    "ModelExecutionAgent": "deep_pink1",
    "ExplainabilityAgent": "spring_green1",
    "SensitivityAgent": "purple",
    "OverfittingAgent": "salmon",
    "ValidationAgent": "yellow",
    "GovernanceSignoffAgent": "bold blue",
    "EvidenceCriticAgent": "bold white",
    # Backward compatibility mappings for plain short-names
    "Dataset Discovery Agent": "cyan",
    "Task Inference Agent": "orange1",
    "Feature Engineering Agent": "green",
    "Architecture Review Agent": "magenta",
    "Hyperparameter Tuning Agent": "gold1",
    "Model Execution Agent": "deep_pink1",
    "Explainability Agent": "spring_green1",
    "Sensitivity Agent": "purple",
    "Overfitting Agent": "salmon",
    "Validation Agent": "yellow",
    "Governance Signoff Agent": "bold blue",
    "Evidence Critic Agent": "bold white",
}


def get_styled_agent_name(agent_name: str) -> Text:
    """Returns a Rich Text block with the agent name locked to its designated color identity."""
    # Strip whitespace or baseline trailing 'Agent' to ensure clean string matching
    clean_key = agent_name.strip()
    color = AGENT_COLOR_REGISTRY.get(clean_key, "white")
    return Text(clean_key, style=color)


def get_ansi_agent_name(agent_name: str) -> str:
    """Returns an ANSI-colorized string representation of the agent name for terminal prompt usage."""
    clean_key = agent_name.strip()
    color = AGENT_COLOR_REGISTRY.get(clean_key, "white")
    from rich.console import Console

    c = Console(color_system="standard", force_terminal=True)
    with c.capture() as capture:
        c.print(clean_key, style=color, end="")
    return capture.get()


class ProgressDashboardUI:
    def __init__(self, telemetry_bus: TelemetryBus):
        self.bus = telemetry_bus
        self.bus.subscribe(self.handle_telemetry_event)
        self.global_pct = 0.0
        self.stage_pct = 0.0
        self.active_stage = "Initialization"
        self.active_agent = "System Coordinator"
        self.status_msg = "Booting Engine"
        self.logs: list[str] = []

    def handle_telemetry_event(self, event: TelemetryEvent) -> None:
        self.active_stage = event.stage
        self.active_agent = event.agent_name
        self.status_msg = event.status_msg
        self.stage_pct = event.progress_percentage
        if event.trace_details:
            from start.cli.view import AGENT_COLOR_REGISTRY

            color = AGENT_COLOR_REGISTRY.get(event.agent_name, "white")
            log_line = f"[[{color}]{event.agent_name}[/{color}]] {event.trace_details.reasoning_step} (Conf: {event.trace_details.confidence_score})"
            if log_line not in self.logs:
                self.logs.append(log_line)

    def set_global_progress(self, percentage: float) -> None:
        self.global_pct = percentage

    def construct_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="stage", size=3),
            Layout(name="body", size=6),
            Layout(name="logs", size=8),
        )

        # Header global component configuration
        g_prog = Progress(
            TextColumn("[bold blue]{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%")
        )
        g_prog.add_task("Global Evaluation Progress", total=100, completed=int(self.global_pct))
        layout["header"].update(Panel(g_prog, title="Global Engine Vector"))

        # Stage component configuration
        s_prog = Progress(
            TextColumn("[bold green]{task.description}"), BarColumn(), TextColumn("{task.percentage:>3.0f}%")
        )
        s_prog.add_task(f"Stage: {self.active_stage}", total=100, completed=int(self.stage_pct))
        layout["stage"].update(Panel(s_prog, title="Pipeline Stage Vector"))

        # Agent tracking metadata panel layout
        body_text = Text()
        body_text.append("Active Actor Node: ", style="bold yellow")
        body_text.append(get_styled_agent_name(self.active_agent))
        body_text.append("\n")
        body_text.append("Current Objective: ", style="bold white")
        body_text.append(f"{self.status_msg}\n")

        snap = self.bus.fetch_agent_snapshot(self.active_agent)
        if snap and snap.metrics:
            body_text.append(f"Telemetry Footprint: {json.dumps(snap.metrics)}", style="italic cyan")
        layout["body"].update(Panel(body_text, title="Agent Execution Window"))

        # Scrolling Agent Action logs block panel mapping logic
        log_text = Text.from_markup("\n".join(self.logs[-6:]))
        layout["logs"].update(Panel(log_text, title="Meticulous Agent Action Log"))
        return layout

    async def monitor_execution(self, workflow_coro):
        with Live(self.construct_layout(), refresh_per_second=12, screen=False) as live:
            task = asyncio.create_task(workflow_coro)
            while not task.done():
                await asyncio.sleep(0.083)  # Render frequency clock locked at exactly 12Hz frame cycles
                live.update(self.construct_layout())
            return await task
