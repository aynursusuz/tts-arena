"""CLI for tts-bench."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="tts-bench",
    help="Run and compare open-source TTS models.",
    add_completion=False,
)
console = Console()


@app.command()
def list_engines() -> None:
    """List all available TTS engines."""
    from tts_bench.engines import list_engines as _list

    engines = _list()
    table = Table(title="Available TTS Engines")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")
    table.add_column("Languages", style="green")
    table.add_column("Clone", style="yellow")
    table.add_column("Stream", style="yellow")
    table.add_column("License", style="dim")

    for e in engines:
        table.add_row(
            e["name"],
            e["description"],
            ", ".join(e["languages"][:3]) + ("..." if len(e["languages"]) > 3 else ""),
            "Yes" if e["voice_cloning"] else "No",
            "Yes" if e["streaming"] else "No",
            e["license"],
        )
    console.print(table)


@app.command()
def synthesize(
    text: Annotated[str, typer.Argument(help="Text to synthesize")],
    engine: Annotated[str, typer.Option("--engine", "-e", help="TTS engine name")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output WAV file")] = Path(
        "output.wav"
    ),
    device: Annotated[str, typer.Option("--device", "-d", help="Device: auto, cuda, cpu")] = "auto",
) -> None:
    """Synthesize speech from text with the given engine."""
    from tts_bench.engines import get_engine

    console.print(f"[cyan]Loading engine:[/cyan] {engine}")
    tts = get_engine(engine, device=device)
    tts.ensure_loaded()

    console.print(f"[cyan]Synthesizing:[/cyan] {text[:80]}...")
    result = tts.synthesize_to_file(text, output)

    console.print(f"[green]Done![/green] Saved to {output}")
    console.print(f"  Duration: {result.duration_seconds:.2f}s")
    console.print(f"  Inference: {result.inference_time_seconds:.2f}s")
    console.print(f"  RTF: {result.real_time_factor:.3f}x")


@app.command()
def benchmark(
    engines: Annotated[
        list[str] | None,
        typer.Option("--engine", "-e", help="Engines to benchmark (repeatable)"),
    ] = None,
    text: Annotated[str | None, typer.Option("--text", "-t", help="Text to benchmark with")] = None,
    results_dir: Annotated[Path, typer.Option("--results-dir")] = Path("benchmarks/results"),
    samples_dir: Annotated[Path, typer.Option("--samples-dir")] = Path("benchmarks/audio_samples"),
    device: Annotated[str, typer.Option("--device", "-d")] = "auto",
) -> None:
    """Run benchmarks across the selected TTS engines."""
    from tts_bench.benchmarks.runner import DEFAULT_TEXT, run_benchmark

    run_benchmark(
        engine_names=engines,
        text=text or DEFAULT_TEXT,
        results_dir=results_dir,
        samples_dir=samples_dir,
        device=device,
    )


if __name__ == "__main__":
    app()
