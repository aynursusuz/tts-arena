"""CLI interface for TTS Arena."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="tts-arena",
    help="A unified toolkit for benchmarking all open-source TTS models.",
    add_completion=False,
)
console = Console()


@app.command()
def list_engines() -> None:
    """List all available TTS engines."""
    from tts_arena.engines import list_engines as _list

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
    text: str = typer.Argument(..., help="Text to synthesize"),
    engine: str = typer.Option(..., "--engine", "-e", help="TTS engine name"),
    output: Path = typer.Option("output.wav", "--output", "-o", help="Output WAV file path"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cuda, cpu"),
) -> None:
    """Synthesize speech from text using a specified engine."""
    from tts_arena.engines import get_engine

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
    engines: Optional[list[str]] = typer.Option(None, "--engine", "-e", help="Engines to benchmark (can repeat). Default: all"),
    text: str = typer.Option(
        "The quick brown fox jumps over the lazy dog. This is a benchmark test for text to speech synthesis quality and speed.",
        "--text", "-t",
        help="Text to benchmark with",
    ),
    output_dir: Path = typer.Option("benchmarks/results", "--output-dir", help="Output directory"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cuda, cpu"),
) -> None:
    """Run benchmarks across TTS engines."""
    from tts_arena.benchmarks.runner import run_benchmark

    run_benchmark(
        engine_names=engines,
        text=text,
        output_dir=output_dir,
        device=device,
    )


if __name__ == "__main__":
    app()
