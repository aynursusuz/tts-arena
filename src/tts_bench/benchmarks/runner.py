"""Benchmark runner."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from tts_bench.engines import ENGINE_REGISTRY, get_engine

console = Console()


def run_benchmark(
    engine_names: list[str] | None = None,
    text: str = "The quick brown fox jumps over the lazy dog.",
    output_dir: str | Path = "benchmarks/results",
    device: str = "auto",
) -> list[dict[str, Any]]:
    """Run benchmarks across specified engines."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if engine_names is None:
        engine_names = list(ENGINE_REGISTRY.keys())

    if not engine_names:
        console.print("[red]No engines available. Install engine extras first.[/red]")
        return []

    results = []
    table = Table(title="TTS Benchmark Results")
    table.add_column("Engine", style="cyan", no_wrap=True)
    table.add_column("Duration (s)", justify="right")
    table.add_column("Inference (s)", justify="right")
    table.add_column("RTF", justify="right")
    table.add_column("VRAM (MB)", justify="right")
    table.add_column("Sample Rate", justify="right")
    table.add_column("Status", style="bold")

    for name in engine_names:
        console.print(f"\n[cyan]{'='*60}[/cyan]")
        console.print(f"[cyan]Benchmarking:[/cyan] {name}")
        console.print(f"[cyan]{'='*60}[/cyan]")

        try:
            engine = get_engine(name, device=device)

            # Measure load time
            load_start = time.perf_counter()
            engine.ensure_loaded()
            load_time = time.perf_counter() - load_start

            # Get VRAM after loading
            vram_after_load = engine.get_vram_usage_mb()

            # Run inference
            result = engine.synthesize(text)

            # Save audio
            audio_path = output_dir / f"{name}.wav"
            engine.synthesize_to_file(text, audio_path)

            # Get VRAM after inference
            vram_after_inference = engine.get_vram_usage_mb()

            entry = {
                "engine": name,
                "status": "success",
                "text": text,
                "duration_seconds": round(result.duration_seconds, 3),
                "inference_time_seconds": round(result.inference_time_seconds, 3),
                "real_time_factor": round(result.real_time_factor, 4),
                "load_time_seconds": round(load_time, 3),
                "sample_rate": result.sample_rate,
                "vram_after_load_mb": round(vram_after_load, 1) if vram_after_load else None,
                "vram_after_inference_mb": round(vram_after_inference, 1) if vram_after_inference else None,
                "audio_file": str(audio_path),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            table.add_row(
                name,
                f"{result.duration_seconds:.2f}",
                f"{result.inference_time_seconds:.2f}",
                f"{result.real_time_factor:.3f}",
                f"{vram_after_inference:.0f}" if vram_after_inference else "N/A",
                str(result.sample_rate),
                "[green]OK[/green]",
            )

            # Cleanup
            engine.unload_model()

        except Exception as e:
            entry = {
                "engine": name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            table.add_row(name, "-", "-", "-", "-", "-", f"[red]FAIL: {str(e)[:40]}[/red]")
            console.print(f"[red]Error with {name}: {e}[/red]")

        results.append(entry)

    # Print results table
    console.print("\n")
    console.print(table)

    # Save results to JSON
    results_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to:[/green] {results_file}")

    return results
