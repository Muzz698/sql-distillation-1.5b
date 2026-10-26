"""Command-line entry point. Phase commands are registered here as they are built."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.logging import RichHandler
from rich.table import Table

from distill_sql.config import PROJECT_ROOT, AppConfig, load_config

app = typer.Typer(no_args_is_help=True, help="Distill-SQL pipeline commands.")
data_app = typer.Typer(no_args_is_help=True, help="Phase 1: datasets, schemas, contamination.")
app.add_typer(data_app, name="data")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="YAML config file.")]
DEFAULT_CONFIG = Path("configs/base.yaml")


def _cfg(path: Path) -> AppConfig:
    return load_config(path if path.is_absolute() else PROJECT_ROOT / path)


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    """Distill-SQL pipeline commands."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        handlers=[RichHandler(show_path=False)],
    )


@app.command("show-config")
def show_config(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Validate a config file and print the resolved values."""
    rprint(_cfg(config).model_dump(mode="json"))


@data_app.command("download")
def data_download(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Download and extract every dataset archive (checksum-verified, idempotent)."""
    from distill_sql.data.download import download, safe_extract

    cfg = _cfg(config)
    for name, ds in cfg.datasets.items():
        archive = download(ds.source, cfg.paths.raw_dir / "downloads")
        safe_extract(archive, cfg.paths.raw_dir / ds.source.extract_to)
        rprint(f"[green]✓[/] {name}: {archive.name} ({ds.license})")


@data_app.command("prepare")
def data_prepare(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Extract schemas, normalise questions and build the train/eval database registry."""
    from distill_sql.data.prepare import prepare

    cfg = _cfg(config)
    registry = prepare(cfg, PROJECT_ROOT)
    report = cfg.paths.processed_dir / "reports" / "data_report.md"
    rprint(report.read_text())
    rprint(
        f"[green]✓[/] {len(registry.databases)} databases registered "
        f"→ {cfg.paths.processed_dir.relative_to(PROJECT_ROOT)}"
    )


@data_app.command("check-contamination")
def data_check_contamination(
    config: ConfigOpt = DEFAULT_CONFIG,
    include: Annotated[
        list[str] | None,
        typer.Option(help="Extra JSONL glob(s) to treat as training data (repeatable)."),
    ] = None,
) -> None:
    """Fail (exit 1) if any eval database, schema or question leaks into training data."""
    from distill_sql.data import contamination
    from distill_sql.data.datasets import Example, Registry
    from distill_sql.utils.io import iter_jsonl, read_json, write_json

    cfg = _cfg(config)
    processed = cfg.paths.processed_dir
    registry_file = processed / "registry.json"
    if not registry_file.exists():
        rprint("[red]No registry found. Run `make data` first.[/]")
        raise typer.Exit(2)
    registry = Registry.model_validate(read_json(registry_file))

    eval_examples = [
        Example.model_validate(row)
        for name, ds in cfg.datasets.items()
        for split in ds.eval_splits
        for row in iter_jsonl(processed / "examples" / name / f"{split}.jsonl")
    ]
    globs = [*cfg.contamination.train_globs, *(include or [])]
    train_rows = contamination.load_train_rows(globs, PROJECT_ROOT)
    report = contamination.check(registry, eval_examples, train_rows, cfg.contamination)
    write_json(processed / "reports" / "contamination.json", report.to_dict())

    table = Table(title="Contamination check")
    table.add_column("Check")
    table.add_column("Value", justify="right")
    table.add_row("train DBs / eval DBs", f"{report.n_train_dbs} / {report.n_eval_dbs}")
    table.add_row("training rows checked", f"{report.n_train_rows:,}")
    table.add_row("eval questions", f"{report.n_eval_questions:,}")
    for rule, n in sorted(report.counts.items()):
        table.add_row(f"[red]{rule}[/]", str(n))
    rprint(table)
    if report.n_train_rows == 0:
        rprint("[yellow]No training rows yet (expected before Phase 2). DB-level checks ran.[/]")
    if not report.passed:
        for rule, examples in report.examples.items():
            rprint(f"[red]{rule}[/] (first {len(examples)}):")
            for ex in examples[:5]:
                rprint(f"  {ex}")
        rprint("[red bold]✗ contamination check FAILED[/]")
        raise typer.Exit(1)
    rprint("[green bold]✓ contamination check passed[/]")


if __name__ == "__main__":
    app()
