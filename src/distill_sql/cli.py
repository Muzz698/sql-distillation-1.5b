"""Command-line entry point. Phase commands are registered here as they are built."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint
from rich.logging import RichHandler
from rich.table import Table

from distill_sql.config import PROJECT_ROOT, AppConfig, Secrets, TeacherConfig, load_config

app = typer.Typer(no_args_is_help=True, help="Distill-SQL pipeline commands.")
data_app = typer.Typer(no_args_is_help=True, help="Phase 1: datasets, schemas, contamination.")
app.add_typer(data_app, name="data")
gen_app = typer.Typer(no_args_is_help=True, help="Phase 2: teacher generation + verification.")
app.add_typer(gen_app, name="gen")

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


# ---- Phase 2 -----------------------------------------------------------------
TeacherOpt = Annotated[
    list[str] | None,
    typer.Option("--teacher", "-t", help="Teacher name(s) from the config. Default: all."),
]
MaxCallsOpt = Annotated[
    int | None, typer.Option(help="Stop each teacher after N API calls (for pilots).")
]


def _teachers(cfg: AppConfig, names: list[str] | None) -> list[TeacherConfig]:
    return [cfg.teacher(n) for n in names] if names else list(cfg.teachers)


@gen_app.command("models")
def gen_models(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """List the model ids each provider key can use (to fill in `teachers:`)."""
    from distill_sql.generation.pipeline import build_context, make_client

    cfg = _cfg(config)
    ctx = build_context(cfg, PROJECT_ROOT)
    seen: set[str] = set()
    for teacher in cfg.teachers:
        if teacher.provider in seen:
            continue
        seen.add(teacher.provider)
        try:
            models = make_client(ctx, teacher, Secrets()).list_models()
        except Exception as exc:
            rprint(f"[red]{teacher.provider}: {exc}[/]")
            continue
        rprint(f"[bold]{teacher.provider}[/] ({len(models)} models)")
        for m in models:
            used = any(t.model == m.removeprefix("models/") for t in cfg.teachers)
            rprint(f"  {'[green]✓[/] ' if used else '  '}{m}")


@gen_app.command("run")
def gen_run(
    config: ConfigOpt = DEFAULT_CONFIG,
    teacher: TeacherOpt = None,
    max_calls: MaxCallsOpt = None,
    dry_run: Annotated[
        bool, typer.Option(help="Print a prompt and a budget; no API calls.")
    ] = False,
) -> None:
    """Generate question/SQL pairs with the teachers (resumable; safe to re-run daily)."""
    from distill_sql.generation.generate import estimate_budget
    from distill_sql.generation.pipeline import build_context, generate, teacher_tasks
    from distill_sql.utils.runs import Run

    cfg = _cfg(config)
    ctx = build_context(cfg, PROJECT_ROOT)
    teachers = _teachers(cfg, teacher)
    if dry_run:
        tasks = teacher_tasks(ctx, teachers[0])
        rprint("[bold]Example prompt[/] (first task):\n")
        for m in ctx.prompts.messages(tasks[0]):
            rprint(f"[cyan]--- {m['role']} ---[/]\n{m['content']}\n")
        table = Table(title="Full-pass budget estimate (per teacher)")
        for col in ("teacher", "calls", "~tok/call", "~calls/day", "~days", "raw pairs"):
            table.add_column(col)
        for t in teachers:
            b = estimate_budget(t, teacher_tasks(ctx, t), ctx.prompts, ctx.gen)
            table.add_row(
                t.name,
                str(b["calls"]),
                str(b["est_tokens_per_call"]),
                str(b["est_calls_per_day"]),
                str(b["est_days"]),
                f"{b['est_raw_pairs']:,}",
            )
        rprint(table)
        return

    run = Run(cfg, "generate", {"teachers": [t.name for t in teachers], "max_calls": max_calls})
    summaries = generate(ctx, teachers, Secrets(), max_calls)
    table = Table(title="Generation run")
    for col in (
        "teacher",
        "done before",
        "calls",
        "cached",
        "pairs",
        "parse err",
        "tokens",
        "stopped",
    ):
        table.add_column(col)
    for s in summaries:
        table.add_row(
            s.teacher,
            str(s.skipped_done),
            str(s.calls),
            str(s.cached),
            str(s.pairs),
            str(s.parse_errors),
            f"{s.tokens:,}",
            s.stopped_reason or "finished",
        )
    rprint(table)
    run.log_metrics({s.teacher: s.__dict__ for s in summaries})
    rprint("Next: [bold]make verify[/] (and re-run this command tomorrow if a daily limit hit).")


@gen_app.command("verify")
def gen_verify(config: ConfigOpt = DEFAULT_CONFIG) -> None:
    """Execute every generated pair, filter, dedupe, decontaminate; write sft.jsonl + report."""
    from distill_sql.generation.pipeline import build_context, verify
    from distill_sql.utils.runs import Run

    cfg = _cfg(config)
    ctx = build_context(cfg, PROJECT_ROOT)
    run = Run(cfg, "verify")
    stats = verify(ctx)
    run.log_metrics(stats)
    rprint((cfg.paths.processed_dir / "reports" / "generation_report.md").read_text())
    rprint(f"[green]✓[/] {stats['n_kept']:,} pairs → {ctx.sft_file.relative_to(PROJECT_ROOT)}")


@gen_app.command("crosscheck")
def gen_crosscheck(
    checker: Annotated[str, typer.Option(help="Teacher that re-answers the kept questions.")],
    config: ConfigOpt = DEFAULT_CONFIG,
    max_calls: MaxCallsOpt = None,
) -> None:
    """Have a second teacher answer kept questions; matching results mark `agreement`."""
    from distill_sql.generation.pipeline import build_context, crosscheck

    cfg = _cfg(config)
    ctx = build_context(cfg, PROJECT_ROOT)
    s = crosscheck(ctx, cfg.teacher(checker), Secrets(), max_calls)
    rate = f"{100 * s.agreed / s.checked:.1f}%" if s.checked else "-"
    rprint(f"{checker}: {s.checked} checked, {s.agreed} agree ({rate}), {s.calls} calls")
    if s.stopped_reason:
        rprint(f"[yellow]stopped: {s.stopped_reason}[/]")
    rprint("Next: [bold]make verify[/] to fold agreement into sft.jsonl and the report.")


if __name__ == "__main__":
    app()
