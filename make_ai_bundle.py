#!/usr/bin/env python3
"""
Build a shareable AI context bundle for this LightGBM/Freqtrade project.

The bundle is meant for sending code context to another AI assistant.  It includes
the research/backtest code, Freqtrade live strategy code, tests, configs, and key
Markdown notes.  It excludes private credentials, databases, logs, model artifacts,
market data, PDFs, and other heavy runtime outputs.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent

DEFAULT_INCLUDE_PATTERNS = (
    ".gitignore",
    "*.md",
    "*.py",
    "*.json",
    "requirements.txt",
    "src/*.py",
    "src/**/*.py",
    "tests/*.py",
    "tests/**/*.py",
    "freqtrade_live/*.py",
    "freqtrade_live/Dockerfile.*",
    "freqtrade_live/docker-compose.yml",
    "freqtrade_live/retrainer/*.py",
    "freqtrade_live/retrainer/**/*.py",
    "freqtrade_live/user_data/config.json",
    "freqtrade_live/user_data/config-private.json.template",
    "freqtrade_live/user_data/strategies/*.py",
    "freqtrade_live/user_data/strategies/**/*.py",
    "DOC/*.txt",
)

NOTEBOOK_PATTERNS = (
    "DOC/10_intraday_features.ipynb",
    "DOC/11_intraday_model.ipynb",
)

EXCLUDE_DIR_NAMES = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "data",
    "models",
    "good_models",
    "ai_bundles",
    "freqaimodels",
    "hyperopts",
    "hyperopt_results",
    "backtest_results",
    "logs",
}

EXCLUDE_PATTERNS = (
    "*.feather",
    "*.sqlite",
    "*.sqlite-*",
    "*.db",
    "*.log",
    "*.pdf",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.pem",
    "*.key",
    ".env",
    "grid_search_results.csv",
    "grid_search_output.log",
    "grid_search_resume.log",
    "YOUTUBE_VIDEO_SCRIPT.md",
    "freqtrade_live/shared/models/**",
    "freqtrade_live/user_data/config-private.json",
    "freqtrade_live/user_data/tradesv3.sqlite*",
    "freqtrade_live/user_data/logs/**",
    "freqtrade_live/user_data/data/**",
    "freqtrade_live/user_data/backtest_results/**",
    "freqtrade_live/user_data/freqaimodels/**",
    "freqtrade_live/user_data/hyperopts/**",
    "freqtrade_live/user_data/hyperopt_results/**",
    "freqtrade_live/user_data/notebooks/**",
    "freqtrade_live/user_data/plot/**",
)

SENSITIVE_KEYS = {
    "api_key",
    "api_secret",
    "secret",
    "secret_key",
    "jwt_secret_key",
    "key",
    "password",
    "passphrase",
    "private_key",
    "token",
    "chat_id",
}

SENSITIVE_LINE_RE = re.compile(
    r"(?im)^(\s*[\"']?(?:api[_-]?key|api[_-]?secret|key|secret|secret[_-]?key|"
    r"jwt[_-]?secret[_-]?key|password|passphrase|private[_-]?key|token|chat[_-]?id)"
    r"[\"']?\s*[:=]\s*)([\"']).*?\2(\s*,?\s*)$"
)


@dataclass
class BundleEntry:
    source_path: Path
    relative_path: str
    bundle_path: str
    content: str
    rendered_from_notebook: bool = False


def rel_posix(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def matches_include_pattern(path: str, pattern: str) -> bool:
    # A bare pattern like "*.py" means "repo-root Python files".  Patterns with
    # a slash are path globs such as "src/**/*.py".
    if "/" not in pattern:
        return "/" not in path and fnmatch.fnmatch(path, pattern)
    return fnmatch.fnmatch(path, pattern)


def matches_exclude_pattern(path: str, pattern: str) -> bool:
    # A bare exclude like "*.log" should match at any depth.
    if "/" not in pattern:
        return fnmatch.fnmatch(Path(path).name, pattern)
    return fnmatch.fnmatch(path, pattern)


def matches_any_include(path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(matches_include_pattern(path, pattern) for pattern in patterns)


def matches_any_exclude(path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    return any(matches_exclude_pattern(path, pattern) for pattern in patterns)


def is_excluded(path: Path) -> bool:
    rel = rel_posix(path)
    if any(part in EXCLUDE_DIR_NAMES for part in path.relative_to(ROOT).parts[:-1]):
        return True
    return matches_any_exclude(rel, EXCLUDE_PATTERNS)


def should_include(path: Path, extra_patterns: list[str], include_notebooks: bool) -> bool:
    if not path.is_file() or is_excluded(path):
        return False
    rel = rel_posix(path)
    if include_notebooks and matches_any_include(rel, NOTEBOOK_PATTERNS):
        return True
    return matches_any_include(rel, DEFAULT_INCLUDE_PATTERNS) or matches_any_include(rel, extra_patterns)


def sanitize_value(key: str, value: Any) -> Any:
    key_l = key.lower()
    if key_l in SENSITIVE_KEYS or any(mark in key_l for mark in ("secret", "password", "token")):
        if value in ("", None):
            return value
        return "<REDACTED>"
    if isinstance(value, dict):
        return {k: sanitize_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_value(key, item) for item in value]
    return value


def sanitize_text(path: Path, text: str) -> str:
    if path.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            sanitized = sanitize_value(path.name, parsed)
            return json.dumps(sanitized, indent=2, sort_keys=False) + "\n"
    return SENSITIVE_LINE_RE.sub(r"\1\2<REDACTED>\2\3", text)


def read_text_file(path: Path, max_file_bytes: int) -> str:
    size = path.stat().st_size
    if size > max_file_bytes:
        raise ValueError(f"file too large ({size} bytes > {max_file_bytes})")
    raw = path.read_bytes()
    return raw.decode("utf-8", errors="replace")


def render_notebook(path: Path, max_file_bytes: int) -> str:
    data = json.loads(read_text_file(path, max_file_bytes))
    lines = [
        f"# Notebook source: {rel_posix(path)}",
        "",
        "Outputs are omitted. Only Markdown and code cell sources are included.",
        "",
    ]
    for idx, cell in enumerate(data.get("cells", []), start=1):
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        cell_type = cell.get("cell_type", "unknown")
        if cell_type == "markdown":
            lines.extend([f"## Markdown cell {idx}", "", source.rstrip(), ""])
        elif cell_type == "code":
            lines.extend([f"## Code cell {idx}", "", "```python", source.rstrip(), "```", ""])
        else:
            lines.extend([f"## {cell_type.title()} cell {idx}", "", source.rstrip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def collect_entries(
    extra_patterns: list[str],
    include_notebooks: bool,
    max_file_bytes: int,
) -> tuple[list[BundleEntry], list[str]]:
    entries: list[BundleEntry] = []
    skipped: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not should_include(path, extra_patterns, include_notebooks):
            continue
        rel = rel_posix(path)
        try:
            if path.suffix.lower() == ".ipynb":
                content = render_notebook(path, max_file_bytes)
                bundle_path = rel + ".md"
                rendered = True
            else:
                content = sanitize_text(path, read_text_file(path, max_file_bytes))
                bundle_path = rel
                rendered = False
        except Exception as exc:  # noqa: BLE001 - report and keep bundling.
            skipped.append(f"{rel}: {exc}")
            continue
        entries.append(BundleEntry(path, rel, bundle_path, content, rendered))
    return entries, skipped


def language_for(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".py": "python",
        ".json": "json",
        ".yml": "yaml",
        ".yaml": "yaml",
        ".md": "markdown",
        ".txt": "text",
        ".dockerfile": "dockerfile",
    }.get(suffix, "")


def code_fence(content: str) -> str:
    longest = 2
    for match in re.finditer(r"`+", content):
        longest = max(longest, len(match.group(0)))
    return "`" * (longest + 1)


def build_markdown(entries: list[BundleEntry], skipped: list[str], args: argparse.Namespace) -> str:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = []
    for entry in entries:
        encoded = entry.content.encode("utf-8")
        manifest.append(
            {
                "path": entry.relative_path,
                "bundle_path": entry.bundle_path,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "notebook_render": entry.rendered_from_notebook,
            }
        )

    lines = [
        "# AI Context Bundle - LightGBM/Freqtrade Strategy",
        "",
        f"Generated UTC: {generated}",
        f"Project root: {ROOT}",
        "",
        "This bundle is generated for AI review. Secrets and private runtime artifacts are excluded or redacted.",
        "",
        "## Included Files",
        "",
    ]
    for item in manifest:
        suffix = " (notebook source render)" if item["notebook_render"] else ""
        lines.append(f"- `{item['bundle_path']}` - {item['bytes']} bytes{suffix}")

    if skipped:
        lines.extend(["", "## Skipped Files", ""])
        lines.extend(f"- {item}" for item in skipped)

    lines.extend(["", "## File Contents", ""])
    for entry in entries:
        lang = language_for(entry.bundle_path)
        fence = code_fence(entry.content)
        lines.extend(
            [
                f"### {entry.bundle_path}",
                "",
                f"{fence}{lang}",
                entry.content.rstrip(),
                fence,
                "",
            ]
        )

    lines.extend(
        [
            "## Manifest JSON",
            "",
            "```json",
            json.dumps(
                {
                    "generated_utc": generated,
                    "project_root": str(ROOT),
                    "include_notebooks": args.include_notebooks,
                    "max_file_bytes": args.max_file_bytes,
                    "files": manifest,
                    "skipped": skipped,
                },
                indent=2,
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(markdown: str, entries: list[BundleEntry], out_dir: Path, name: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{name}.md"
    zip_path = out_dir / f"{name}.zip"

    md_path.write_text(markdown, encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}.md", markdown)
        for entry in entries:
            zf.writestr(entry.bundle_path, entry.content)
    return md_path, zip_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="ai_bundles",
        help="Directory for generated bundle files. Default: ai_bundles",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output filename stem. Default includes UTC timestamp.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra glob pattern to include, relative to repo root. Can be repeated.",
    )
    parser.add_argument(
        "--no-notebooks",
        dest="include_notebooks",
        action="store_false",
        help="Do not include rendered notebook sources from DOC/*.ipynb.",
    )
    parser.set_defaults(include_notebooks=True)
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=2_000_000,
        help="Skip text files larger than this size. Default: 2000000.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be included without writing bundle files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = args.name or f"boosting_strategy_LightGBM_ai_bundle_{timestamp}"
    out_dir = (ROOT / args.output_dir).resolve()

    entries, skipped = collect_entries(args.extra, args.include_notebooks, args.max_file_bytes)
    if not entries:
        print("No files matched the bundle rules.", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"Would include {len(entries)} files:")
        for entry in entries:
            note = " (notebook render)" if entry.rendered_from_notebook else ""
            print(f"  {entry.bundle_path}{note}")
        if skipped:
            print("\nSkipped:")
            for item in skipped:
                print(f"  {item}")
        return 0

    markdown = build_markdown(entries, skipped, args)
    md_path, zip_path = write_outputs(markdown, entries, out_dir, name)
    print(f"Wrote Markdown bundle: {md_path}")
    print(f"Wrote zip bundle:      {zip_path}")
    print(f"Included files:        {len(entries)}")
    if skipped:
        print(f"Skipped files:         {len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
