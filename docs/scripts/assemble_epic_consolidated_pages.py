#!/usr/bin/env python3
"""Assemble docs/EPIC3_*.md and docs/EPIC4_*.md as single-file consolidated copies of epicN chapters.

Canonical design remains under docs/epic3/ and docs/epic4/. These EPIC* files are staging
copies for external wiki upload.
"""

from __future__ import annotations

import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent

EPIC3_CHAPTERS = [
    "01-glossary.md",
    "02-tasks-and-scene.md",
    "03-teleoperation.md",
    "04-recording-lerobot.md",
    "05-training.md",
    "06-evaluation.md",
    "07-findings-troubleshooting.md",
    "08-future-work.md",
]

EPIC4_CHAPTERS = [
    "01-glossary.md",
    "02-background-and-stack.md",
    "03-vr-teleoperation.md",
    "04-vr-recording.md",
    "05-findings-troubleshooting.md",
    "06-future-work.md",
]

# chapter filename (with or without .md) -> consolidated page + optional default anchor
EPIC3_FILES = {f: "EPIC3_SIMULATION_TRAINING_PIPELINE.md" for f in EPIC3_CHAPTERS}
EPIC3_FILES.update({f.removesuffix(".md"): "EPIC3_SIMULATION_TRAINING_PIPELINE.md" for f in EPIC3_CHAPTERS})

EPIC4_FILES = {f: "EPIC4_VR_INTEGRATION.md" for f in EPIC4_CHAPTERS}
EPIC4_FILES.update({f.removesuffix(".md"): "EPIC4_VR_INTEGRATION.md" for f in EPIC4_CHAPTERS})

# Known chapter H1 titles -> slug used as section anchor after demotion to ##
# GitHub-style: lowercase, spaces to -, strip most punctuation
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CONTINUE_RE = re.compile(
    r"\n## Continue reading\n.*?(?=\n## |\n# |\Z)",
    re.DOTALL | re.IGNORECASE,
)


def github_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s\-]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def demote_headings(body: str) -> str:
    def repl(m: re.Match[str]) -> str:
        hashes, title = m.group(1), m.group(2)
        if len(hashes) >= 6:
            return m.group(0)
        return f"{'#' * (len(hashes) + 1)} {title}"

    return HEADING_RE.sub(repl, body)


def strip_continue_reading(body: str) -> str:
    return CONTINUE_RE.sub("\n", body)


def rewrite_link(url: str, epic: str) -> str:
    """Rewrite markdown link targets for a consolidated page living in docs/."""
    if url.startswith(("http://", "https://", "mailto:")):
        return url

    # Images / assets
    if url.startswith("../assets/"):
        return url.replace("../assets/", "assets/", 1)
    if url.startswith("assets/"):
        return url

    # Code / scripts from docs/epicN/
    if url.startswith("../../"):
        return "../" + url[len("../../") :]

    # Split path#frag
    path, frag = (url.split("#", 1) + [""])[:2]
    frag_suffix = f"#{frag}" if frag else ""

    # Already in-page
    if path == "" or path == "#":
        return f"#{frag}" if frag else url

    base = Path(path).name
    stem = Path(path).stem
    same = EPIC3_FILES if epic == "epic3" else EPIC4_FILES

    # Cross-epic before same-basename (01-glossary.md exists in both folders).
    # Folder READMEs stay as repo paths (not inlined into EPIC*).
    norm = path.replace("\\", "/")
    if "/epic4/" in f"/{norm}" or norm.startswith("epic4/"):
        if base.lower() == "readme.md" or stem.lower() == "readme":
            return "epic4/README.md" + frag_suffix
        if frag:
            return f"EPIC4_VR_INTEGRATION.md#{frag}"
        return f"EPIC4_VR_INTEGRATION.md#{stem}"

    if "/epic3/" in f"/{norm}" or norm.startswith("epic3/"):
        if base.lower() == "readme.md" or stem.lower() == "readme":
            return "epic3/README.md" + frag_suffix
        if frag:
            return f"EPIC3_SIMULATION_TRAINING_PIPELINE.md#{frag}"
        return f"EPIC3_SIMULATION_TRAINING_PIPELINE.md#{stem}"

    # Same-folder chapter links: 02-tasks-and-scene.md#foo (no ../epicN/)
    if base in same or stem in same or path in same:
        if frag:
            return f"#{frag}"
        return f"#{stem}"

    # Same-folder README.md → epicN/README.md on consolidated page
    if base.lower() == "readme.md" or stem.lower() == "readme":
        return f"{epic}/README.md" + frag_suffix

    # Parent docs links
    if path.startswith("../"):
        rest = path[3:]
        return rest + frag_suffix

    if path.startswith(("setup/", "IL_", "ACT_", "EPIC", "README")):
        return path + frag_suffix

    if path in ("EPIC3_SIMULATION_TRAINING_PIPELINE.md", "EPIC4_VR_INTEGRATION.md"):
        return path + frag_suffix

    return url
def collect_title_slugs(docs: Path, epic: str, chapters: list[str]) -> dict[str, str]:
    """Map chapter stem / filename -> github slug of H1 title."""
    mapping: dict[str, str] = {}
    folder = docs / epic
    for name in chapters:
        text = (folder / name).read_text(encoding="utf-8")
        first = text.split("\n", 1)[0]
        title = first[2:].strip() if first.startswith("# ") else name
        stem = Path(name).stem
        slug = github_slug(title)
        mapping[stem] = slug
        mapping[name] = slug
    return mapping


def map_stem_frag(frag: str, slugs: dict[str, str]) -> str:
    """If frag is a chapter stem/filename, replace with that chapter's H1 slug."""
    return slugs.get(frag, frag)


def process_chapter(
    text: str,
    epic: str,
    slug3: dict[str, str],
    slug4: dict[str, str],
) -> tuple[str, str]:
    """Return (h1_title, processed_body without H1)."""
    text = text.replace("\r\n", "\n")
    lines = text.split("\n")
    if not lines or not lines[0].startswith("# "):
        raise ValueError("Chapter must start with H1")
    title = lines[0][2:].strip()
    body = "\n".join(lines[1:]).lstrip("\n")
    body = strip_continue_reading(body)
    body = demote_headings(body)
    local_slugs = slug3 if epic == "epic3" else slug4

    def link_sub(m: re.Match[str]) -> str:
        label, url = m.group(1), m.group(2)
        new_url = rewrite_link(url, epic)
        if new_url.startswith("#"):
            new_url = f"#{map_stem_frag(new_url[1:], local_slugs)}"
        elif "#" in new_url:
            page, frag = new_url.split("#", 1)
            if page.startswith("EPIC3"):
                frag = map_stem_frag(frag, slug3)
            elif page.startswith("EPIC4"):
                frag = map_stem_frag(frag, slug4)
            new_url = f"{page}#{frag}"
        return f"[{label}]({new_url})"

    body = LINK_RE.sub(link_sub, body)
    section = f"## {title}\n\n{body.strip()}\n"
    return title, section


def assemble_epic(
    epic: str,
    chapters: list[str],
    out_name: str,
    page_title: str,
    goal: str,
    extra_banner_lines: list[str],
) -> None:
    folder = DOCS / epic
    slug3 = collect_title_slugs(DOCS, "epic3", EPIC3_CHAPTERS)
    slug4 = collect_title_slugs(DOCS, "epic4", EPIC4_CHAPTERS)

    sections: list[tuple[str, str]] = []
    for name in chapters:
        raw = (folder / name).read_text(encoding="utf-8")
        title, section = process_chapter(raw, epic, slug3, slug4)
        sections.append((title, section))

    toc_lines = [f"- [{t}](#{github_slug(t)})" for t, _ in sections]

    banner_bits = [
        f"> **In-repo docs:** The same design content is maintained as separate chapters under",
        f"> `docs/{epic}/` in the project repository (plus the [IL Workflow Runbook](IL_WORKFLOW_RUNBOOK.md)",
        f"> and [setup](setup/README.md) for how-to). This page is a single consolidated copy of that",
        f"> design material for convenience; for chapter-by-chapter browsing on GitHub, use `docs/{epic}/`.",
        f">",
        f"> Repo-relative links (runbook, setup, `scripts/`, `source/`) target the GitHub tree and may not",
        f"> resolve inside an external wiki.",
    ]
    for line in extra_banner_lines:
        banner_bits.append(f"> {line}")

    parts = [
        f"# {page_title}",
        "",
        "\n".join(banner_bits),
        "",
        "## Goal",
        "",
        goal,
        "",
        "## Table of contents",
        "",
        *toc_lines,
        "",
    ]
    for _, section in sections:
        parts.append(section)
        parts.append("")

    out_path = DOCS / out_name
    out_path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


def main() -> None:
    assemble_epic(
        epic="epic3",
        chapters=EPIC3_CHAPTERS,
        out_name="EPIC3_SIMULATION_TRAINING_PIPELINE.md",
        page_title="Epic 3 — Simulation Training Pipeline",
        goal=(
            "Build a digital twin of the Trossen Mobile AI in Isaac Sim and an imitation-learning "
            "pipeline: record human demonstrations, train policies (ACT / Pi0), and evaluate "
            "closed-loop in simulation. Pi0 sim eval remains deferred."
        ),
        extra_banner_lines=[
            "Reporting metrics: [ACT Evaluation Report](ACT_EVAL_REPORT_100K.md).",
            "Related design (VR): [Epic 4 consolidated page](EPIC4_VR_INTEGRATION.md).",
        ],
    )
    assemble_epic(
        epic="epic4",
        chapters=EPIC4_CHAPTERS,
        out_name="EPIC4_VR_INTEGRATION.md",
        page_title="Epic 4 — VR Integration",
        goal=(
            "Connect VR headsets to Isaac Sim for in-simulation teleoperation — safe demonstration "
            "practice and synthetic data collection without physical hardware risk."
        ),
        extra_banner_lines=[
            "Related design (IL pipeline): [Epic 3 consolidated page](EPIC3_SIMULATION_TRAINING_PIPELINE.md).",
            "One-time VR host: [VR workstation setup](setup/vr-workstation.md). Every session: "
            "[runbook §1](IL_WORKFLOW_RUNBOOK.md#1-vr-session-startup-every-time).",
        ],
    )


if __name__ == "__main__":
    main()
