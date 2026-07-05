"""Append a dated entry to RESEARCH_LOG.md.

Usage from any script:
    from log_entry import log
    log("Generated X clips", "- clip_1.mov: academy_3v1, 20s, ball visible")
"""

from datetime import date
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent.parent / "RESEARCH_LOG.md"
_ANCHOR = "<!-- AUTO-UPDATED — do not edit below this line manually -->"


def log(heading: str, *bullet_lines: str):
    """Append a dated section to the Activity Log.

    Args:
        heading:      Short title for the entry (e.g. "Generated 4 clips").
        *bullet_lines: Each becomes a '- ...' bullet. Pass pre-formatted
                       lines (with or without leading '- ').
    """
    if not LOG.exists():
        print(f"[log_entry] WARNING: log file not found at {LOG}")
        return

    today = date.today().isoformat()
    bullets = "\n".join(
        f"- {ln.lstrip('- ')}" for ln in bullet_lines
    ) if bullet_lines else ""
    block = f"\n### {today} — {heading}\n\n{bullets}\n" if bullets else f"\n### {today} — {heading}\n"

    text = LOG.read_text()
    if _ANCHOR not in text:
        # Fallback: just append at end
        LOG.write_text(text.rstrip() + "\n" + block)
    else:
        LOG.write_text(text.rstrip() + "\n" + block)

    print(f"[log_entry] appended '{heading}' to {LOG.name}")
