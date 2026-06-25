#!/usr/bin/env python3
"""
Generate inner speech VTT streams for personas, seeded from a source VTT transcript.

Usage:
    python inner_speech.py transcript.vtt personas.yaml --output-dir ./output
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import yaml


# ---------------------------------------------------------------------------
# VTT parsing
# ---------------------------------------------------------------------------

@dataclass
class Cue:
    index: int
    start: float   # seconds
    end: float     # seconds
    text: str


def parse_timestamp(ts: str) -> float:
    """Convert HH:MM:SS.mmm or MM:SS.mmm to seconds."""
    ts = ts.strip()
    parts = ts.replace(",", ".").split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    raise ValueError(f"Unrecognised timestamp: {ts!r}")


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for VTT output."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def parse_vtt(path: Path) -> list[Cue]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    cues: list[Cue] = []
    i = 0
    # skip WEBVTT header
    while i < len(lines) and not lines[i].startswith("WEBVTT"):
        i += 1
    i += 1

    cue_index = 0
    while i < len(lines):
        line = lines[i].strip()

        # skip blank lines and NOTE blocks
        if not line:
            i += 1
            continue
        if line.startswith("NOTE"):
            while i < len(lines) and lines[i].strip():
                i += 1
            continue

        # optional cue identifier (a line that doesn't contain "-->")
        if "-->" not in line:
            i += 1
            if i >= len(lines):
                break
            line = lines[i].strip()

        # timing line
        if "-->" in line:
            m = re.match(r"([\d:.,]+)\s+-->\s+([\d:.,]+)", line)
            if not m:
                i += 1
                continue
            start = parse_timestamp(m.group(1))
            end = parse_timestamp(m.group(2))
            i += 1

            # collect payload lines
            payload_lines = []
            while i < len(lines) and lines[i].strip():
                payload_lines.append(lines[i].strip())
                i += 1

            text = " ".join(payload_lines)
            # strip VTT tags like <c>, <b>, speaker labels, etc.
            text = re.sub(r"<[^>]+>", "", text).strip()
            if text:
                cues.append(Cue(index=cue_index, start=start, end=end, text=text))
                cue_index += 1
        else:
            i += 1

    return cues


# ---------------------------------------------------------------------------
# Persona config
# ---------------------------------------------------------------------------

@dataclass
class PersonaConfig:
    name: str
    prompt: str


@dataclass
class Config:
    tokens_per_second: float
    min_tokens: int
    max_tokens: int
    personas: list[PersonaConfig]


def load_config(path: Path) -> Config:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    personas = [
        PersonaConfig(name=name, prompt=data["prompt"])
        for name, data in raw["personas"].items()
    ]
    return Config(
        tokens_per_second=float(raw.get("tokens_per_second", 3)),
        min_tokens=int(raw.get("min_tokens", 8)),
        max_tokens=int(raw.get("max_tokens", 80)),
        personas=personas,
    )


# ---------------------------------------------------------------------------
# Inner speech generation
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are simulating the unfiltered inner speech of a listener as they hear a spoken transcript in real time.

Persona: {persona_prompt}

Rules:
- Think in first person, present tense — this is a live thought stream, not a summary.
- React to the *current* words while drawing on everything heard so far.
- You have a strict token budget. Be concise; cut off naturally if you run out of space.
- Do NOT narrate that you are thinking. Just think.
- Do NOT repeat or paraphrase the transcript directly — respond to it.
- Output only the inner speech. No labels, no quotes, no explanations.
"""

USER_TEMPLATE = """\
Transcript so far (everything heard up to this moment):
---
{history}
---

New words just spoken:
"{current}"

Think your inner response in {budget} tokens or fewer.
"""


def generate_inner_speech(
    client: anthropic.Anthropic,
    persona: PersonaConfig,
    history: str,
    current: str,
    budget: int,
) -> str:
    system = SYSTEM_TEMPLATE.format(persona_prompt=persona.prompt)
    user = USER_TEMPLATE.format(
        history=history if history else "(nothing yet)",
        current=current,
        budget=budget,
    )
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=budget,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text.strip()


# ---------------------------------------------------------------------------
# VTT output
# ---------------------------------------------------------------------------

def write_vtt(cues: list[tuple[float, float, str]], path: Path) -> None:
    lines = ["WEBVTT", ""]
    for i, (start, end, text) in enumerate(cues, 1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_persona(
    persona: PersonaConfig,
    cues: list[Cue],
    config: Config,
    client: anthropic.Anthropic,
    output_dir: Path,
    verbose: bool,
) -> None:
    output_cues: list[tuple[float, float, str]] = []
    history_parts: list[str] = []

    print(f"\n[{persona.name}] generating inner speech for {len(cues)} cues...")

    for cue in cues:
        duration = cue.end - cue.start
        budget = int(duration * config.tokens_per_second)
        budget = max(config.min_tokens, min(config.max_tokens, budget))

        history = " ".join(history_parts)
        thought = generate_inner_speech(
            client=client,
            persona=persona,
            history=history,
            current=cue.text,
            budget=budget,
        )

        output_cues.append((cue.start, cue.end, thought))
        history_parts.append(cue.text)

        if verbose:
            print(f"  [{format_timestamp(cue.start)}] budget={budget}t | {thought[:80]}{'...' if len(thought) > 80 else ''}")

    out_path = output_dir / f"inner_{persona.name}.vtt"
    write_vtt(output_cues, out_path)
    print(f"[{persona.name}] written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inner speech VTTs from a transcript.")
    parser.add_argument("vtt", type=Path, help="Source VTT transcript file")
    parser.add_argument("personas", type=Path, help="Personas YAML config file")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each thought as it's generated")
    args = parser.parse_args()

    if not args.vtt.exists():
        print(f"Error: VTT file not found: {args.vtt}", file=sys.stderr)
        sys.exit(1)
    if not args.personas.exists():
        print(f"Error: Personas file not found: {args.personas}", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cues = parse_vtt(args.vtt)
    print(f"Parsed {len(cues)} cues from {args.vtt.name}")

    config = load_config(args.personas)
    print(f"Loaded {len(config.personas)} persona(s): {[p.name for p in config.personas]}")
    print(f"Budget: {config.tokens_per_second} tok/sec, min={config.min_tokens}, max={config.max_tokens}")

    client = anthropic.Anthropic()

    for persona in config.personas:
        process_persona(
            persona=persona,
            cues=cues,
            config=config,
            client=client,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
