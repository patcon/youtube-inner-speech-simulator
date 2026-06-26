#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "anthropic",
#   "pyyaml",
#   "youtube-transcript-api",
# ]
# ///
"""
Generate inner speech VTT streams for personas, seeded from a source VTT transcript.

Usage:
    uv run create_inner_speech.py https://youtube.com/watch?v=ID personas.yaml
    uv run create_inner_speech.py transcript.vtt personas.yaml --output-dir ./output
"""

import argparse
import datetime
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
import yaml
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import WebVTTFormatter


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


def parse_youtube_timecode(tc: str) -> float:
    """Convert YouTube-style HH:MM:SS or MM:SS (whole seconds) to seconds."""
    tc = tc.strip()
    parts = tc.split(":")
    try:
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + int(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + int(s)
    except ValueError:
        pass
    raise ValueError(f"Unrecognised timecode: {tc!r} (expected MM:SS or HH:MM:SS)")


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for VTT output."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def parse_vtt(path: Path) -> list[Cue]:
    return parse_vtt_text(path.read_text(encoding="utf-8"))


def parse_vtt_text(text: str) -> list[Cue]:
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
# YouTube transcript fetching
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> str | None:
    m = re.search(r'(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else None


def fetch_youtube_vtt(video_id: str) -> str:
    transcript = YouTubeTranscriptApi().fetch(video_id)
    return WebVTTFormatter().format_transcript(transcript)


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
        model="claude-sonnet-4-6",
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
    run_ts: str,
    video_id: str | None = None,
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

    vid_part = f"_{video_id}" if video_id else ""
    out_path = output_dir / f"inner_{persona.name}{vid_part}_{run_ts}.vtt"
    write_vtt(output_cues, out_path)
    print(f"[{persona.name}] written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate inner speech VTTs from a transcript.")
    parser.add_argument("vtt", help="Source VTT file path or YouTube URL")
    parser.add_argument("personas", type=Path, help="Personas YAML config file")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each thought as it's generated")
    parser.add_argument("--only", metavar="PERSONA", nargs="+", help="Run only these persona name(s)")
    parser.add_argument("--max-time", type=float, metavar="SECONDS", help="Stop this many seconds after --start-timecode (or from the beginning if not set)")
    parser.add_argument("--start-timecode", metavar="TIMECODE", help="Start from this timecode (MM:SS or HH:MM:SS); continues to end unless --end-timecode or --max-time set")
    parser.add_argument("--end-timecode", metavar="TIMECODE", help="Stop at this absolute timecode (MM:SS or HH:MM:SS)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    parser.add_argument("--dry-run", "-n", action="store_true", help="Estimate tokens and cost without generating any output")
    args = parser.parse_args()

    if not args.personas.exists():
        print(f"Error: Personas file not found: {args.personas}", file=sys.stderr)
        sys.exit(1)

    # Parse YouTube-style timecodes into seconds
    start_seconds: float | None = None
    end_seconds: float | None = None
    try:
        if args.start_timecode:
            start_seconds = parse_youtube_timecode(args.start_timecode)
        if args.end_timecode:
            end_seconds = parse_youtube_timecode(args.end_timecode)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.max_time is not None:
        effective_start = start_seconds if start_seconds is not None else 0.0
        max_time_end = effective_start + args.max_time
        if end_seconds is None or max_time_end < end_seconds:
            end_seconds = max_time_end

    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_id: str | None = None
    if args.vtt.startswith("http://") or args.vtt.startswith("https://"):
        video_id = extract_video_id(args.vtt)
        if not video_id:
            print(f"Error: could not extract a YouTube video ID from: {args.vtt}", file=sys.stderr)
            sys.exit(1)
        print(f"Fetching YouTube transcript for video ID: {video_id}")
        vtt_text = fetch_youtube_vtt(video_id)
        cues = parse_vtt_text(vtt_text)
        source_name = video_id
    else:
        vtt_path = Path(args.vtt)
        if not vtt_path.exists():
            print(f"Error: VTT file not found: {vtt_path}", file=sys.stderr)
            sys.exit(1)
        cues = parse_vtt(vtt_path)
        source_name = vtt_path.name

    if start_seconds is not None:
        cues = [c for c in cues if c.start >= start_seconds]
    if end_seconds is not None:
        cues = [c for c in cues if c.start < end_seconds]

    range_desc = ""
    if start_seconds is not None or end_seconds is not None:
        lo = format_timestamp(start_seconds) if start_seconds is not None else "start"
        hi = format_timestamp(end_seconds) if end_seconds is not None else "end"
        range_desc = f" [{lo} → {hi}]"
    print(f"Parsed {len(cues)} cues from {source_name}{range_desc}")

    config = load_config(args.personas)
    if args.only:
        unknown = set(args.only) - {p.name for p in config.personas}
        if unknown:
            print(f"Error: unknown persona(s): {sorted(unknown)}", file=sys.stderr)
            sys.exit(1)
        config.personas = [p for p in config.personas if p.name in args.only]
    print(f"Loaded {len(config.personas)} persona(s): {[p.name for p in config.personas]}")
    print(f"Budget: {config.tokens_per_second} tok/sec, min={config.min_tokens}, max={config.max_tokens}")

    total_calls = len(cues) * len(config.personas)
    print(f"\nThis will make {total_calls} API call(s) ({len(cues)} cues × {len(config.personas)} persona(s)).")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set. Get a key at https://console.anthropic.com/ and run:\n  export ANTHROPIC_API_KEY=sk-...", file=sys.stderr)
        sys.exit(1)

    print("Estimating token usage...", end=" ", flush=True)
    client = anthropic.Anthropic()
    total_input_tokens = 0
    total_output_tokens = 0
    for persona in config.personas:
        history_parts: list[str] = []
        for cue in cues:
            duration = cue.end - cue.start
            budget = int(duration * config.tokens_per_second)
            budget = max(config.min_tokens, min(config.max_tokens, budget))
            system = SYSTEM_TEMPLATE.format(persona_prompt=persona.prompt)
            user = USER_TEMPLATE.format(
                history=" ".join(history_parts) if history_parts else "(nothing yet)",
                current=cue.text,
                budget=budget,
            )
            result = client.messages.count_tokens(
                model="claude-sonnet-4-6",
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            total_input_tokens += result.input_tokens
            total_output_tokens += budget
            history_parts.append(cue.text)
    # Approximate cost: Sonnet input $3/MTok, output $15/MTok
    est_cost = (total_input_tokens * 3 + total_output_tokens * 15) / 1_000_000
    print(f"done.\n  ~{total_input_tokens:,} input tokens, ~{total_output_tokens:,} max output tokens, ~${est_cost:.2f} estimated cost")

    if args.dry_run:
        print("Dry run — exiting without generating output.")
        sys.exit(0)

    if not args.yes:
        answer = input("Proceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            sys.exit(0)

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if video_id:
        transcript_path = args.output_dir / f"transcript_{video_id}_{run_ts}.vtt"
        transcript_path.write_text(vtt_text, encoding="utf-8")
        print(f"Saved original transcript to {transcript_path}")

    for persona in config.personas:
        process_persona(
            persona=persona,
            cues=cues,
            config=config,
            client=client,
            output_dir=args.output_dir,
            verbose=args.verbose,
            run_ts=run_ts,
            video_id=video_id,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
