# youtube inner-speech simulator

Generate persona-driven inner speech streams from video transcripts.

Given a VTT transcript and a set of personas, this tool simulates what different listeners might be thinking in real time as they hear a talk — producing a new VTT file per persona with timestamped inner monologue synced to the original speech.

The token budget for each thought is proportional to the duration of the transcript cue, so faster speech produces shorter thoughts and slower speech gives more room to reflect.

## Usage

```bash
pip install anthropic pyyaml
export ANTHROPIC_API_KEY=sk-...

python inner_speech.py transcript.vtt personas.yaml --output-dir ./output --verbose
```

This produces one file per persona in the output directory, e.g. `inner_no_apologies_right.vtt`, with the same timestamps as the source transcript.

## Files

- `inner_speech.py` — main script
- `personas.yaml` — persona configs and token budget settings

## Persona config

```yaml
tokens_per_second: 3   # thinking budget per second of speech
min_tokens: 8          # floor (prevents zero-budget calls on very short cues)
max_tokens: 60         # ceiling (prevents runaway output on slow speech)

personas:
  my_persona:
    prompt: >
      Describe the listener in concrete biographical terms...
```

The included `personas.yaml` contains all nine groups from the [Pew Research 2026 Political Typology](https://www.pewresearch.org/politics/2026/06/10/beyond-red-vs-blue-the-political-typology/), grounded in biographical personas rather than abstract trait lists.

## How it works

For each cue in the transcript, the script:

1. Calculates a token budget from the cue duration
2. Sends the persona, the full transcript heard so far, and the new cue text to Claude
3. Constrains the response to the token budget via `max_tokens`
4. Writes the resulting thought as a cue in the output VTT at the same timestamp

The accumulated transcript history grows with each cue, so the persona is always reacting to new words while carrying everything heard before. For very long transcripts, consider adding a rolling history window to avoid hitting context limits.

## Next steps

Planned: embed sliding windows of both transcript and inner speech and visualize divergence across personas in 3D semantic space.
