# Silicon Sandbox — SME MVP (LLM branch)

**Ask "how would the incumbent fight back?" before you make the move — and get three different answers depending on how smart you assume the incumbent is.**

```bash
pip3 install -r requirements.txt
export OPENAI_API_KEY='your-key-here'
python3 silicon_sandbox.py            # runs demo_scenario.json, writes a PDF report
```

---

## The idea

When a small business makes a competitive move, the thing that kills it isn't usually the move — it's the *retaliation* it didn't see coming. This tool simulates that retaliation with LLM agents standing in for the incumbents.

The core mechanic is **representational complexity**: the same incumbent is instantiated at three cognitive levels, and each one produces a *meaningfully different* prediction for the identical scenario. A price-only thinker and a multi-dimensional thinker don't just disagree on intensity — from round 2 they diverge into completely separate interaction chains.

| Complexity | What the incumbent agent reasons over |
| --- | --- |
| **LOW** | Direct price competition and immediate market share, nothing else |
| **MODERATE** | Price, market share, and customer-segment dynamics |
| **HIGH** | Price, brand, supply chain, talent, regulation, long-term trajectory |

This is grounded in the strategic-representations research program — the finding that a decision-maker's *mental model* of competition, not just the facts, drives the choice they make. Here that theory is executable.

## What you get per agent

Each agent returns a structured `Prediction`, not prose:

- **Response type** — `ignore` / `match` / `escalate` / `differentiate` / `acquire` / `legal`
- **Intensity** — 1–5 resource commitment
- **Timing** — immediate / short-term / long-term
- **Reasoning** — 2–3 sentences of justification

A PDF report lands in `~/Downloads` at the end of every run.

## Multi-round mode

```bash
python3 silicon_sandbox.py --rounds 3
```

- **Round 1** — each complexity level predicts the incumbent's first response.
- **Round 2+** — an SME agent reads each incumbent's last move and decides its own next step (continue / adjust / escalate / retreat / pivot). Because the SME reacts differently to each incumbent, **every complexity level becomes its own independent trajectory** with full history carried forward. That's the payoff: you see how the incumbent's cognitive configuration shapes the *entire* competitive arc, not just one turn.

## Scenarios

The run reads `demo_scenario.json`. Swap in your own industry, incumbents, and move:

```jsonc
{
  "environment": {
    "industry": "...",
    "market_structure": "...",
    "incumbents": [{ "name": "...", "description": "..." }]
  },
  "sme_move": { "description": "...", "move_type": "..." }
}
```

The bundled demo pits **Framework** (modular, repairable laptops) against **Apple** — a real, concrete asymmetry that makes the complexity levels diverge in an intuitive way.

## Model selection

```bash
python3 silicon_sandbox.py --model gpt-5-mini   # default: gpt-5
```

## Requirements

- Python 3.9+
- OpenAI API key with access to your chosen model

## How this fits the larger project

This is the **LLM branch** of Silicon Sandbox — my venture exploring whether agent-based simulation can let SMEs war-game competitive moves the way only large firms can today.

- **This repo** — interpretable competitor reasoning via LLM agents with explicit cognitive profiles.
- **[JEPA-Silicon-Sandbox](https://github.com/krishnakem/JEPA-Silicon-Sandbox)** — the same thesis pursued with a learned latent world model instead of an LLM.

Running both branches side by side is deliberate: it's how I'm testing which representation better predicts real incumbent retaliation, work I'm continuing as a research assistant at the Ross School of Business.

## License

See repository. Research/portfolio use.
