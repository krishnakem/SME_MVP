# Silicon Sandbox MVP

**Proof-of-concept CLI for simulating incumbent retaliation to SME competitive moves.**

This prototype demonstrates the core mechanism of the Silicon Sandbox framework (Kemisetti, 2026): LLM-based agents, each configured with a different level of *representational complexity* (Csaszar, 2018; Csaszar & Ostler, 2020), produce meaningfully different retaliation predictions when presented with the same competitive scenario. The key variable being manipulated is the agent's cognitive profile — how broadly or narrowly it represents the competitive landscape — grounded in the strategic representations research program, which shows that a decision-maker's mental model determines the strategic choices they make.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# 3. Run the demo scenario
python silicon_sandbox.py
```

## Usage

### Default Demo

Running without arguments uses a built-in scenario: a local restaurant undercuts a national fast-casual chain by 30% on lunch combos, targeting college students.

```bash
python silicon_sandbox.py
```

### Custom Scenario

Provide your own scenario via a JSON file:

```bash
python silicon_sandbox.py --scenario demo_scenario.json
```

See `demo_scenario.json` for the expected format. The JSON structure requires:
- `environment.industry` — description of the industry
- `environment.market_structure` — how the market is organized
- `environment.incumbents[]` — list of incumbents, each with `name` and `description`
- `sme_move.description` — what the SME is doing
- `sme_move.move_type` — short label for the move

### Model Selection

Switch the underlying LLM (default: `gpt-4o`):

```bash
python silicon_sandbox.py --model gpt-4o-mini
```

## Understanding the Output

The system creates **three versions** of each incumbent, each with a different cognitive profile:

| Complexity Level | What the Agent Considers | Based On |
|---|---|---|
| **LOW** (Simple) | Direct price competition and immediate market share only | Narrow representational frame |
| **MODERATE** | Price, market share, and customer segment dynamics | Multi-dimensional but bounded |
| **HIGH** (Complex) | Price, brand, supply chain, talent, regulatory, long-term trajectory | Rich representational frame |

For each agent, the output shows:
- **Response type**: ignore, match, escalate, differentiate, acquire, or legal
- **Intensity**: 1–5 scale of resource commitment
- **Timing**: immediate, short-term, or long-term
- **Reasoning**: 2–3 sentences explaining the agent's assessment

The comparison summary highlights whether different cognitive configurations produced different strategic responses — the central question this framework investigates.

## Theoretical Grounding

The representational complexity variable is drawn from:

- **Csaszar, F. A. (2018).** "What Makes a Decision Strategic? Strategic Representations." — Argues that strategic decisions are defined by how the decision-maker mentally frames the problem.
- **Csaszar, F. A. & Ostler, J. (2020).** "A Contingency Theory of Representational Complexity in Organizations." *Organization Science.* — Demonstrates that the optimal level of representational complexity depends on both the environment and the firm's experience.
- **Csaszar, F. A. & Levinthal, D. A. (2016).** "Mental Representation and the Discovery of New Strategies." *Strategic Management Journal.* — Shows that decision-makers with different mental models make systematically different strategic choices.
- **Csaszar, F. A., Ketkar, H., & Kim, H. (2024).** "Artificial Intelligence and Strategic Decision-Making: Evidence from Entrepreneurs and Investors." *Strategy Science.* — Establishes that LLMs can generate and evaluate strategies at a level comparable to human entrepreneurs and investors.

## Requirements

- Python 3.10+
- OpenAI API key with access to `gpt-4o` (or chosen model)
