# Silicon Sandbox MVP

**Proof-of-concept CLI for simulating incumbent retaliation to SME competitive moves.**

LLM-based agents, each configured with a different level of *representational complexity*, produce meaningfully different retaliation predictions when presented with the same competitive scenario. The key variable is the agent's cognitive profile — how broadly or narrowly it represents the competitive landscape.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY='your-key-here'

# 3. Run the simulation
python silicon_sandbox.py
```

## Usage

The simulation loads its scenario from `demo_scenario.json` in the same directory. Edit that file to change the industry, incumbents, or SME move.

### Model Selection

Switch the underlying LLM (default: `gpt-5`):

```bash
python silicon_sandbox.py --model gpt-5-mini
```

### Multi-Round Simulation

Run multiple rounds of competitive interaction where the SME and incumbents react to each other:

```bash
python silicon_sandbox.py --rounds 3
```

In multi-round mode:
- **Round 1** runs the same as the default single-round simulation.
- **Round 2+** introduces an SME agent that evaluates each incumbent's previous response and decides its next move (continue, adjust, escalate, retreat, or pivot). Each incumbent then responds to the SME's adjustment with full conversation history carried forward.
- Since the SME reacts differently to each incumbent's response, each complexity level becomes its own independent interaction chain from Round 2 onward. This shows how the incumbent's cognitive configuration shapes the entire trajectory of the competitive interaction.

### Scenario Format

See `demo_scenario.json` for the expected structure:

- `environment.industry` — description of the industry
- `environment.market_structure` — how the market is organized
- `environment.incumbents[]` — list of incumbents, each with `name` and `description`
- `sme_move.description` — what the SME is doing
- `sme_move.move_type` — short label for the move

## Understanding the Output

The system creates **three versions** of each incumbent, each with a different cognitive profile:

| Complexity Level | What the Agent Considers |
|---|---|
| **LOW** (Simple) | Direct price competition and immediate market share only |
| **MODERATE** | Price, market share, and customer segment dynamics |
| **HIGH** (Complex) | Price, brand, supply chain, talent, regulatory, long-term trajectory |

For each agent, the output shows:
- **Response type**: ignore, match, escalate, differentiate, acquire, or legal
- **Intensity**: 1–5 scale of resource commitment
- **Timing**: immediate, short-term, or long-term
- **Reasoning**: 2–3 sentences explaining the agent's assessment

## Requirements

- Python 3.9+
- OpenAI API key with access to `gpt-5` (or chosen model)
