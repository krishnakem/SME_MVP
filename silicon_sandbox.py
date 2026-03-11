"""
Silicon Sandbox MVP — Proof-of-Concept CLI

Simulates incumbent retaliation to SME competitive moves using LLM-based agents
configured with varying levels of representational complexity.

Each agent embodies a different cognitive profile — simple, moderate, or complex —
that shapes how it perceives competitive threats and formulates responses. This is
grounded in the strategic representations research program, 
which demonstrates that a decision-maker's mental model of competition
determines the strategic choices they make.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from dataclasses import dataclass

from openai import OpenAI


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration for a single incumbent agent."""
    name: str
    complexity: str          # "simple", "moderate", or "complex"
    firm_description: str    # Industry position, size, history
    cognitive_profile: str   # Explicit reasoning constraints


@dataclass
class Prediction:
    """Structured prediction from an incumbent agent."""
    agent_name: str
    complexity: str
    response_type: str       # ignore / match / escalate / differentiate / acquire / legal
    intensity: int           # 1–5
    timing: str              # immediate / short-term / long-term
    reasoning: str


@dataclass
class SMEAdjustment:
    """Structured adjustment from the SME agent in multi-round simulation."""
    adjusted_move: str
    move_type: str
    reasoning: str


# ---------------------------------------------------------------------------
# Cognitive profiles — grounded in representational complexity framework
#
# The key insight is that strategic decisions are shaped by how the
# decision-maker mentally represents the competitive landscape. Agents
# with simpler representations consider fewer dimensions and update slowly;
# agents with complex representations consider many interacting factors but
# may overcomplicate simple situations.
# ---------------------------------------------------------------------------

COGNITIVE_PROFILES = {
    "simple": textwrap.dedent("""\
        YOUR COGNITIVE PROFILE — LOW REPRESENTATIONAL COMPLEXITY

        You evaluate competitive threats through ONE primary lens: direct price
        competition and immediate market share impact.

        Constraints on your reasoning:
        - You focus exclusively on price and volume. If a competitor's move does
          not directly undercut your price or steal customers you can name, you
          consider it a minor nuisance.
        - You do NOT consider long-term strategic positioning, brand equity effects,
          second-order market dynamics, or indirect competitive spillovers.
        - You do NOT model the competitor's likely next moves or broader strategy.
        - You update your mental model of the competitive landscape slowly — only
          clear, sustained market share loss would change your assessment.
        - You have blind spots: you may dismiss threats that operate through
          channels other than direct price competition.

        When reasoning, stay within these constraints. Do not spontaneously
        introduce strategic dimensions beyond price and immediate volume impact."""),

    "moderate": textwrap.dedent("""\
        YOUR COGNITIVE PROFILE — MODERATE REPRESENTATIONAL COMPLEXITY

        You evaluate competitive threats across 2–3 strategic dimensions: price,
        market share, and customer segment dynamics.

        Constraints on your reasoning:
        - You consider direct price effects AND shifts in customer segments. You
          recognize that losing a demographic (e.g., college students) could have
          downstream effects on revenue mix.
        - You can identify SOME indirect competitive effects — for instance, that a
          competitor gaining a loyal following in one segment may expand later — but
          you do not model complex multi-step strategic sequences.
        - You do NOT fully account for brand positioning, supply-chain effects,
          regulatory dynamics, or talent-market implications.
        - You update your mental model at a moderate pace — you respond to clear
          trends within a quarter, but you do not react to every signal.
        - You occasionally miss threats that operate through dimensions you are not
          tracking (e.g., a competitor building operational capabilities).

        When reasoning, consider price, share, and customer segments. Acknowledge
        but do not deeply analyze dimensions beyond these three."""),

    "complex": textwrap.dedent("""\
        YOUR COGNITIVE PROFILE — HIGH REPRESENTATIONAL COMPLEXITY

        You evaluate competitive threats across MANY interacting dimensions: price,
        market share, customer segments, brand positioning, supply chain, talent
        acquisition, regulatory environment, and long-term strategic trajectory.

        Constraints on your reasoning:
        - You model the competitor's likely next 2–3 moves and consider what their
          current action signals about their broader strategy.
        - You consider second-order effects: how this move changes the competitive
          landscape for other players, how it affects your supplier or talent
          relationships, and how it shifts customer expectations.
        - You track feedback loops — e.g., if you respond aggressively, how might
          that escalate or reshape the market structure?
        - You update your mental model quickly and continuously in response to new
          competitive signals.
        - Your potential blind spot: you may OVERCOMPLICATE simple situations,
          seeing strategic depth where a straightforward competitive move exists.
          You may over-invest analytical resources and delay action.

        When reasoning, consider the full range of strategic dimensions and their
        interactions. Model the competitor's trajectory, not just their current move."""),
}


# ---------------------------------------------------------------------------
# SME agent system prompt — used in multi-round simulations
# ---------------------------------------------------------------------------

SME_AGENT_PROMPT = textwrap.dedent("""\
    You are a small business owner / entrepreneur who has just entered a
    competitive market. You made an initial competitive move and the incumbent
    has responded. Your job is to evaluate the incumbent's response and decide
    your next move.

    Consider these options:
    - CONTINUE: stay the course with your current strategy
    - ADJUST: modify your approach based on the incumbent's response
    - ESCALATE: intensify your competitive pressure
    - RETREAT: pull back from direct competition
    - PIVOT: shift to a completely different competitive approach

    You are resourceful but resource-constrained. You cannot match the
    incumbent dollar-for-dollar, but you can be creative, fast, and targeted.

    RESPONSE FORMAT:
    You must respond with valid JSON and nothing else. Use this exact structure:
    {
        "adjusted_move": "<1-2 sentences describing what you do next>",
        "move_type": "<short label for your move>",
        "reasoning": "<2-3 sentences explaining why you chose this response>"
    }

    Respond ONLY with the JSON object.""")


# ---------------------------------------------------------------------------
# Default scenario file path
# ---------------------------------------------------------------------------

SCENARIO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_scenario.json")


# ---------------------------------------------------------------------------
# System prompt construction
# ---------------------------------------------------------------------------

def build_system_prompt(agent: AgentConfig) -> str:
    """
    Construct the full system prompt for an incumbent agent.

    The prompt has three layers:
      1. Role framing — the agent acts as a decision-maker at the incumbent firm.
      2. Firm persona — industry position, size, competitive history.
      3. Cognitive profile — representational complexity constraints that shape
         how the agent perceives and responds to competitive threats.
    """
    return textwrap.dedent(f"""\
        You are a senior strategic decision-maker at {agent.name}.
        Your job is to evaluate competitive threats and recommend your firm's response.

        ABOUT YOUR FIRM:
        {agent.firm_description}

        {agent.cognitive_profile}

        RESPONSE FORMAT:
        You must respond with valid JSON and nothing else. Use this exact structure:
        {{
            "response_type": "<one of: ignore, match, escalate, differentiate, acquire, legal>",
            "intensity": <integer 1 to 5>,
            "timing": "<one of: immediate, short-term, long-term>",
            "reasoning": "<2-3 sentences explaining your assessment>"
        }}

        Definitions:
        - response_type: your recommended competitive response
            - ignore: no action needed
            - match: replicate the competitor's move (e.g., match their price)
            - escalate: respond more aggressively than the competitor's move
            - differentiate: respond by emphasizing your unique strengths
            - acquire: attempt to acquire or absorb the competitor
            - legal: pursue legal or regulatory action
        - intensity: 1 = minimal effort, 5 = maximum corporate resources
        - timing: immediate = within days, short-term = within weeks/months,
          long-term = strategic shift over quarters/years
        - reasoning: explain WHY this response, given how you perceive the threat

        Respond ONLY with the JSON object.""")


def build_user_message(scenario: dict) -> str:
    """Format the competitive scenario as the user message sent to each agent."""
    env = scenario["environment"]
    move = scenario["sme_move"]

    return textwrap.dedent(f"""\
        COMPETITIVE SCENARIO

        Industry: {env['industry']}
        Market Structure: {env['market_structure']}

        SME MOVE:
        {move['description']}
        Move Type: {move['move_type']}

        Given your firm's position and how you evaluate competitive threats,
        what is your recommended response? Provide your prediction as JSON.""")


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def query_agent(client: OpenAI, model: str, agent: AgentConfig, scenario: dict) -> Prediction:
    """
    Send the scenario to a single incumbent agent and parse the response.

    Each agent receives the same scenario but interprets it through its own
    cognitive profile — this is the core mechanism the Silicon Sandbox tests.
    Different representational complexity produces different retaliation predictions
    from the same competitive stimulus.
    """
    system_prompt = build_system_prompt(agent)
    user_message = build_user_message(scenario)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=4096,
    )

    content = response.choices[0].message.content
    raw = (content or "").strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Prediction(
            agent_name=agent.name,
            complexity=agent.complexity,
            response_type="parse_error",
            intensity=0,
            timing="unknown",
            reasoning=f"Failed to parse JSON: {raw[:300]}",
        )

    return Prediction(
        agent_name=agent.name,
        complexity=agent.complexity,
        response_type=data.get("response_type", "unknown"),
        intensity=int(data.get("intensity", 0)),
        timing=data.get("timing", "unknown"),
        reasoning=data.get("reasoning", ""),
    )


def query_agent_with_history(
    client: OpenAI, model: str, agent: AgentConfig, messages: list[dict],
) -> tuple[Prediction, list[dict]]:
    """
    Query an incumbent agent using a full conversation history.

    Returns the prediction and the updated messages list (with the new
    assistant response appended) so subsequent rounds can build on it.
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=4096,
    )

    content = response.choices[0].message.content
    raw = (content or "").strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "response_type": "parse_error",
            "intensity": 0,
            "timing": "unknown",
            "reasoning": f"Failed to parse JSON: {raw[:300]}",
        }

    # Append assistant response to history for future rounds
    updated_messages = messages + [{"role": "assistant", "content": raw}]

    return Prediction(
        agent_name=agent.name,
        complexity=agent.complexity,
        response_type=data.get("response_type", "unknown"),
        intensity=int(data.get("intensity", 0)),
        timing=data.get("timing", "unknown"),
        reasoning=data.get("reasoning", ""),
    ), updated_messages


def query_sme_agent(
    client: OpenAI, model: str, scenario: dict, incumbent_prediction: Prediction,
) -> SMEAdjustment:
    """
    Query the SME agent to decide its next move based on the incumbent's response.
    """
    move = scenario["sme_move"]
    user_message = textwrap.dedent(f"""\
        YOUR ORIGINAL MOVE:
        {move['description']}
        Move Type: {move['move_type']}

        THE INCUMBENT'S RESPONSE:
        Incumbent: {incumbent_prediction.agent_name}
        Response Type: {incumbent_prediction.response_type}
        Intensity: {incumbent_prediction.intensity}/5
        Timing: {incumbent_prediction.timing}
        Reasoning: {incumbent_prediction.reasoning}

        Given this response, what is your next move? Respond as JSON.""")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "developer", "content": SME_AGENT_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_completion_tokens=4096,
    )

    content = response.choices[0].message.content
    raw = (content or "").strip()

    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return SMEAdjustment(
            adjusted_move=f"[Parse error: {raw[:200]}]",
            move_type="error",
            reasoning="Failed to parse SME agent response.",
        )

    return SMEAdjustment(
        adjusted_move=data.get("adjusted_move", ""),
        move_type=data.get("move_type", ""),
        reasoning=data.get("reasoning", ""),
    )


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def create_agents(scenario: dict) -> list[AgentConfig]:
    """
    Create one agent per incumbent × complexity level.

    For the MVP, each incumbent is simulated at all three representational
    complexity levels (simple, moderate, complex) so the user can see how
    cognitive configuration changes the predicted response — the central
    research question of the Silicon Sandbox.
    """
    agents = []
    for incumbent in scenario["environment"]["incumbents"]:
        for level, profile in COGNITIVE_PROFILES.items():
            agents.append(AgentConfig(
                name=incumbent["name"],
                complexity=level,
                firm_description=incumbent["description"],
                cognitive_profile=profile,
            ))
    return agents


# ---------------------------------------------------------------------------
# Output display
# ---------------------------------------------------------------------------

SEPARATOR = "=" * 78
ROUND_SEP = "─" * 74


def _complexity_label(complexity: str) -> str:
    return {"simple": "LOW", "moderate": "MODERATE", "complex": "HIGH"}.get(
        complexity, complexity.upper()
    )


def _display_prediction(pred: Prediction) -> None:
    """Print a single agent prediction block."""
    print(f"  Agent: {pred.agent_name}  |  Complexity: {_complexity_label(pred.complexity)}")
    print(f"  {'-' * 50}")
    print(f"  Response Type : {pred.response_type}")
    print(f"  Intensity     : {'█' * pred.intensity}{'░' * (5 - pred.intensity)} ({pred.intensity}/5)")
    print(f"  Timing        : {pred.timing}")
    print(f"  Reasoning     : {_wrap(pred.reasoning, 60)}")
    print()


def display_results(predictions: list[Prediction]) -> None:
    """Print a structured comparison of agent predictions (single-round)."""

    print(f"\n{SEPARATOR}")
    print("  SILICON SANDBOX — INCUMBENT RETALIATION PREDICTIONS")
    print(f"{SEPARATOR}\n")

    for pred in predictions:
        _display_prediction(pred)

    print(f"{SEPARATOR}\n")


def display_multiround_results(
    num_rounds: int,
    round_data: list[dict],
) -> None:
    """
    Print multi-round simulation results.

    round_data is a list of dicts, one per round. Each dict has:
      - "sme_move": str (description of SME move/adjustment for this round)
      - "predictions": list of (SMEAdjustment | None, Prediction) tuples
        In round 1 the SMEAdjustment is None.
    """
    print(f"\n{SEPARATOR}")
    print(f"  SILICON SANDBOX — MULTI-ROUND SIMULATION ({num_rounds} rounds)")
    print(f"{SEPARATOR}")

    for round_num, rd in enumerate(round_data, 1):
        print(f"\n  ── ROUND {round_num} {ROUND_SEP[len(f' ROUND {round_num} '):]}")
        print()

        if round_num == 1:
            print(f"  SME Move: {rd['sme_move']}")
            print()
            for pred in rd["predictions"]:
                _display_prediction(pred)
        else:
            for adjustment, pred in rd["predictions"]:
                label = _complexity_label(pred.complexity)
                print(f"  SME Adjustment (responding to {label} incumbent): {adjustment.adjusted_move}")
                print(f"  SME Reasoning: {_wrap(adjustment.reasoning, 60)}")
                print()
                _display_prediction(pred)

    print(f"{SEPARATOR}\n")


def _wrap(text: str, width: int) -> str:
    """Wrap text with proper indentation for display."""
    lines = textwrap.wrap(text, width=width)
    return ("\n" + " " * 20).join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def load_scenario(path: str) -> dict:
    """Load scenario from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Silicon Sandbox MVP — Simulate incumbent retaliation to SME moves "
            "using LLM agents with varying representational complexity."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="OpenAI model to use (default: gpt-5).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of simulation rounds (default: 1). Multi-round simulations "
             "add SME agent reactions and incumbent follow-up responses.",
    )
    args = parser.parse_args()

    # Validate API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Load scenario
    print(f"\nLoading scenario from: {SCENARIO_PATH}")
    scenario = load_scenario(SCENARIO_PATH)

    # Create agents and run simulation
    agents = create_agents(scenario)
    num_rounds = max(1, args.rounds)

    if num_rounds == 1:
        # --- Single-round mode (original behavior) ---
        print(f"Simulating {len(agents)} agent configuration(s) using {args.model}...\n")

        predictions = []
        for agent in agents:
            label = f"  [{agent.complexity.upper():>8}] {agent.name}"
            print(f"{label} — querying...", end="", flush=True)
            pred = query_agent(client, args.model, agent, scenario)
            print(f" → {pred.response_type} (intensity {pred.intensity}/5)")
            predictions.append(pred)

        display_results(predictions)
    else:
        # --- Multi-round mode ---
        print(
            f"Simulating {len(agents)} agent(s) × {num_rounds} rounds "
            f"using {args.model}...\n"
        )

        system_prompt = build_system_prompt(agents[0])  # rebuilt per-agent below
        user_message = build_user_message(scenario)

        # Per-agent conversation histories: agent index → messages list
        histories: dict[int, list[dict]] = {}
        round_data: list[dict] = []

        for round_num in range(1, num_rounds + 1):
            print(f"  ── Round {round_num} of {num_rounds} ──")

            if round_num == 1:
                # Round 1: same as single-round but track history
                predictions = []
                for i, agent in enumerate(agents):
                    label = f"    [{agent.complexity.upper():>8}] {agent.name}"
                    print(f"{label} — querying...", end="", flush=True)

                    messages = [
                        {"role": "developer", "content": build_system_prompt(agent)},
                        {"role": "user", "content": user_message},
                    ]
                    pred, updated = query_agent_with_history(
                        client, args.model, agent, messages
                    )
                    histories[i] = updated
                    predictions.append(pred)
                    print(f" → {pred.response_type} (intensity {pred.intensity}/5)")

                round_data.append({
                    "sme_move": scenario["sme_move"]["description"],
                    "predictions": predictions,
                })
            else:
                # Round 2+: SME adjusts per-agent, then incumbent responds
                prev_predictions = round_data[-1]["predictions"]
                # In round 1, predictions is a flat list; in round 2+ it's
                # list of (adjustment, prediction) tuples.
                if round_num == 2:
                    prev_preds = prev_predictions
                else:
                    prev_preds = [p for _, p in prev_predictions]

                round_entries = []
                for i, agent in enumerate(agents):
                    prev_pred = prev_preds[i]
                    label = f"    [{agent.complexity.upper():>8}] {agent.name}"

                    # SME agent decides next move
                    print(f"{label} — SME adjusting...", end="", flush=True)
                    adjustment = query_sme_agent(
                        client, args.model, scenario, prev_pred
                    )
                    print(f" → {adjustment.move_type}")

                    # Add SME adjustment to incumbent's conversation history
                    sme_follow_up = (
                        f"The SME has responded to your action. Here is their next move:\n\n"
                        f"SME Adjustment: {adjustment.adjusted_move}\n"
                        f"Move Type: {adjustment.move_type}\n"
                        f"Reasoning: {adjustment.reasoning}\n\n"
                        f"Given this new development, what is your updated response? "
                        f"Respond as JSON."
                    )
                    histories[i] = histories[i] + [
                        {"role": "user", "content": sme_follow_up}
                    ]

                    # Incumbent responds
                    print(f"{label} — responding...", end="", flush=True)
                    pred, updated = query_agent_with_history(
                        client, args.model, agent, histories[i]
                    )
                    histories[i] = updated
                    print(f" → {pred.response_type} (intensity {pred.intensity}/5)")

                    round_entries.append((adjustment, pred))

                round_data.append({
                    "sme_move": None,
                    "predictions": round_entries,
                })

            print()

        display_multiround_results(num_rounds, round_data)


if __name__ == "__main__":
    main()
