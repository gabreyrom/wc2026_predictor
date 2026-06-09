"""
Exact Bracket Probability Propagation for the Knockout Stage.

Given P(team reaches each R32 slot), propagate win probabilities analytically
through R16 -> QF -> SF -> Final.

No sampling noise: P(A wins WC) = Σ over all paths P(A beats each opponent).

This is complementary to monte_carlo.py:
  - Use exact bracket propagation for top-team probabilities (no noise)
  - Use MC for 3rd-place qualification and tail probabilities
"""

import numpy as np
from itertools import combinations

from src.model.dixon_coles import DixonColesModel


# ── Win probability between two teams ────────────────────────────────────────

def p_win_knockout(
    model: DixonColesModel,
    team_i: str,
    team_j: str,
    max_goals: int = 7,
) -> float:
    """
    P(team_i beats team_j) in a knockout match (penalties if draw).
    = P(team_i wins in 90 min) + 0.5 * P(draw in 90 min)

    Passes match_importance=1.0 so the model uses the WC-knockout rho,
    which typically produces fewer 0-0/1-1 draws than group stage.
    """
    pred = model.predict(team_i, team_j, match_importance=1.0, max_goals=max_goals)
    return pred["home"] + 0.5 * pred["draw"]


# ── Bracket propagation ───────────────────────────────────────────────────────

class BracketPropagator:
    """
    Propagates team win probabilities through a single-elimination bracket.

    Usage:
        prop = BracketPropagator(model, slots)
        # slots: list of (team_name, initial_prob) for each R32 slot (32 entries)
        probs = prop.propagate()
        # Returns {team: P(wins WC)}
    """

    def __init__(
        self,
        model: DixonColesModel,
        slots: list[tuple[str, float]],
    ):
        """
        Args:
            model : fitted DixonColesModel
            slots : list of (team_name, prob_in_slot) for R32, length must be 2^k
        """
        assert len(slots) in {32, 16, 8, 4, 2}, \
            f"Expected bracket size 2^k, got {len(slots)}"
        self.model = model
        self.slots = slots
        self._n = len(slots)

    def propagate(self) -> dict[str, float]:
        """
        Propagate probabilities through all rounds.
        Returns {team: P(wins tournament)}.
        """
        # State: prob_reach[slot_idx][team] = P(this team is in slot idx at this round)
        # Initially: slot 0 has one team with its qualification probability
        #
        # But with multi-team slots (each slot can have multiple candidate teams),
        # we represent the state as a list of dicts {team: prob}.

        n_slots = self._n
        # State[i] = {team: P(team occupies slot i)}
        state: list[dict[str, float]] = []
        for team, p in self.slots:
            state.append({team: p} if team else {})

        # Track who-has-beaten-whom probabilities
        win_probs: dict[str, float] = {team: 0.0 for team, _ in self.slots}

        current = state
        round_size = n_slots

        round_names = {32: "R32", 16: "R16", 8: "QF", 4: "SF", 2: "Final"}

        while round_size >= 2:
            round_name = round_names.get(round_size, f"Round{round_size}")
            next_state: list[dict[str, float]] = []

            for i in range(0, round_size, 2):
                slot_a = current[i]    # {team: prob}
                slot_b = current[i+1]  # {team: prob}

                next_slot: dict[str, float] = {}

                for team_a, p_a in slot_a.items():
                    for team_b, p_b in slot_b.items():
                        if team_a == team_b:
                            continue
                        joint = p_a * p_b

                        p_ab = p_win_knockout(self.model, team_a, team_b)
                        p_ba = 1 - p_ab

                        # team_a advances
                        next_slot[team_a] = next_slot.get(team_a, 0.0) + joint * p_ab
                        # team_b advances
                        next_slot[team_b] = next_slot.get(team_b, 0.0) + joint * p_ba

                        # If this is the Final, record win probabilities
                        if round_size == 2:
                            win_probs[team_a] += joint * p_ab
                            win_probs[team_b] += joint * p_ba

                next_state.append(next_slot)

            current = next_state
            round_size //= 2

        return win_probs

    def reach_probs(self) -> dict[str, dict[str, float]]:
        """
        Compute P(team reaches each round) for all rounds.
        Returns {team: {round_name: prob}}.
        """
        n_slots = self._n
        round_names = ["R32", "R16", "QF", "SF", "Final", "Winner"]
        all_teams = [team for team, _ in self.slots if team]

        result: dict[str, dict[str, float]] = {
            t: {r: 0.0 for r in round_names} for t in all_teams
        }

        current = [{team: p} for team, p in self.slots]
        round_size = n_slots
        round_idx = 0

        while round_size >= 2:
            round_name = round_names[round_idx]
            next_state: list[dict[str, float]] = []

            for team_dict in current:
                for team, p in team_dict.items():
                    result[team][round_name] += p

            for i in range(0, round_size, 2):
                slot_a = current[i]
                slot_b = current[i+1]
                next_slot: dict[str, float] = {}

                for team_a, p_a in slot_a.items():
                    for team_b, p_b in slot_b.items():
                        if team_a == team_b:
                            continue
                        joint = p_a * p_b
                        p_ab = p_win_knockout(self.model, team_a, team_b)
                        p_ba = 1 - p_ab
                        next_slot[team_a] = next_slot.get(team_a, 0.0) + joint * p_ab
                        next_slot[team_b] = next_slot.get(team_b, 0.0) + joint * p_ba

                next_state.append(next_slot)

            current = next_state
            round_size //= 2
            round_idx += 1

        # Final winner
        for slot in current:
            for team, p in slot.items():
                result[team]["Winner"] = p

        return result


# ── Build R32 slots from MC + exact results ───────────────────────────────────

def build_r32_slots_from_mc(
    groups: dict[str, list[str]],
    mc_results,  # pd.DataFrame from monte_carlo.run_simulations
) -> list[tuple[str, float]]:
    """
    Build the 32 R32 slots as (team, probability) pairs from MC simulation results.
    MC already accounts for 3rd-place qualification.

    Bracket order: simplified pairing (group winners vs runners-up).
    Returns list of 32 (team, prob) pairs.
    """
    slots = []
    group_keys = sorted(groups.keys())

    # 1st place teams
    firsts = []
    for g in group_keys:
        for team in groups[g]:
            row = mc_results[mc_results["team"] == team]
            if not row.empty:
                # Use MC R32 probability weighted by group position
                firsts.append((team, float(row["R32"].values[0])))

    # Simplified: just use top MC teams to fill 32 slots
    top32 = mc_results.nlargest(32, "R32")[["team", "R32"]].values.tolist()
    return [(str(t), float(p)) for t, p in top32]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.model.dixon_coles import fit
    from src.data.fetch_matches import fetch_and_process
    from src.simulation.monte_carlo import run_simulations, ROUND_ORDER
    from tournament.wc2026_draw import GROUPS

    print("Loading data and fitting model...")
    df = fetch_and_process(force=False)
    model = fit(df, xi=0.003)

    print("\nRunning MC for bracket seeding...")
    mc = run_simulations(GROUPS, model, n=5_000, seed=42)

    # Build R32 slots and propagate
    slots = build_r32_slots_from_mc(GROUPS, mc)
    prop = BracketPropagator(model, slots)
    reach = prop.reach_probs()

    print("\n=== Exact Bracket Win Probabilities ===")
    sorted_teams = sorted(reach.items(), key=lambda x: -x[1].get("Winner", 0))
    for team, probs in sorted_teams[:16]:
        print(f"  {team:20s}  "
              f"QF:{probs['QF']:.1%}  SF:{probs['SF']:.1%}  "
              f"Final:{probs['Final']:.1%}  Win:{probs['Winner']:.1%}")
