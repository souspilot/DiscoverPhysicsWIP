"""
dp_parse_json.py — Parse DiscoverPhysics JSON artefacts into RunRecord objects.

The JSON artefact is fully structured, so this replaces the brittle text parser.
It produces the SAME RunRecord/Round dataclasses used by dp_probe.py, plus a few
extra fields the JSON exposes that the .txt logs did not:

  - Round.action            : typed action label ('experiment' / 'final_law' / ...)
  - Round.critic_feedback   : critic text if the critic was on
  - RunRecord.gt_trajectories : ground-truth + predicted test trajectories from
                                evaluation.trajectories (times, gt, pred, error).
                                This means behavioral comparison against ground
                                truth needs no simulator for the eval cases.
  - RunRecord.judge_reasoning : the judge's free-text reasoning
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any


@dataclass
class Round:
    index: int
    experiment_input: Optional[list]
    experiment_output: Optional[list]
    llm_reply_text: str = ""
    action: Optional[str] = None
    experiment_error: Optional[str] = None
    critic_feedback: Optional[str] = None


@dataclass
class RunRecord:
    world: str
    model: str
    seed: Optional[int]
    noise: Optional[float]
    rounds: list = field(default_factory=list)
    final_law_src: Optional[str] = None
    explanation_text: Optional[str] = None
    mse_result: Optional[str] = None
    mean_particle_mse: Optional[float] = None
    max_particle_mse: Optional[float] = None
    explanation_score: Optional[float] = None
    explanation_score_raw: Optional[int] = None
    optimal_explanation: Optional[str] = None
    judge_reasoning: Optional[str] = None
    gt_trajectories: Optional[list] = None   # list of {case,times,p1,p2,gt,pred,error}
    fitted_params: Optional[dict] = None
    source_path: Optional[str] = None


def parse_run_json(path):
    d = json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))

    rounds = []
    for rd in d.get("rounds", []):
        rounds.append(Round(
            index=rd.get("round"),
            experiment_input=rd.get("experiment_input"),
            experiment_output=rd.get("experiment_output"),
            llm_reply_text=(rd.get("llm_reply") or "").strip(),
            action=rd.get("action"),
            experiment_error=rd.get("experiment_error"),
            critic_feedback=rd.get("critic_feedback"),
        ))

    ev = d.get("evaluation") or {}
    expl = ev.get("explanation") or {}
    fit = ev.get("fit") or {}

    passed = ev.get("passed")
    mse_result = None if passed is None else ("PASS" if passed else "FAIL")

    return RunRecord(
        world=d.get("world"),
        model=d.get("model"),
        seed=d.get("noise_seed"),
        noise=d.get("noise_std"),
        rounds=rounds,
        final_law_src=d.get("final_law"),
        explanation_text=expl.get("agent_explanation"),
        mse_result=mse_result,
        mean_particle_mse=ev.get("mean_pos_error"),
        max_particle_mse=ev.get("max_pos_error"),
        explanation_score=expl.get("score"),
        explanation_score_raw=expl.get("raw_score"),
        optimal_explanation=expl.get("optimal_explanation"),
        judge_reasoning=expl.get("reasoning"),
        gt_trajectories=ev.get("trajectories"),
        fitted_params=fit.get("fitted_params"),
        source_path=str(path),
    )


def parse_dir(dirpath, pattern="*_seed*.json"):
    records = []
    for p in sorted(Path(dirpath).glob(pattern)):
        try:
            records.append(parse_run_json(p))
        except Exception as e:
            print(f"[WARN] failed to parse {p.name}: {e}")
    return records


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    recs = parse_dir(d)
    for r in recs:
        n_out = sum(1 for rd in r.rounds if rd.experiment_output)
        actions = [rd.action for rd in r.rounds]
        print(f"{r.world:16s} seed={r.seed} rounds={len(r.rounds):2d} "
              f"exp_out={n_out} law={'Y' if r.final_law_src else 'N'} "
              f"MSE={r.mse_result} mean={r.mean_particle_mse:.4g} "
              f"expl={r.explanation_score} gt_traj={'Y' if r.gt_trajectories else 'N'} "
              f"actions={actions}")