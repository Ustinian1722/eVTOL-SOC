"""Prequential, mission-only SOC adapter update-utility study (22 CMU VTC6 cells).

Primary question: does a *label-free estimate of expected physics-consistency
improvement*, rather than raw voltage drift, improve **WHEN** to update?

Model initially trained on the target cell's first N=20 *normal mission*
cycles labelled by nominal 3Ah coulomb counting (NO RPT in initialization).
No runtime RPT curve, RPT capacity, SOH, SOC target, or life-cycle ID is read
by the adapter/update trigger. Past within-mission discharge differences
(delta-Q), V/I/T, and N20 source regression are the *only* update signals.

Test SOC_proxy is read AFTER the entire chronological replay solely for
offline scoring. That proxy IS historically RPT-capacity anchored. Therefore
scores are not independent BMS/electrochemical SOC validation.

The proxy is phase-end data; this is not a samplewise continuous SOC estimator.
See reports/14_无RPT更新价值门控诊断.md.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "soc_proxy_phase_labels.csv"
N20 = 20
NOMINAL_AH = 3.0
BUDGET = 4
WINDOW = 35
HOLDOUT = 8
COOLDOWN = 35
SEED = 20261012
FEATURES = [
    "phase_Ns", "duration_s", "I_mean_A", "I_peak_A",
    "Vmin_V", "Vmax_V", "Tmean_C", "Tmax_C",
]
VF = ["phase_Ns", "duration_s", "I_mean_A", "I_peak_A",
      "Vmax_V", "Tmean_C", "Tmax_C"]
ARMS = [
    ("frozen", "none", 3.0, None),
    ("periodic_C3", "periodic", 3.0, None),
    ("periodic_C27", "periodic", 2.7, None),
    ("periodic_C33", "periodic", 3.3, None),
    ("value_gate_005", "value", 3.0, .05),
    ("value_gate_010", "value", 3.0, .10),
    ("value_gate_020", "value", 3.0, .20),
    ("periodic_value_guard", "periodic_gate", 3.0, .10),
    ("voltage_alarm", "voltage", 3.0, None),
    ("random_B4", "random", 3.0, None),
]


def scale_robust(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median)))
    return median, max(1.4826 * mad, 0.2 * float(np.std(x)), 1e-3)


def source_fit(train: pd.DataFrame):
    m = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    m.fit(train[FEATURES], train["SOC_nominal_3Ah"])
    Z = np.c_[np.ones(len(train)), m.named_steps["standardscaler"].transform(train[FEATURES])]
    regr = m.named_steps["ridge"]
    frozen = np.r_[regr.intercept_, regr.coef_]
    y = train["SOC_nominal_3Ah"].to_numpy(dtype=float)
    vm = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    vm.fit(train[VF], train["Vmin_V"])
    vr = np.abs(vm.predict(train[VF]) - train["Vmin_V"].to_numpy(dtype=float)) * 1000.
    by_task = pd.DataFrame({"cycle":train["charge_cycle_index"], "err":vr}).groupby("cycle")["err"].mean()
    return m, Z, y, frozen, vm, scale_robust(by_task.to_numpy())


def mission_differences(X: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Only within-mission relative Q can constrain a slope, NOT SOC offset."""
    if len(q) < 2 or not np.all(np.isfinite(q)):
        return np.zeros((0, X.shape[1] - 1)), np.zeros(0)
    if np.any(np.diff(q) < -0.003):
        return np.zeros((0, X.shape[1] - 1)), np.zeros(0)
    dq = q[1:] - q[0]
    good = np.isfinite(dq) & (dq > .005) & (dq < 2.8)
    return (X[1:, 1:] - X[0, 1:])[good], dq[good]


def physics_residual(theta: np.ndarray, observations: list[dict], assumed_Ah: float) -> float:
    scores = []
    for item in observations:
        if len(item["dq"]):
            delta_pred = item["dZ"] @ theta[1:]
            scores.extend(np.abs(delta_pred + item["dq"] / assumed_Ah).tolist())
    return float(np.mean(scores)) if scores else float("nan")


def propose(theta: np.ndarray, frozen: np.ndarray, srcZ: np.ndarray,
            srcY: np.ndarray, old: list[dict], assumed_Ah: float):
    """Fit physics on *past training subset* + N20 anchors; no target SOC."""
    dz = [v["dZ"] for v in old if len(v["dq"])]
    dq = [v["dq"] for v in old if len(v["dq"])]
    if len(old) < 10 or sum(map(len, dq)) < 12:
        return None
    dZ = np.concatenate(dz)
    dQ = np.concatenate(dq)
    X = np.r_[srcZ, np.c_[np.zeros(len(dZ)), dZ]]
    y = np.r_[srcY, -dQ / assumed_Ah]
    w = np.r_[np.full(len(srcY), 2.), np.full(len(dQ), 6.)]
    prior = 15.0
    weighted = np.sqrt(w)
    xx = X * weighted[:, None]
    yy = y * weighted
    eye = np.eye(X.shape[1])
    candidate = np.linalg.solve(xx.T @ xx + prior * eye + 1e-9 * eye,
                                xx.T @ yy + prior * frozen)
    shift = candidate - frozen
    norm = float(np.linalg.norm(shift))
    if norm > 1.0:
        candidate = frozen + shift / norm
    return candidate if np.isfinite(candidate).all() else None


def candidate_diagnostics(theta: np.ndarray, candidate: np.ndarray,
                          frozen: np.ndarray, srcZ: np.ndarray,
                          srcY: np.ndarray, holdout: list[dict], assumed_Ah: float):
    before = physics_residual(theta, holdout, assumed_Ah)
    after = physics_residual(candidate, holdout, assumed_Ah)
    source_before = float(np.mean(np.abs(srcZ @ theta - srcY)))
    source_after = float(np.mean(np.abs(srcZ @ candidate - srcY)))
    rel_gain = 0 if not np.isfinite(before) or before < 0.004 else (before-after)/before
    return {
        "physics_loss_before": float(before),
        "physics_loss_after": float(after),
        "physics_rel_gain": float(rel_gain),
        "source_MAE_before": source_before,
        "source_MAE_after": source_after,
        "source_degradation": source_after-source_before,
    }


@dataclass
class ReplayInputs:
    sourceZ: np.ndarray
    sourceY: np.ndarray
    frozen: np.ndarray
    missions: list[dict]
    v_center: float
    v_scale: float


def prepare(cell: pd.DataFrame) -> ReplayInputs:
    cell = cell.sort_values(["charge_cycle_index","phase_Ns"]).copy()
    cy = cell["charge_cycle_index"].to_numpy(dtype=int)
    ids = np.unique(cy)
    if len(ids) <= N20:
        raise ValueError("Too few mission cycles")
    training = cell[cy <= ids[N20-1]]
    test = cell[cy > ids[N20-1]]
    model, srcZ, srcY, frozen, vmodel, vstats = source_fit(training)
    Z = np.c_[np.ones(len(test)),
              model.named_steps["standardscaler"].transform(test[FEATURES])]
    q = test["QDischarge_end_Ah"].to_numpy(dtype=float)
    vt = test["Vmin_V"].to_numpy(dtype=float)
    vr = vmodel.predict(test[VF])
    yscore = test["SOC_proxy"].to_numpy(dtype=float)  # SCORING ONLY, not stored with mission records
    test_ids = test["charge_cycle_index"].to_numpy(dtype=int)
    edges = np.r_[0, np.flatnonzero(np.diff(test_ids)) + 1, len(test_ids)]
    missions = []
    for start, stop in zip(edges[:-1], edges[1:]):
        dZ, dq = mission_differences(Z[start:stop], q[start:stop])
        missions.append({
            "cycle": int(test_ids[start]),
            "Z": Z[start:stop],
            "dZ": dZ,
            "dq": dq,
            "voltage_abs_resid_mV": float(np.abs(vt[start:stop] - vr[start:stop]).mean() * 1000),
            "score_after_replay": yscore[start:stop],  # never accessed by replay updater
        })
    return ReplayInputs(srcZ, srcY, frozen, missions, *vstats)


def fit_and_gate(theta: np.ndarray, inputs: ReplayInputs, history: list[dict], C: float,
                 min_gain: float):
    if len(history) < 20:
        return None, None, "insufficient_completed_missions"
    training = history[:-HOLDOUT]
    holdout = history[-HOLDOUT:]
    candidate = propose(theta, inputs.frozen, inputs.sourceZ,
                        inputs.sourceY, training, C)
    if candidate is None:
        return None, None, "insufficient_physics_pairs"
    diag = candidate_diagnostics(theta, candidate, inputs.frozen,
                                 inputs.sourceZ, inputs.sourceY, holdout, C)
    if (not np.isfinite(diag["physics_loss_after"]) or
        diag["physics_rel_gain"] < min_gain or
        diag["source_degradation"] > 0.005):
        return candidate, diag, "guard_rejected"
    return candidate, diag, "gate_accepted"


def evaluate_one(inputs: ReplayInputs, policy: tuple, cell_id: str):
    name, trigger, C, min_gain = policy
    theta = inputs.frozen.copy()
    buffer = deque(maxlen=WINDOW)
    rng = np.random.default_rng(SEED+int(hashlib.sha256(cell_id.encode()).hexdigest()[:8],16))
    last_event, attempted, accepted = 0, 0, 0
    alarm_count=0
    chunks_pred = []
    eventlog = []
    for pos,mission in enumerate(inputs.missions, start=1):
        # Strict online order: PREDICT now; sensor measurements can update later.
        chunks_pred.append(np.clip(mission["Z"] @ theta, 0, 1))
        # Done with prediction. Only a sanitized, label-free copy reaches the update logic.
        item = {k:mission[k] for k in ["cycle","Z","dZ","dq","voltage_abs_resid_mV"]}
        buffer.append(item)
        z=(item["voltage_abs_resid_mV"]-inputs.v_center)/inputs.v_scale
        alarm_count = (alarm_count+1) if z > 3 else 0
        if name=="frozen" or attempted>=BUDGET or pos-last_event < COOLDOWN:
            continue
        enough = len(buffer)>=WINDOW
        fire = (
            (trigger in {"periodic","periodic_gate"} and pos % 150 == 0) or
            (trigger=="value" and pos % 20 == 0) or
            (trigger=="voltage" and alarm_count>=3) or
            (trigger=="random" and rng.random()<1/150)
        )
        if not enough or not fire:
            continue
        candidate,diag,status=fit_and_gate(
            theta, inputs, list(buffer), C,
            float(min_gain) if min_gain is not None else -float("inf")
        )
        attempted+=1
        updated = False
        if candidate is not None and (
            trigger not in {"value","periodic_gate"} or status=="gate_accepted"
        ):
            theta = candidate
            accepted+=1
            updated = True
        eventlog.append({
            "cell_id":cell_id, "policy":name,
            "mission_ordinal":pos, "cycle":item["cycle"],
            "attempted_update_no":attempted, "updated":int(updated),
            "reason":status,
            **(diag or {}),
        })
        if updated:
            last_event=pos
        alarm_count=0

    # Offline scoring in separate code path; no label enters update proposals.
    prediction = np.concatenate(chunks_pred)
    truth = np.concatenate([m["score_after_replay"] for m in inputs.missions])
    errors_pp=100*(prediction-truth)
    percycle = np.asarray([
        float(np.mean(np.abs(p-m["score_after_replay"]))*100)
        for p,m in zip(chunks_pred, inputs.missions)
    ])
    late_start=int(.75*len(percycle))
    return {
        "cell_id":cell_id, "arm":name,
        "capacity_prior_Ah":C,
        "missions":len(inputs.missions),
        "attempted_updates":attempted,
        "accepted_updates":accepted,
        "phase_rows":len(prediction),
        "MAE_pp":float(np.mean(np.abs(errors_pp))),
        "RMSE_pp":float(np.sqrt(np.mean(errors_pp**2))),
        "late_MAE_pp":float(np.mean(percycle[late_start:])),
        "P95_cycle_MAE_pp":float(np.quantile(percycle,.95)),
        "max_cycle_MAE_pp":float(np.max(percycle)),
    },eventlog,prediction


def summarize(metrics: pd.DataFrame):
    frozen = metrics[metrics.arm=="frozen"].set_index("cell_id")
    paired=metrics.join(frozen[["MAE_pp","late_MAE_pp"]],on="cell_id",rsuffix="_frozen")
    paired["delta_pp"]=paired.MAE_pp-paired.MAE_pp_frozen
    paired["late_delta_pp"]=paired.late_MAE_pp-paired.late_MAE_pp_frozen
    macro=paired.groupby("arm",as_index=False).agg(
        n_cells=("cell_id","nunique"),
        median_MAE_pp=("MAE_pp","median"),
        mean_MAE_pp=("MAE_pp","mean"),
        median_late_MAE_pp=("late_MAE_pp","median"),
        mean_paired_delta_pp=("delta_pp","mean"),
        improved_cells=("delta_pp", lambda x: int((x<0).sum())),
        med_attempts=("attempted_updates","median"),
        med_accepted=("accepted_updates","median"),
    )
    ci=[]
    for arm,g in paired.groupby("arm"):
        d=g.delta_pp.to_numpy()
        rng=np.random.default_rng(SEED)
        boot=rng.choice(d,size=(6000,len(d)),replace=True).mean(axis=1)
        lo,hi=np.quantile(boot,[.025,.975])
        ci.append({"arm":arm,"CI_low":lo,"CI_high":hi})
    macro=macro.merge(pd.DataFrame(ci),on="arm")
    return paired,macro


def selected_cell_protocol(cell_ids: list[str]) -> dict[str,str]:
    """Stable ID-only retrospective dev/challenge partition, not fresh lockbox:
    earlier exploratory experiments already examined all 22 cells."""
    ids=sorted(cell_ids,key=lambda c: hashlib.sha256(("split_"+c).encode()).hexdigest())
    return {cell:("development" if i < 14 else "retrospective_challenge")
            for i,cell in enumerate(ids)}


def report(paired:pd.DataFrame,summary:pd.DataFrame,events:pd.DataFrame)->str:
    lines=[
        "# Mission-only update utility test (SOC proxy; N20 nominal-3Ah)",
        "",
        "- 22 cells, initial N20 target missions supervised by nominal-3Ah coulomb count; zero RPT used in training or ONLINE updates.",
        "- Offline SOC_proxy SCORE still uses past RPT capacity by its historical definition, not independent true SOC.",
        "- Future/current SOC proxy cannot affect prediction updates, update gates, or triggers.",
        "- One-sided chronological replay: predict mission, observe mission, optionally update future missions.",
        "- B=4 is maximum attempted updates (value gate may accept fewer); stream normalization and RPT are absent.",
        "- Utility gate evaluates held-out completed-mission physical loss with an N20 source-regression guard.",
        "- All triggers and physics capacity assumptions fixed prior to target score; nominal 3Ah is not a learned aged capacity.",
        "",
        "## Whole-cohort paired outcome",
        "| Method | Median MAE (pp) | Mean Δ vs frozen (pp) [95% cell CI] | Better cells | Median attempted / accepted |",
        "|---|---:|---:|---:|---:|",
    ]
    for x in summary.sort_values("median_MAE_pp").itertuples():
        lines.append(
            f"| {x.arm} | {x.median_MAE_pp:.3f} | {x.mean_paired_delta_pp:+.3f} "
            f"[{x.CI_low:+.3f},{x.CI_high:+.3f}] | {x.improved_cells}/{x.n_cells} | "
            f"{x.med_attempts:.1f}/{x.med_accepted:.1f} |"
        )
    lines += [
        "", "## Retrospective ID-only split (NOT a never-seen final test)",
        "",
        "Cell memberships use stable SHA256 of cell ID; earlier work already inspected these same 22 cells.",
        "This is a diagnostic for stability and protocol composition, not independent generalization proof.",
        "",
        "| Cohort | Policy | Cells | Median MAE | Mean Δ vs frozen |",
        "|---|---|---:|---:|---:|",
    ]
    for (cohort,arm),g in paired.groupby(["cohort","arm"]):
        lines.append(f"| {cohort} | {arm} | {len(g)} | {g.MAE_pp.median():.3f} | {g.delta_pp.mean():+.3f} |")
    lines.extend([
        "",
        "## UPDATE events and caution",
        f"- Total attempted: {len(events)}, accepted: {int(events.updated.sum()) if len(events) else 0}.",
        "- A gate rejecting updates may approximate Frozen; only *accepted* updates count as actual adaptation.",
        "- Value gate's candidate evaluated on already completed mission data; test SOC target never used for gating.",
        "- Capacity-prior sensitivity (2.7/3.0/3.3 Ah) probes mis-specification, not an online SOH estimator.",
        "- Event-time updates and per-cell scatter available in artifact CSV.",
        "- Best retrospective arm must NOT be treated as independently selected or state-of-art.",
    ])
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(DATA)
    required=set(FEATURES+VF+["cell_id","charge_cycle_index","SOC_nominal_3Ah","SOC_proxy","QDischarge_end_Ah"])
    if not required.issubset(raw):
        raise ValueError(f"Data missing expected columns: {required - set(raw)}")
    if raw["cell_id"].nunique()!=22: raise AssertionError("Expected 22 physical cells")
    partition=selected_cell_protocol(raw.cell_id.unique().tolist())
    rows=[];logs=[]
    for cell_id,group in raw.groupby("cell_id",sort=True):
        info=prepare(group)
        for spec in ARMS:
            result,events,_=evaluate_one(info,spec,cell_id)
            rows.append(result);logs+=events
    metrics=pd.DataFrame(rows)
    paired,macro=summarize(metrics)
    paired["cohort"]=paired.cell_id.map(partition)
    eventdf=pd.DataFrame(logs)
    assert len(metrics)==22*len(ARMS)
    assert metrics.accepted_updates.le(BUDGET).all()
    assert metrics.attempted_updates.le(BUDGET).all()
    assert metrics.accepted_updates.le(metrics.attempted_updates).all()
    assert metrics.query("arm=='frozen'").MAE_pp.notna().all()
    paired.to_csv(out/"utility_by_cell.csv",index=False)
    macro.to_csv(out/"utility_summary.csv",index=False)
    eventdf.to_csv(out/"utility_events.csv",index=False)
    (out/"report.md").write_text(report(paired,macro,eventdf),encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({
        "dataset":"CMU eVTOL VTC6 mission phases",
        "rpt_allowed_online":False,
        "initial_SOC_labels":"nominal_3Ah mission coulomb count, no RPT",
        "offline_score":"SOC_proxy from historical prior RPT, reporting proxy MAE only",
        "mission_sensor_observations_for_updates":["V/I/T phase statistics","within-mission QDischarge differences"],
        "runtime_never_uses":["SOC_proxy","RPT capacity","SOH","future measurements","life_stage"],
        "source_N20_missions":N20,
        "max_attempted_updates":BUDGET,
        "candidate_fit_buffer_missions":WINDOW-HOLDOUT,
        "completed_mission_validation_block":HOLDOUT,
        "source_guard_MAE_increase_limit":0.005,
        "protocol":"predict -> observe completed mission -> candidate train on prior subset -> physics guard on heldout completed missions -> update next",
        "cells":len(partition),
        "retrospective_split":partition,
        "warning":"Split is NOT pristine because all cells used in previous analyses",
        "policies":ARMS,
    },indent=2,ensure_ascii=False),encoding="utf-8")
    print((out/"report.md").read_text(encoding="utf-8"))


if __name__=="__main__":
    main()
