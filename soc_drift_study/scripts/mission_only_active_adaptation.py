"""Mission-only, label-free online SOC-proxy adaptation benchmark.

Research prototype, NOT a reproduction of the cited neural TTA / battery papers.

Training: the first 20 *mission* cycles per physical cell (two initialization
tracks: historical SOC_proxy or nominal-3Ah CC). After that:
 - all triggers/updates read only completed mission V/I/T phase summaries and
   within-mission *differences* in QDischarge_end_Ah (no RPT, SOH, future SOC).
 - predictions are committed BEFORE the completed mission updates the state;
 - any test SOC_proxy is read ONLY in the scoring function AFTER replay;
 - no RPT rows, low-rate reference samples or capacity_used_Ah enter adaptation.
SOC_proxy is still RPT-anchored in the OFFLINE scoring labels. Treat as a proxy.

Several lightweight ideas are *inspired by*, not identical reimplementations of:
online RLS, unlabeled domain adaptation, self-training, self-supervised temporal
consistency, drift detection and continual test-time adaptation.
"""
from __future__ import annotations

import argparse
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
RESULTS = ROOT / "results"
FEATURES = [
    "phase_Ns", "duration_s", "I_mean_A", "I_peak_A",
    "Vmin_V", "Vmax_V", "Tmean_C", "Tmax_C",
]
V_FEATURES = [
    "phase_Ns", "duration_s", "I_mean_A", "I_peak_A",
    "Vmax_V", "Tmean_C", "Tmax_C",
]
N20 = 20
NOMINAL_AH = 3.0
BUDGET = 4
UPDATE_COOLDOWN = 35
WINDOW = 35
MIN_WINDOW = 10
SEED = 20261012

# A fixed grid chosen WITHOUT looking at target errors. Arm/trigger variants
# share exactly the same evaluation cells and chronological mission stream.
ARMS = [
    ("frozen", "none", "none"),
    ("periodic_norm", "norm", "periodic"),
    ("periodic_covshift", "covshift", "periodic"),
    ("periodic_pseudo", "pseudo", "periodic"),
    ("periodic_physics", "physics", "periodic"),
    ("periodic_physics_guard", "guarded", "periodic"),
    ("voltage_physics", "physics", "voltage"),
    ("ph_voltage_physics", "physics", "ph_voltage"),
    ("cusum_voltage_physics", "physics", "cusum_voltage"),
    ("disagreement_physics", "physics", "disagreement"),
    ("joint_physics", "physics", "joint"),
    ("random_physics", "physics", "random"),
    ("stream_physics", "physics", "stream"),
]


def robust_stats(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    center = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - center)))
    scale = max(1.4826 * mad, 0.2 * float(np.nanstd(values)), 1e-4)
    return center, scale


def ridge_theta(design: np.ndarray, target: np.ndarray, weights: np.ndarray,
                prior: np.ndarray, anchor: float = 10.0) -> np.ndarray:
    """Least squares with prior centered on frozen initial parameters."""
    sw = np.sqrt(np.asarray(weights, dtype=float))
    x = design * sw[:, None]
    y = target * sw
    lam = np.eye(design.shape[1]) * anchor
    # Intercept should not be excessively regularized relative to source data.
    lhs = x.T @ x + lam
    rhs = x.T @ y + lam @ prior
    return np.linalg.solve(lhs + 1e-9 * np.eye(len(prior)), rhs)


def relative_rows(Z: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Causal within-mission charge-conservation: SOC_j-SOC_first=-dQ/3Ah.

    No absolute accumulated Q or RPT capacity is used in updates. All rows
    become available only when the entire mission has finished.
    """
    if len(q) < 2 or not np.all(np.isfinite(q)):
        return np.empty((0, Z.shape[1])), np.empty(0)
    dq = q[1:] - q[0]
    if (np.diff(q) < -0.003).any():
        return np.empty((0, Z.shape[1])), np.empty(0)
    valid = np.isfinite(dq) & (dq > 0.005) & (dq < 2.8)
    dz = Z[1:] - Z[0]
    return dz[valid], -dq[valid] / NOMINAL_AH


def mission_consistency(theta: np.ndarray, Z: np.ndarray, q: np.ndarray) -> float:
    dx, dy = relative_rows(Z, q)
    if not len(dy):
        return float("nan")
    return float(np.mean(np.abs(dx @ theta[1:] - dy)))


def safe_relative_update(
    old: np.ndarray, source: np.ndarray, y_source: np.ndarray,
    buffer: list[dict], kind: str, frozen: np.ndarray,
) -> tuple[np.ndarray, bool, str]:
    """Use only completed target *mission* observations in any adaptation."""
    if len(buffer) < MIN_WINDOW:
        return old.copy(), False, "short_buffer"
    fit_buffer = buffer[:-5] if kind == "guarded" else buffer
    if len(fit_buffer) < 5:
        return old.copy(), False, "short_buffer"
    dZ = np.concatenate([v["dZ"] for v in fit_buffer if len(v["dy"])], axis=0)
    dy = np.concatenate([v["dy"] for v in fit_buffer if len(v["dy"])], axis=0)
    if len(dy) < 12:
        return old.copy(), False, "insufficient_physical_excitation"

    if kind in {"physics", "guarded"}:
        dx = np.concatenate([np.zeros((len(dy), 1)), dZ], axis=1)
        design = np.concatenate([source, dx], axis=0)
        target = np.concatenate([y_source, dy])
        weight = np.r_[np.full(len(y_source), 2.0), np.full(len(dy), 6.0)]
        candidate = ridge_theta(design, target, weight, frozen, anchor=15.0)
    elif kind == "pseudo":
        pseudo_z = np.concatenate([v["Z"][0] + v["dZ"] for v in fit_buffer if len(v["dy"])], axis=0)
        pseudo_y = np.concatenate([
            np.clip((np.r_[1.0, v["Z"][0]] @ frozen) + v["dy"], 0, 1)
            for v in fit_buffer if len(v["dy"])
        ])
        dx = np.column_stack([np.ones(len(pseudo_y)), pseudo_z])
        candidate = ridge_theta(
            np.r_[source, dx],
            np.r_[y_source, pseudo_y],
            np.r_[np.full(len(y_source), 2.0), np.full(len(pseudo_y), 0.5)],
            frozen,
            anchor=15,
        )
    elif kind == "covshift":
        zs = source[:, 1:]
        target_z = np.concatenate([v["Z"] for v in fit_buffer], axis=0)
        # Unsupervised source reweighting towards recent target input support.
        # Standardized pairwise Euclidean similarity, no target SOC labels.
        diff = zs[:, None, :] - target_z[None, :, :]
        d2 = np.mean(diff * diff, axis=2)
        closeness = np.mean(np.exp(-0.5 * np.clip(d2, 0, 80)), axis=1)
        weight = np.clip(closeness / max(float(closeness.mean()), 1e-6), 0.2, 3.0)
        candidate = ridge_theta(source, y_source, weight, frozen, anchor=10)
    else:
        raise ValueError(f"Unknown policy {kind}")

    # Fixed parameter trust region, not selected on target SOC-proxy.
    shift = candidate - frozen
    displacement = float(np.linalg.norm(shift))
    if displacement > 1.0:
        candidate = frozen + shift / displacement
    if not np.all(np.isfinite(candidate)):
        return old.copy(), False, "nonfinite_candidate"
    if kind == "guarded":
        holdout = buffer[-5:]
        old_err = np.nanmean([mission_consistency(old, v["Z"], v["q"]) for v in holdout])
        new_err = np.nanmean([mission_consistency(candidate, v["Z"], v["q"]) for v in holdout])
        source_before = float(np.mean(np.abs(source @ old - y_source)))
        source_after = float(np.mean(np.abs(source @ candidate - y_source)))
        if (not np.isfinite(old_err) or not np.isfinite(new_err)
                or new_err >= 0.98 * old_err
                or source_after - source_before > 0.002):
            return old.copy(), False, "holdout_physics_or_source_guard"
    return candidate, True, "updated"


@dataclass
class Detector:
    v_center: float
    v_scale: float
    d_center: float
    d_scale: float
    ph: float = 0.0
    cusum: float = 0.0
    above_v: int = 0
    above_d: int = 0

    def step(self, v: float, d: float) -> dict[str, bool]:
        vz = (v - self.v_center) / self.v_scale
        dz = (d - self.d_center) / self.d_scale
        self.above_v = self.above_v + 1 if vz > 3.0 else 0
        self.above_d = self.above_d + 1 if dz > 3.0 else 0
        self.ph = max(0.0, self.ph + vz - 0.5)
        self.cusum = max(0.0, self.cusum + vz - 1.0)
        fired = {
            "voltage": self.above_v >= 3,
            "disagreement": self.above_d >= 3,
            "ph_voltage": self.ph >= 12,
            "cusum_voltage": self.cusum >= 20,
            "joint": self.above_d >= 2 and self.above_v >= 2,
        }
        return fired

    def reset_alarm(self) -> None:
        self.ph = self.cusum = 0.0
        self.above_v = self.above_d = 0


def _matrix(model, frame: pd.DataFrame) -> np.ndarray:
    z = model.named_steps["standardscaler"].transform(frame[FEATURES])
    return np.c_[np.ones(len(z)), z]


def _fit_source(train: pd.DataFrame, label: str):
    base = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    base.fit(train[FEATURES], train[label])
    design = _matrix(base, train)
    regr = base.named_steps["ridge"]
    initial = np.r_[regr.intercept_, regr.coef_]
    voltage = make_pipeline(StandardScaler(), Ridge(alpha=5.0))
    voltage.fit(train[V_FEATURES], train["Vmin_V"])
    return design, train[label].to_numpy(dtype=float), initial, voltage, base


def _training_signals(train: pd.DataFrame, base, voltage) -> tuple[tuple[float,float],tuple[float,float]]:
    residual = np.abs(train["Vmin_V"].to_numpy() - voltage.predict(train[V_FEATURES])) * 1000
    bycycle = pd.DataFrame({"cycle": train["charge_cycle_index"], "v": residual}).groupby("cycle").v.mean()
    phys = []
    for _, g in train.groupby("charge_cycle_index", sort=True):
        Z = _matrix(base, g)
        q = g["QDischarge_end_Ah"].to_numpy(dtype=float)
        phys.append(mission_consistency(
            np.r_[base.named_steps["ridge"].intercept_,base.named_steps["ridge"].coef_], Z[:,1:], q
        ))
    return robust_stats(bycycle.to_numpy()), robust_stats(np.asarray(phys)[np.isfinite(phys)])


def replay_cell(cell: pd.DataFrame, target_track: str, arm: tuple[str,str,str]) -> tuple[dict,list[dict]]:
    name, kind, trigger = arm
    cell = cell.sort_values(["charge_cycle_index", "phase_Ns"]).copy()
    ids = sorted(int(x) for x in cell["charge_cycle_index"].unique())
    early = set(ids[:N20])
    train = cell[cell["charge_cycle_index"].isin(early)]
    test = cell[~cell["charge_cycle_index"].isin(early)]
    source, ysource, frozen, voltage, base = _fit_source(train, target_track)
    v_stats, d_stats = _training_signals(train, base, voltage)
    detector = Detector(*v_stats, *d_stats)
    theta = frozen.copy()
    last_update = 0
    attempts = accepted = rejected = 0
    buffer = deque(maxlen=WINDOW)
    z_window = deque(maxlen=WINDOW)
    norm_offset = None  # updated only after an accepted mission-only event
    predictions = []
    # Vectorized one-time feature transforms: O(tasks) tiny NumPy operations
    # rather than 500k pandas/scikit transformations in CI. Same protocol.
    allZ = _matrix(base, test)
    index_values = test["charge_cycle_index"].to_numpy(dtype=int)
    boundaries = np.r_[0, np.flatnonzero(np.diff(index_values)) + 1, len(test)]
    allQ = test["QDischarge_end_Ah"].to_numpy(dtype=float)
    allSocScoreOnly = test["SOC_proxy"].to_numpy(dtype=float)
    allPhase = test["phase_Ns"].to_numpy(dtype=int)
    allVoltage = test["Vmin_V"].to_numpy(dtype=float)
    voltagePreds = voltage.predict(test[V_FEATURES])
    rng = np.random.default_rng(SEED + sum(ord(c) for c in str(cell.cell_id.iloc[0])))
    task_total = 0
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        cycle = int(index_values[start])
        task_total += 1
        Zfull = allZ[start:stop]
        Z = Zfull[:, 1:]
        q = allQ[start:stop]
        # Feature normalization uses *past* completed missions only.
        if kind == "norm" and norm_offset is not None:
            current_prediction = np.clip(theta[0] + (Z - norm_offset) @ theta[1:], 0, 1)
        else:
            current_prediction = np.clip(Zfull @ theta, 0, 1)
        predictions.append({
            "cycle": int(cycle),
            "errors_pp": (current_prediction -
                          allSocScoreOnly[start:stop]) * 100,
            "phase": allPhase[start:stop],
            "pred": current_prediction,
        })
        # Only after prediction: mission's sensor/throughput measurements may
        # enter trigger and update buffers. Evaluation labels are NOT read.
        frozen_prediction = np.clip(Zfull @ frozen, 0, 1)
        v_score = float(np.mean(np.abs(
            allVoltage[start:stop] - voltagePreds[start:stop]
        ) * 1000))
        d_score = mission_consistency(frozen, Z, q)
        if not np.isfinite(d_score):
            d_score = d_stats[0]
        dZ, dy = relative_rows(Z, q)
        buffer.append({"Z": Z, "q": q, "dZ": dZ, "dy": dy})
        z_window.append(Z)
        flags = detector.step(v_score, d_score)
        eligible = task_total - last_update >= UPDATE_COOLDOWN and len(buffer) >= MIN_WINDOW
        if kind == "none":
            fire = False
        elif trigger == "stream":
            fire = eligible and (task_total % 40 == 0)
        elif trigger == "periodic":
            fire = eligible and task_total % 150 == 0
        elif trigger == "random":
            fire = eligible and rng.random() < 1 / 150
        else:
            fire = eligible and flags.get(trigger, False)
        if not fire or (trigger != "stream" and attempts >= BUDGET):
            continue
        attempts += 1
        if kind == "norm":
            source_mean = source[:,1:].mean(axis=0)
            stream_mean = np.concatenate(list(z_window)).mean(axis=0)
            norm_offset = 0.15 * np.clip(stream_mean - source_mean, -1.0, 1.0)
            accepted += 1  # only future missions see frozen offset
            reason = "norm_statistics_updated"
        else:
            theta_new, ok, reason = safe_relative_update(
                theta, source, ysource, list(buffer), kind, frozen
            )
            if ok:
                theta = theta_new
                accepted += 1
            else:
                rejected += 1
        last_update = task_total
        detector.reset_alarm()

    errors = np.concatenate([v["errors_pp"] for v in predictions])
    percycle = np.array([np.mean(np.abs(v["errors_pp"])) for v in predictions])
    late = percycle[max(0,int(0.75*len(percycle))):]
    return {
        "cell_id": str(cell.cell_id.iloc[0]), "train_label_track": target_track,
        "arm": name, "update_type": kind, "trigger": trigger,
        "test_missions": task_total, "test_phase_rows": len(errors),
        "attempted_updates": attempts, "accepted_updates": accepted,
        "rejected_updates": rejected,
        "proxy_MAE_pp": float(np.mean(np.abs(errors))),
        "proxy_RMSE_pp": float(np.sqrt(np.mean(errors**2))),
        "late_cycle_MAE_pp": float(np.mean(late)),
        "P95_cycle_MAE_pp": float(np.quantile(percycle,.95)),
    }, []


def summarize(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frozen = metrics[metrics.arm.eq("frozen")].set_index(["train_label_track","cell_id"])
    joined = metrics.join(
        frozen[["proxy_MAE_pp"]], on=["train_label_track","cell_id"],rsuffix="_frozen"
    )
    joined["delta_MAE_pp"] = joined.proxy_MAE_pp - joined.proxy_MAE_pp_frozen
    summary = joined.groupby(["train_label_track","arm","update_type","trigger"],as_index=False).agg(
        cells=("cell_id","nunique"),
        median_MAE_pp=("proxy_MAE_pp","median"),
        mean_MAE_pp=("proxy_MAE_pp","mean"),
        median_late_MAE_pp=("late_cycle_MAE_pp","median"),
        median_P95_cycle_MAE_pp=("P95_cycle_MAE_pp","median"),
        mean_paired_delta_pp=("delta_MAE_pp","mean"),
        improved_cells=("delta_MAE_pp",lambda x: int((x < 0).sum())),
        median_attempts=("attempted_updates","median"),
        median_accepted=("accepted_updates","median"),
    )
    cis = []
    for keys,g in joined.groupby(["train_label_track","arm"]):
        delta = g["delta_MAE_pp"].to_numpy()
        rng = np.random.default_rng(SEED)
        boots = rng.choice(delta,size=(5000,len(delta)), replace=True).mean(axis=1)
        lo,hi=np.quantile(boots,[.025,.975])
        cis.append({"train_label_track":keys[0],"arm":keys[1],
                    "cell_bootstrap_lo":float(lo),"cell_bootstrap_hi":float(hi)})
    summary=summary.merge(pd.DataFrame(cis),on=["train_label_track","arm"])
    return joined,summary


def _report(summary:pd.DataFrame,metric:pd.DataFrame) -> str:
    lines=[
        "# Mission-only online SOC adaptation: frozen comparisons",
        "",
        "This table is a PROXY evaluation using past-RPT-anchored SOC_proxy labels. ",
        "After N20, no RPT sample/capacity, SOC_proxy, SOH, future label or full-life index is passed to update/trigger logic.",
        "All predictions are committed before their own mission contributes features/charge increments to adaptation.",
        "N20 initialization: two tracks, one with original historical SOC_proxy training labels (past RPT anchor),",
        "one with nominal-3Ah mission coulomb count (no RPT even in model training).",
        "Do NOT call these exact reproductions of cited published online adaptation architectures.",
        "Nominal 3Ah enters only as fixed within-mission dSOC/dQ constraint, not a newly measured capacity.",
        "Detector thresholds are computed on N20 and fixed; all algorithm parameters are predeclared.",
        "A budget is a max number of attempted refreshes, not a guarantee of equal refresh counts;",
        "streaming is an intentionally more expensive non-budget-matched reference.",
        "All metrics below are percentage points; lower is better.",
        "",
    ]
    for label,g in summary.groupby("train_label_track"):
        lines +=[
            f"## N20 training label = {label}",
            "",
            "| Arm | Median MAE | Mean MAE | Late MAE | Mean paired delta | 95% cell-bootstrap CI | Better cells | Median attempts / accepted |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for x in g.sort_values("median_MAE_pp").itertuples():
            lines.append(
                f"| {x.arm} | {x.median_MAE_pp:.3f} | {x.mean_MAE_pp:.3f} | "
                f"{x.median_late_MAE_pp:.3f} | {x.mean_paired_delta_pp:+.3f} | "
                f"[{x.cell_bootstrap_lo:+.3f}, {x.cell_bootstrap_hi:+.3f}] | "
                f"{x.improved_cells}/{x.cells} | {x.median_attempts:.1f} / {x.median_accepted:.1f} |"
            )
        lines +=[""]
    lines += [
        "## Strict B4-matched WHEN comparisons (same physics HOW)",
        "",
        "Only cells with four *accepted* updates for both periodic and the trigger",
        "are compared; active minus periodic SOC_proxy MAE and cell-bootstrap CI.",
        "",
        "| Training labels | Trigger | Matched cells | Mean delta to periodic (pp) | 95% paired CI (pp) | Better cells |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for track, g in metric.groupby("train_label_track"):
        g=g[g.accepted_updates.eq(BUDGET)].copy()
        base=g[g.arm.eq("periodic_physics")].set_index("cell_id")
        for policy in [
            "voltage_physics","ph_voltage_physics",
            "cusum_voltage_physics","disagreement_physics",
            "joint_physics","random_physics",
        ]:
            alt=g[g.arm.eq(policy)].set_index("cell_id")
            common=base.index.intersection(alt.index)
            if len(common)<5: continue
            delta=(
                alt.loc[common,"proxy_MAE_pp"] -
                base.loc[common,"proxy_MAE_pp"]
            ).to_numpy()
            rng=np.random.default_rng(SEED)
            draws=rng.choice(delta,size=(5000,len(delta)),replace=True).mean(axis=1)
            lo,hi=np.quantile(draws,[.025,.975])
            lines.append(
                f"| {track} | {policy} | {len(delta)} | {delta.mean():+.3f} | "
                f"[{lo:+.3f},{hi:+.3f}] | {(delta<0).sum()}/{len(delta)} |"
            )
    lines += [
        "## Interpretation guardrails",
        "- This is an exploratory fixed-method sweep, not held-out model selection: do not crown the smallest test MAE as an independently validated novel method.",
        "- Physics residual constraints rely on *nominal* capacity and may themselves be misspecified as the cell ages.",
        "- Offline labels are derived using historical RPT capacity, so a numerical SOC gain here is not independent real-SOC accuracy.",
        "- This is phase-end, not sample-by-sample continuous SOC; no field agricultural drones or pack-level tests.",
        "- Triggered methods have different counts of accepted updates. A fair When-to-Update study requires matched counts on a validation-frozen How-to-Update policy.",
        "- The adaptation algorithm NEVER uses the SOC_proxy target to select an update or to decide a gate.",
        "- If no method improves frozen under physical-cell bootstrap, report a genuine negative result and diagnose observability/identifiability, not a fake success.",
    ]
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    label_file=RESULTS/"soc_proxy_phase_labels.csv"
    data=pd.read_csv(label_file)
    required=set(FEATURES+V_FEATURES+[
        "cell_id","charge_cycle_index","phase_Ns","QDischarge_end_Ah",
        "SOC_proxy","SOC_nominal_3Ah",
    ])
    missing=required-set(data.columns)
    if missing: raise ValueError(f"Missing fields: {missing}")
    outputs=[]
    for track in ["SOC_proxy","SOC_nominal_3Ah"]:
        for cell_id,cell in data.groupby("cell_id",sort=True):
            for arm in ARMS:
                row,_=replay_cell(cell,track,arm)
                outputs.append(row)
    metrics=pd.DataFrame(outputs)
    assert metrics.cell_id.nunique()==22
    assert set(metrics.groupby(["cell_id","train_label_track"]).size().unique())=={len(ARMS)}
    paired,summary=summarize(metrics)
    paired.to_csv(out/"mission_only_cell_metrics.csv",index=False)
    summary.to_csv(out/"mission_only_summary.csv",index=False)
    content=_report(summary,paired)
    (out/"report.md").write_text(content,encoding="utf-8")
    (out/"metadata.json").write_text(json.dumps({
        "dataset":"CMU VTC6 phase-end mission only",
        "source_cycle_count":N20,
        "selected_arms":ARMS,
        "no_online_RPT":True,
        "online_allowed_fields":FEATURES+["QDischarge_end_Ah","Vmin_V"],
        "online_forbidden":["SOC_proxy","SOC_nominal_3Ah","capacity_used_Ah","capacity_anchor_index","RPT data","future cycle","test targets"],
        "frozen_thresholds_from_N20":True,
        "prediction_before_mission_update":True,
        "all_target_RPT_info_scoring_only":True,
        "fixed_max_attempts":BUDGET,
        "periodic_mission_stride":150,
        "update_cooldown_missions":UPDATE_COOLDOWN,
        "physical_cell_count":22,
        "research_only":True,
    },ensure_ascii=False,indent=2),encoding="utf-8")
    print(content)


if __name__=="__main__":
    main()
