"""Reference queue: uses paid labels only after delivery."""
import numpy as np
import pandas as pd
from scenario_safe_when_selector import ACTIVE
from stateful_marginal_roi_maintenance import FIELDS

def queue_replay(cell, models, arm, budget, price, delay, gate=float("inf")):
    if delay < 0 or budget < 0: raise ValueError("invalid delay/budget")
    config = ACTIVE.get(arm)
    if config is None and arm != "frozen" and not (arm.startswith("periodic") and arm[8:].isdigit()):
        raise ValueError("invalid arm")
    pending, bought, accepted, rejected = {}, [], [], []
    last_accepted = last_requested = -1
    ops = set(cell.opportunities)
    losses = np.empty(cell.n, dtype=float)
    for t in range(cell.n):
        if t in pending:
            e = pending.pop(t)
            a, b = cell.bank[e][0]
            x, y = cell.deployed_params(last_accepted)
            pred = cell.test[e].predicted
            jump = float(100*np.mean(np.abs(np.clip(a*pred+b,0,1)-np.clip(x*pred+y,0,1))))
            if jump <= gate:
                last_accepted = e
                accepted.append(e)
            else: rejected.append(e)
        losses[t] = cell.frozen_loss[t] if last_accepted < 0 else cell.loss[last_accepted][t]
        if t not in ops or len(bought)>=budget or pending: continue
        buy = False
        if arm == "frozen": pass
        elif config is None: buy = ((t+1)%int(arm[8:])==0)
        else:
            mode,gap,threshold = config
            if last_requested<0 or t-last_requested>=gap:
                f = cell.feature(t,last_accepted,len(accepted))
                frame = pd.DataFrame([f])[list(FIELDS)]
                roi = float(models[0].predict(frame)[0])
                if mode == "old_roi": buy = roi>.15+5*price
                else:
                    regret = float(models[1].predict(frame)[0])
                    reserve = .1*len(bought)/max(1,budget) if mode=="reserve" else 0.
                    buy = regret>threshold+reserve and (mode=="delay_only" or roi>.15+5*price)
        if buy:
            bought.append(t)
            last_requested = t
            pending[t+delay+1] = t
    mae = float(losses.mean())
    return dict(MAE_pp=mae,J=mae+price*len(bought),purchases=len(bought),
        accepted=len(accepted),rejected=len(rejected),pending_end=len(pending),
        label_points=sum(cell.bank[e][1] for e in bought),
        events=";".join(map(str,bought)),
        accepted_events=";".join(map(str,accepted)))
