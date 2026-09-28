"""Phase 3 step 4-5: reactive vs trend vs forecaster risk signals, lead time vs false alarms.

Per outer fold (the same subject-grouped folds as the classifier, cal=60 s) we train on
the other subjects: the GRU classifier, the forecaster, and a "step-GRU" (drowsiness
classifier on 9 x 5 s aggregate blocks, the only classifier that can read forecasts).
The held-out subjects' ramps and real continuous alert videos are then streamed
through every signal at 5 s decisions:
  reactive     GRU p_now on the last 45 s (Platt-calibrated)
  trend        line through the last 60 s of reactive p_now, extrapolated 60 s ahead
  fc30 / fc60  forecaster -> step-GRU on the blocks 30 s / 60 s ahead
  step_now     step-GRU on the last 45 s of blocks (control: same model as fc*, no forecast)
  fused_mean / fused_max of (reactive, trend, fc30)
Alarm = risk >= threshold on 2 consecutive decisions, 60 s refractory (same for all).
Thresholds come from OTHER folds' alert videos at a false-alarm/hour budget; false
alarms are counted on the real alert videos, lead time on the ramps (onset = ramp
midpoint; hard splice = negative control).

Usage (GPU env): python -m src.eval.lead_time
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression

from src.eval.ramps import CALIB_S, FPS, LEAD_IN_S, load_ramps
from src.features.windows import NORM_COLS, iter_calibrated_videos, make_windows
from src.models import forecaster as fc
from src.models.classifier_gru import DrowsinessGRU
from src.models.training import DEVICE, RESULTS_DIR, cv_folds, fit_torch, load_dataset, predict_torch

BLOCK = 5  # s
STEP_LEN = 9  # blocks per step-GRU window = 45 s
T0 = 120  # s: first decision every signal can be evaluated at (trend needs 60 s of p_now)
SIGNALS = ("reactive", "trend", "fc30", "fc60", "step_now", "fused_mean", "fused_max", "fused_rf")
BUDGETS = (1.0, 3.0, 6.0)  # false alarms per hour
THRESHOLDS = np.linspace(0.02, 0.995, 98)
REFRACTORY_S = 60
TAB_COLS = None  # set from the forecast dataset


class StepGRU(nn.Module):
    def __init__(self, n_feat: int = 11, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h.squeeze(0)).squeeze(-1)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def platt(p, y):
    lr = LogisticRegression(C=1e4).fit(logit(p)[:, None], y)
    return lambda q: lr.predict_proba(logit(q)[:, None])[:, 1]


def step_windows(X, video_id, y_row, subjects):
    """9-block windows inside each video; label = the video's label."""
    xs, ys, ss = [], [], []
    for v in np.unique(video_id):
        idx = np.flatnonzero(video_id == v)
        w = sliding_window_view(X[idx], STEP_LEN, axis=0).transpose(0, 2, 1)
        xs.append(w)
        ys.append(np.full(len(w), y_row[idx[0]]))
        ss.append(np.full(len(w), subjects[idx[0]]))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ss)


@torch.no_grad()
def run_model(model, x):
    model.eval()
    return model(torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(DEVICE)).cpu().numpy()


def raw_signals(frames, ear_thr, gru, gru_scale, fmodel, step, mu, sd):
    """Streams one sequence. Returns decision times (s) and raw (uncalibrated) probabilities."""
    df = pd.DataFrame(frames, columns=list(NORM_COLS))
    df["t"] = np.arange(len(df)) / FPS
    summary, _ = make_windows(df, BLOCK, BLOCK, FPS, ear_thr)
    S = ((summary[fc_cols()].to_numpy(np.float32) - mu) / sd).astype(np.float32)
    nb = len(S)
    js = np.arange(12, nb + 1)  # decision after block j-1, at t = 5 j
    # reactive: GRU on the 450 frames ending at t
    W = sliding_window_view(frames[: nb * 50], 450, axis=0)[:, :, :]  # (n, 5, 450)
    starts = 50 * (js - STEP_LEN)
    Xg = W[starts].transpose(0, 2, 1)
    p_now = predict_torch(gru, gru_scale, np.ascontiguousarray(Xg))
    # step-GRU on history, and on history + forecast
    H = sliding_window_view(S, 12, axis=0).transpose(0, 2, 1)  # (nb-11, 12, F)
    hist9 = H[:, -STEP_LEN:]
    p_step = step(hist9)
    F = run_model(fmodel, H)  # (n, 12, F)
    full = np.concatenate([H, F], axis=1)  # (n, 24, F)
    out = {"t": 5.0 * js, "p_now": p_now, "step_now": p_step}
    for name, h in (("fc30", 6), ("fc60", 12)):
        out[name] = step(full[:, 12 + h - STEP_LEN : 12 + h])
    return out


_COLS = []


def fc_cols():
    return _COLS


def ewma(p, alpha=0.3):
    out = np.empty_like(p)
    out[0] = p[0]
    for i in range(1, len(p)):
        out[i] = alpha * p[i] + (1 - alpha) * out[i - 1]
    return out


def trend_signal(p, phi=0.9, horizon=12, alpha=0.3):
    """Damped (Holt-style) trend on a smoothed reactive series: p_now is a noisy
    per-5s classifier output, so an OLS slope over just 12 raw points is dominated
    by that noise and, even damped, saturates the risk near 1 far too often
    (confirmed empirically: >5/h false alarms at every threshold). EWMA-smooth first
    (root cause: less noisy input, not just a smaller extrapolation multiplier), then
    take the OLS slope of the smoothed series and extrapolate with geometric damping."""
    ps = ewma(p, alpha)
    x = np.arange(12) - 5.5
    W = sliding_window_view(ps, 12)
    slope = W @ x / (x @ x)
    last_fitted = W.mean(1) + slope * 5.5  # OLS value at the most recent point (position 11)
    damped = phi * (1 - phi**horizon) / (1 - phi)  # sum_{h=1..horizon} phi^h
    out = np.full(len(p), np.nan)
    out[11:] = np.clip(last_fitted + slope * damped, 0, 1)
    return out


def episodes(risk, times, thr):
    above = risk >= thr
    cand = np.flatnonzero(above[1:] & above[:-1]) + 1
    out, last = [], -1e9
    for i in cand:
        if times[i] - last >= REFRACTORY_S:
            out.append(times[i])
            last = times[i]
    return out


def main():
    global _COLS
    d = load_dataset(CALIB_S)
    Xs, y, subj = d["X_seq"], d["y"], d["subjects"]
    folds = cv_folds(y, subj)
    fold_of = {s: i for i, (_, te) in enumerate(folds) for s in np.unique(subj[te])}
    fd = np.load(f"features/forecast_cal{CALIB_S}.npz")
    FX, fvid, fsubj, fy = fd["X"], fd["video_id"], fd["subjects"], fd["y"]
    _COLS = [str(c) for c in fd["cols"]]
    xs, ys_, first = fc.make_samples(FX, fvid)
    samp_fold = np.array([fold_of[s] for s in fsubj[first]])
    SW, SY, SS = step_windows(FX, fvid, fy, fsubj)
    sw_fold = np.array([fold_of[s] for s in SS])

    videos = {}
    for subject, label, df, thr in iter_calibrated_videos(CALIB_S, FPS):
        if label == "non_drowsy":
            videos[subject] = (df[list(NORM_COLS)].to_numpy(np.float32), thr)
    ramps = list(load_ramps())

    seqs = []  # dicts with kind, subject, fold, dur, shape, onset, raw signals
    oof_gru = np.zeros(len(y))
    oof_step = np.zeros(len(SY))
    for f, (tr, te) in enumerate(folds):
        tr_sub = {s for s, i in fold_of.items() if i != f}
        gru, gscale = fit_torch(DrowsinessGRU, Xs[tr], y[tr], seed=0)
        oof_gru[te] = predict_torch(gru, gscale, Xs[te])
        rows = np.isin(fsubj, list(tr_sub))
        mu, sd = FX[rows].mean(0), FX[rows].std(0) + 1e-6
        nrm = lambda a: ((a - mu) / sd).astype(np.float32)
        ftr = samp_fold != f
        fmodel = fc.fit(nrm(xs[ftr]), nrm(ys_[ftr]), 0)
        fmodel.eval()
        str_ = sw_fold != f
        sgru, sscale = fit_torch(StepGRU, nrm(SW[str_]), SY[str_], seed=0)
        step = lambda x: 1 / (1 + np.exp(-run_model(sgru, x / sscale)))  # blocks -> P(drowsy)
        oof_step[~str_] = step(nrm(SW[~str_]))
        print(f"fold {f}: models trained", flush=True)

        def stream(kind, subject, frames, thr, **meta):
            r = raw_signals(frames, thr, gru, gscale, fmodel, step, mu, sd)
            seqs.append(dict(kind=kind, subject=subject, fold=f, **meta, **r))

        for s, (fr, thr) in videos.items():
            if fold_of.get(s) == f:
                stream("alert", s, fr, thr)
        for subject, dur, shape, rep, onset, thr, fr in ramps:
            if fold_of.get(subject) == f:
                stream("ramp", subject, fr, thr, dur=dur, shape=shape, onset=onset)
        print(f"fold {f}: streamed ({len(seqs)} sequences so far)", flush=True)

    # ---- pass 2: per-fold Platt from the other folds' OOF, derived + fused signals
    fold_gru = np.array([fold_of[s] for s in subj])
    for f in range(len(folds)):
        pg = platt(oof_gru[fold_gru != f], y[fold_gru != f])
        ps = platt(oof_step[sw_fold != f], SY[sw_fold != f])
        for q in (s for s in seqs if s["fold"] == f):
            react = pg(q["p_now"])
            fc30, fc60, snow = ps(q["fc30"]), ps(q["fc60"]), ps(q["step_now"])
            keep = q["t"] >= T0
            trend = trend_signal(react)
            sig = {"reactive": react, "trend": trend, "fc30": fc30, "fc60": fc60, "step_now": snow,
                   "fused_mean": (react + trend + fc30) / 3, "fused_max": np.maximum.reduce([react, trend, fc30]),
                   "fused_rf": (react + fc30) / 2}  # recommended: trend is too unstable to deploy (see PROGRESS.md)
            q["risk"] = {k: v[keep] for k, v in sig.items()}
            q["times"] = q["t"][keep]
            assert not np.isnan(q["risk"]["trend"]).any()

    # ---- pass 3: per (sequence, signal, threshold) alarm outcomes
    n_thr = len(THRESHOLDS)
    for q in seqs:
        q["out"] = {}
        for k in SIGNALS:
            if q["kind"] == "alert":
                q["out"][k] = np.array([len(episodes(q["risk"][k], q["times"], th)) for th in THRESHOLDS], float)
            else:  # lead time (s) of the first alarm at/after ramp start; nan = miss
                lead = np.full(n_thr, np.nan)
                for i, th in enumerate(THRESHOLDS):
                    hit = [t for t in episodes(q["risk"][k], q["times"], th) if t >= LEAD_IN_S]
                    if hit:
                        lead[i] = q["onset"] - hit[0]
                q["out"][k] = lead
    alert = [q for q in seqs if q["kind"] == "alert"]
    ramp = [q for q in seqs if q["kind"] == "ramp"]
    hours = {id(q): (q["times"][-1] - q["times"][0]) / 3600 for q in alert}
    print(f"{len(alert)} alert videos ({sum(hours.values()):.1f} h), {len(ramp)} ramps", flush=True)

    def fa_per_hour(qs, k):
        return np.sum([q["out"][k] for q in qs], 0) / sum(hours[id(q)] for q in qs)

    # curve: pooled over all held-out (out-of-fold) sequences
    cur = []
    for k in SIGNALS:
        fa = fa_per_hour(alert, k)
        for dur in (0, 120, 300, 600):
            L = np.array([q["out"][k] for q in ramp if q["dur"] == dur])
            for i, th in enumerate(THRESHOLDS):
                cur.append(dict(signal=k, dur=dur, thr=th, fa_per_h=fa[i], hit_rate=np.mean(~np.isnan(L[:, i])),
                                median_lead=np.nanmedian(L[:, i]) if (~np.isnan(L[:, i])).any() else np.nan))
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(cur).to_csv(RESULTS_DIR / "phase3_curve.csv", index=False)

    # operating points: threshold from the other folds' alert videos at the FA budget
    rows = []
    for k in SIGNALS:
        for budget in BUDGETS:
            leads = {dur: [] for dur in (0, 120, 300, 600)}
            fa_n, fa_h = 0.0, 0.0
            for f in range(len(folds)):
                other = [q for q in alert if q["fold"] != f]
                ok = np.flatnonzero(fa_per_hour(other, k) <= budget)
                i = ok[0] if len(ok) else n_thr - 1  # lowest threshold within budget
                for q in alert:
                    if q["fold"] == f:
                        fa_n += q["out"][k][i]
                        fa_h += hours[id(q)]
                for q in ramp:
                    if q["fold"] == f:
                        leads[q["dur"]].append(q["out"][k][i])
            r = dict(signal=k, budget=budget, fa_per_h=fa_n / fa_h)
            for dur, L in leads.items():
                L = np.array(L)
                r[f"hit_{dur}"] = np.mean(~np.isnan(L))
                r[f"lead_{dur}"] = np.nanmedian(L) if (~np.isnan(L)).any() else np.nan
                r[f"early_{dur}"] = np.mean(np.nan_to_num(L, nan=-1) > 0)
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "phase3_leadtime.csv", index=False)
    pd.set_option("display.width", 250)
    print(res.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
