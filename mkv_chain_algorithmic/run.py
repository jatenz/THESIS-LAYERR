import argparse
import os
import numpy as np
import pandas as pd

from src.ingest import load_csv, detect_mode, pick_first_existing, drop_totals_like_rows
from src.preprocess import (
    standardize_columns,
    build_time_series_from_time_series_export,
    build_time_series_from_per_video_export,
    add_features,
    ensure_metrics,
)
from src.states import classify_states_4, add_absorbing_churn
from src.markov import (
    transition_counts,
    transition_matrix_smoothed,
    enforce_absorbing,
    steady_state,
    k_step_forecast,
    expected_time_to_absorption,
    frames,
)
from src.hmm_viterbi import build_simple_hmm, viterbi
from src.evaluate import f1_macro, accuracy, thresholded_accuracy
from src.db import write_df
from src.report import make_summary

def parse_int_list(s):
    if s is None or str(s).strip() == "":
        return []
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [int(p) for p in parts]

def parse_float_list(s):
    if s is None or str(s).strip() == "":
        return []
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    return [float(p) for p in parts]

def next_state_labels(states):
    return states[1:]

def baseline_predict_next_as_current(states):
    return states[:-1]

def markov_predict_next(states, P, state_order):
    idx = {s: i for i, s in enumerate(state_order)}
    preds = []
    confs = []
    for s in states[:-1]:
        if s not in idx:
            preds.append(state_order[0])
            confs.append(0.0)
            continue
        row = np.asarray(P[idx[s]], dtype=float)
        j = int(np.argmax(row))
        preds.append(state_order[j])
        confs.append(float(row[j]))
    return preds, confs

def markov_predict_next_topk(states, P, state_order, k=2):
    idx = {s: i for i, s in enumerate(state_order)}
    out = []
    for s in states[:-1]:
        if s not in idx:
            out.append([])
            continue
        row = np.asarray(P[idx[s]], dtype=float)
        top = np.argsort(-row)[: int(k)]
        out.append([state_order[int(i)] for i in top])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--freq", default="W")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--db", default="viewer_loyalty.db")
    ap.add_argument("--date_col", default=None)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--enable_churn", action="store_true")
    ap.add_argument("--churn_k", type=int, default=3)

    ap.add_argument("--forecast_steps", default="1,2,4,8")

    ap.add_argument("--do_hmm", action="store_true")
    ap.add_argument("--do_leiden", action="store_true")
    ap.add_argument("--leiden_k", type=int, default=10)
    ap.add_argument("--leiden_resolution", type=float, default=1.0)

    ap.add_argument("--acc_thresholds", default="0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--topk", type=int, default=2)

    args = ap.parse_args()

    cwd = os.getcwd()
    os.makedirs(args.out, exist_ok=True)

    raw = load_csv(args.csv)
    raw = standardize_columns(raw)

    mode = detect_mode(raw)
    cols = set(raw.columns)

    if mode == "time_series":
        date_col = args.date_col or pick_first_existing(cols, ["date", "day", "week", "month"])
        if date_col is None:
            raise ValueError(f"No usable time column found. Columns: {list(raw.columns)}")
        g = build_time_series_from_time_series_export(raw, date_col, args.freq)
    elif mode == "per_video":
        raw2 = drop_totals_like_rows(raw)
        g = build_time_series_from_per_video_export(raw2, args.freq)
    else:
        if args.date_col and args.date_col in cols:
            g = build_time_series_from_time_series_export(raw, args.date_col, args.freq)
        else:
            raise ValueError(f"Unknown export format. Columns: {list(raw.columns)}")

    if len(g) == 0:
        raise ValueError("No rows after aggregation. Check your CSV dates and selected report.")

    g = ensure_metrics(g)
    g = add_features(g)
    g = classify_states_4(g)

    state_order = ["NEW", "ENGAGED", "LOYAL", "INACTIVE"]
    if args.enable_churn:
        g = add_absorbing_churn(g, inactive_streak_k=args.churn_k)
        if "CHURN" not in state_order:
            state_order = state_order + ["CHURN"]

    states = g["state_obs"].astype(str).tolist()

    C = transition_counts(states, state_order)
    P = transition_matrix_smoothed(C, alpha=args.alpha)
    if args.enable_churn and "CHURN" in state_order:
        P = enforce_absorbing(P, state_order, absorbing_state="CHURN")

    Cdf, Pdf = frames(C, P, state_order)
    ss = steady_state(P)
    ss_df = pd.DataFrame({"state": state_order, "steady_prob": [float(x) for x in ss]})

    k_steps = parse_int_list(args.forecast_steps) or [1, 2, 4, 8]
    start_state = states[-1]
    forecast_df = k_step_forecast(P, state_order, start_state, k_steps)

    t_churn_df = None
    if args.enable_churn and "CHURN" in state_order:
        t_churn_df, _ = expected_time_to_absorption(P, state_order, absorbing_state="CHURN")

    eval_rows = []
    cm_df = pd.DataFrame(np.zeros((len(state_order), len(state_order)), dtype=int), index=state_order, columns=state_order)

    if len(states) >= 2:
        y_true = next_state_labels(states)

        y_pred_base = baseline_predict_next_as_current(states)
        f1_base, cm_base = f1_macro(y_true, y_pred_base, state_order)
        acc_base = accuracy(y_true, y_pred_base)
        eval_rows.append({"metric": "baseline_next_state_f1_macro", "value": float(f1_base)})
        eval_rows.append({"metric": "baseline_next_state_accuracy", "value": float(acc_base)})

        y_pred_mkv, y_conf_mkv = markov_predict_next(states, P, state_order)
        f1_mkv, cm_mkv = f1_macro(y_true, y_pred_mkv, state_order)
        acc_mkv = accuracy(y_true, y_pred_mkv)
        cm_df = pd.DataFrame(cm_mkv, index=state_order, columns=state_order)

        eval_rows.append({"metric": "markov_next_state_f1_macro", "value": float(f1_mkv)})
        eval_rows.append({"metric": "markov_next_state_accuracy", "value": float(acc_mkv)})

        thresholds = parse_float_list(args.acc_thresholds)
        for th in thresholds:
            th_acc, th_cov, th_n = thresholded_accuracy(y_true, y_pred_mkv, y_conf_mkv, th)
            eval_rows.append({"metric": f"markov_accuracy_at_threshold_{th:.2f}", "value": float(th_acc)})
            eval_rows.append({"metric": f"markov_coverage_at_threshold_{th:.2f}", "value": float(th_cov)})
            eval_rows.append({"metric": f"markov_confident_count_at_threshold_{th:.2f}", "value": int(th_n)})

        topk = int(max(1, args.topk))
        topk_preds = markov_predict_next_topk(states, P, state_order, k=topk)
        topk_hits = 0
        for yt, plist in zip(y_true, topk_preds):
            if yt in plist:
                topk_hits += 1
        topk_acc = float(topk_hits / len(y_true)) if len(y_true) else 0.0
        eval_rows.append({"metric": f"markov_top{topk}_accuracy", "value": float(topk_acc)})

        g["markov_next_pred"] = [""] + y_pred_mkv
        g["markov_next_conf"] = [np.nan] + [float(x) for x in y_conf_mkv]
    else:
        eval_rows.append({"metric": "note", "value": "insufficient_sequence_length"})

    if args.do_hmm and len(states) >= 2:
        obs_idx = [state_order.index(s) for s in states]
        A, B, pi = build_simple_hmm(len(state_order))
        hidden_idx = viterbi(obs_idx, A, B, pi)
        g["state_hidden"] = [state_order[i] for i in hidden_idx]

    if args.do_leiden:
        from src.leiden_cluster import leiden_membership
        feat_cols = ["views_z", "watch_z", "return_z", "engage_z", "loyalty_score"]
        X = g[feat_cols].to_numpy(dtype=float)
        k = int(min(max(2, args.leiden_k), len(X)))
        clusters = leiden_membership(X, k=k, resolution=float(args.leiden_resolution))
        g["period_cluster"] = [int(c) for c in clusters]

    evaluation_df = pd.DataFrame(eval_rows)

    paths = {
        "time_series_states.csv": g,
        "transition_counts.csv": Cdf,
        "transition_probabilities.csv": Pdf,
        "steady_state.csv": ss_df,
        "forecast_kstep.csv": forecast_df,
        "evaluation.csv": evaluation_df,
        "confusion_matrix.csv": cm_df,
    }

    for name, df in paths.items():
        p = os.path.join(args.out, name)
        df.to_csv(p, index=(name in ["transition_counts.csv", "transition_probabilities.csv", "confusion_matrix.csv"]))

    if t_churn_df is not None:
        t_churn_df.to_csv(os.path.join(args.out, "expected_time_to_churn.csv"), index=False)

    summary_df = make_summary(state_order, ss, mode, len(g), args.alpha, args.churn_k if args.enable_churn else None, k_steps)
    summary_df.to_csv(os.path.join(args.out, "summary.csv"), index=False)

    db_path = os.path.join(cwd, args.db)
    write_df(db_path, "periods", g)
    write_df(db_path, "transition_counts", Cdf.reset_index().rename(columns={"index": "from_state"}))
    write_df(db_path, "transition_probabilities", Pdf.reset_index().rename(columns={"index": "from_state"}))
    write_df(db_path, "steady_state", ss_df)
    write_df(db_path, "forecast_kstep", forecast_df)
    write_df(db_path, "evaluation", evaluation_df)
    write_df(db_path, "confusion_matrix", cm_df.reset_index().rename(columns={"index": "true_state"}))
    write_df(db_path, "summary", summary_df)

    if t_churn_df is not None:
        write_df(db_path, "expected_time_to_churn", t_churn_df)

    print("Ran in:", cwd)
    print("Detected mode:", mode)
    print("Aggregated rows:", len(g))
    print("State order:", ",".join(state_order))
    print("Saved outputs folder:", os.path.join(cwd, args.out))
    print("Saved database:", db_path)
    print("Laplace alpha:", float(args.alpha))
    if args.enable_churn:
        print("Churn enabled. Inactive streak k:", int(args.churn_k))
    print("Forecast steps:", ",".join([str(int(k)) for k in k_steps]))
    print()
    print("Transition Probability Matrix")
    print(Pdf.to_string())
    print()
    print("Steady State Distribution")
    print(ss_df.to_string(index=False))
    print()
    if len(states) >= 2:
        for r in eval_rows:
            print(str(r["metric"]) + ":", r["value"])
        if args.enable_churn and t_churn_df is not None:
            print()
            print("Expected steps to churn")
            print(t_churn_df.to_string(index=False))

if __name__ == "__main__":
    main()