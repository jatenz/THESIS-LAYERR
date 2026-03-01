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
from src.states import (
    classify_states_4,
    add_absorbing_churn,
    BASE_STATES_4,
)
from src.markov import (
    transition_counts,
    transition_matrix_smoothed,
    enforce_absorbing,
    steady_state,
    k_step_forecast,
    expected_time_to_absorption,
    frames,
)
from src.hmm_viterbi import baum_welch, viterbi
from src.evaluate import f1_macro, accuracy, thresholded_accuracy, topk_accuracy
from src.db import write_df
from src.report import make_summary
from src.graphing import generate_graphs

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

def time_split(df: pd.DataFrame, ratio: float):
    r = float(ratio)
    r = min(max(r, 0.50), 0.95)
    n = len(df)
    cut = int(round(n * r))
    cut = min(max(cut, 1), n - 1)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True), cut

def fit_markov(states_train, state_order, alpha, enable_churn):
    return fit_markov_with_abs_option(states_train, state_order, alpha, enable_churn, enforce_abs=True)

def fit_markov_with_abs_option(states_train, state_order, alpha, enable_churn, enforce_abs: bool = True):
    C = transition_counts(states_train, state_order)
    P = transition_matrix_smoothed(C, alpha=float(alpha))
    if enforce_abs and enable_churn and "CHURN" in state_order:
        P = enforce_absorbing(P, state_order, absorbing_state="CHURN")
    return C, P

def eval_next_state(states_seq, P, state_order, acc_thresholds, topk):
    rows = []
    cm_df = pd.DataFrame(np.zeros((len(state_order), len(state_order)), dtype=int), index=state_order, columns=state_order)
    if len(states_seq) < 2:
        rows.append({"metric": "note", "value": "insufficient_sequence_length"})
        return pd.DataFrame(rows), cm_df, [], []

    y_true = next_state_labels(states_seq)

    y_pred_base = baseline_predict_next_as_current(states_seq)
    f1_base, _ = f1_macro(y_true, y_pred_base, state_order)
    acc_base = accuracy(y_true, y_pred_base)
    rows.append({"metric": "baseline_next_state_f1_macro", "value": float(f1_base)})
    rows.append({"metric": "baseline_next_state_accuracy", "value": float(acc_base)})

    y_pred_mkv, y_conf_mkv = markov_predict_next(states_seq, P, state_order)
    f1_mkv, cm_mkv = f1_macro(y_true, y_pred_mkv, state_order)
    acc_mkv = accuracy(y_true, y_pred_mkv)
    cm_df = pd.DataFrame(cm_mkv, index=state_order, columns=state_order)

    rows.append({"metric": "markov_next_state_f1_macro", "value": float(f1_mkv)})
    rows.append({"metric": "markov_next_state_accuracy", "value": float(acc_mkv)})

    thresholds = acc_thresholds
    for th in thresholds:
        th_acc, th_cov, th_n = thresholded_accuracy(y_true, y_pred_mkv, y_conf_mkv, th)
        rows.append({"metric": f"markov_accuracy_at_threshold_{th:.2f}", "value": float(th_acc)})
        rows.append({"metric": f"markov_coverage_at_threshold_{th:.2f}", "value": float(th_cov)})
        rows.append({"metric": f"markov_confident_count_at_threshold_{th:.2f}", "value": int(th_n)})

    topk_preds = markov_predict_next_topk(states_seq, P, state_order, k=topk)
    rows.append({"metric": f"markov_top{int(topk)}_accuracy", "value": float(topk_accuracy(y_true, topk_preds))})

    return pd.DataFrame(rows), cm_df, y_pred_mkv, y_conf_mkv

def matrix_drift(Pa, Pb):
    Pa = np.asarray(Pa, dtype=float)
    Pb = np.asarray(Pb, dtype=float)
    return float(np.linalg.norm(Pa - Pb, 1))

def mean_state_durations(states):
    if len(states) == 0:
        return pd.DataFrame(columns=["state", "mean_duration", "segments"])
    segs = []
    cur = states[0]
    run = 1
    for s in states[1:]:
        if s == cur:
            run += 1
        else:
            segs.append((cur, run))
            cur = s
            run = 1
    segs.append((cur, run))
    df = pd.DataFrame(segs, columns=["state", "duration"])
    out = df.groupby("state").agg(mean_duration=("duration", "mean"), segments=("duration", "count")).reset_index()
    out["mean_duration"] = out["mean_duration"].astype(float)
    out["segments"] = out["segments"].astype(int)
    return out.sort_values("state").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--freq", default="W")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--db", default="viewer_loyalty.db")
    ap.add_argument("--date_col", default=None)

    ap.add_argument("--new_periods", type=int, default=2)

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--enable_churn", action="store_true")
    ap.add_argument("--churn_k", type=int, default=3)

    ap.add_argument("--forecast_steps", default="1,2,4,8")
    ap.add_argument("--train_ratio", type=float, default=0.70)

    ap.add_argument("--do_hmm", action="store_true")
    ap.add_argument("--hmm_iter", type=int, default=25)
    ap.add_argument("--hmm_tol", type=float, default=1e-6)
    ap.add_argument("--hmm_seed", type=int, default=7)

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

    g = classify_states_4(g, new_periods=int(args.new_periods))
    state_order = list(BASE_STATES_4)
    
    if args.enable_churn:
        g = add_absorbing_churn(g, inactive_streak_k=args.churn_k)
        if "CHURN" not in state_order:
            state_order = state_order + ["CHURN"]

    states_all = g["state_obs"].astype(str).tolist()

    g_train, g_test, cut = time_split(g, args.train_ratio)
    states_train = g_train["state_obs"].astype(str).tolist()
    states_test = g_test["state_obs"].astype(str).tolist()

    C_train, P_train = fit_markov(states_train, state_order, args.alpha, args.enable_churn)
    Cdf_train, Pdf_train = frames(C_train, P_train, state_order)
    ss = steady_state(P_train)
    ss_df = pd.DataFrame({"state": state_order, "steady_prob": [float(x) for x in ss]})

    ss_nonabs_df = None
    if args.enable_churn and "CHURN" in state_order:
        _, P_nonabs = fit_markov_with_abs_option(states_train, state_order, args.alpha, args.enable_churn, enforce_abs=False)
        ss_nonabs = steady_state(P_nonabs)
        ss_nonabs_df = pd.DataFrame({"state": state_order, "steady_prob": [float(x) for x in ss_nonabs]})


    k_steps = parse_int_list(args.forecast_steps) or [1, 2, 4, 8]
    start_state = states_all[-1]
    forecast_df = k_step_forecast(P_train, state_order, start_state, k_steps)

    t_churn_df = None
    if args.enable_churn and "CHURN" in state_order:
        t_churn_df, _ = expected_time_to_absorption(P_train, state_order, absorbing_state="CHURN")

    thresholds = parse_float_list(args.acc_thresholds)
    topk = int(max(1, args.topk))

    eval_train_df, cm_train_df, train_pred, train_conf = eval_next_state(states_train, P_train, state_order, thresholds, topk)
    eval_test_df, cm_test_df, test_pred, test_conf = eval_next_state(states_test, P_train, state_order, thresholds, topk)

    drift_score = 0.0
    if len(states_train) >= 4:
        mid = max(2, len(states_train) // 2)
        C_a, P_a = fit_markov(states_train[:mid], state_order, args.alpha, args.enable_churn)
        C_b, P_b = fit_markov(states_train[mid:], state_order, args.alpha, args.enable_churn)
        drift_score = matrix_drift(P_a, P_b)

    duration_df = mean_state_durations(states_all)
    drift_df = pd.DataFrame([{"train_window_drift_l1": float(drift_score), "train_rows": int(len(g_train)), "test_rows": int(len(g_test)), "train_cut_index": int(cut)}])

    if len(train_pred) == len(states_train) - 1:
        g_train["markov_next_pred"] = [""] + train_pred
        g_train["markov_next_conf"] = [np.nan] + [float(x) for x in train_conf]
    if len(test_pred) == len(states_test) - 1:
        g_test["markov_next_pred"] = [""] + test_pred
        g_test["markov_next_conf"] = [np.nan] + [float(x) for x in test_conf]

    if args.do_hmm and len(states_train) >= 2:
        obs_idx = [state_order.index(s) for s in states_train if s in state_order]
        if len(obs_idx) >= 2:
            A, B, pi, ll = baum_welch(obs_idx, n_states=len(state_order), n_iter=int(args.hmm_iter), tol=float(args.hmm_tol), seed=int(args.hmm_seed))
            hidden_idx = viterbi(obs_idx, A, B, pi)
            g_train = g_train.iloc[:len(hidden_idx)].copy()
            g_train["state_hidden"] = [state_order[i] for i in hidden_idx]
            hmm_fit_df = pd.DataFrame([{"hmm_loglik": float(ll), "hmm_iter": int(args.hmm_iter), "hmm_tol": float(args.hmm_tol), "hmm_seed": int(args.hmm_seed)}])
            hmm_A_df = pd.DataFrame(A, index=state_order, columns=state_order)
            hmm_B_df = pd.DataFrame(B, index=state_order, columns=state_order)
        else:
            hmm_fit_df = pd.DataFrame([{"hmm_loglik": 0.0, "hmm_iter": int(args.hmm_iter), "hmm_tol": float(args.hmm_tol), "hmm_seed": int(args.hmm_seed)}])
            hmm_A_df = pd.DataFrame(np.zeros((len(state_order), len(state_order))), index=state_order, columns=state_order)
            hmm_B_df = pd.DataFrame(np.zeros((len(state_order), len(state_order))), index=state_order, columns=state_order)
    else:
        hmm_fit_df = pd.DataFrame([{"hmm_loglik": "", "hmm_iter": "", "hmm_tol": "", "hmm_seed": ""}])
        hmm_A_df = pd.DataFrame(np.zeros((len(state_order), len(state_order))), index=state_order, columns=state_order)
        hmm_B_df = pd.DataFrame(np.zeros((len(state_order), len(state_order))), index=state_order, columns=state_order)

    if args.do_leiden:
        from src.leiden_cluster import leiden_membership
        feat_cols = ["views_z", "watch_z", "return_z", "engage_z", "loyalty_score"]
        X = g[feat_cols].to_numpy(dtype=float)
        k = int(min(max(2, args.leiden_k), len(X)))
        clusters = leiden_membership(X, k=k, resolution=float(args.leiden_resolution))
        g["period_cluster"] = [int(c) for c in clusters]

    summary_df = make_summary(state_order, ss, mode, len(g), args.alpha, args.churn_k if args.enable_churn else None, k_steps)
    summary_df["train_ratio"] = float(args.train_ratio)
    summary_df["train_rows"] = int(len(g_train))
    summary_df["test_rows"] = int(len(g_test))
    summary_df["train_window_drift_l1"] = float(drift_score)

    paths = {
        "time_series_states.csv": g,
        "train_periods.csv": g_train,
        "test_periods.csv": g_test,
        "transition_counts_train.csv": Cdf_train,
        "transition_probabilities_train.csv": Pdf_train,
        "steady_state.csv": ss_df,
        "steady_state_nonabsorbing.csv": ss_nonabs_df if ss_nonabs_df is not None else ss_df,

        "forecast_kstep.csv": forecast_df,
        "evaluation_train.csv": eval_train_df,
        "evaluation_test.csv": eval_test_df,
        "confusion_matrix_train.csv": cm_train_df,
        "confusion_matrix_test.csv": cm_test_df,
        "state_durations.csv": duration_df,
        "transition_drift.csv": drift_df,
        "hmm_fit.csv": hmm_fit_df,
        "hmm_A.csv": hmm_A_df,
        "hmm_B.csv": hmm_B_df,
        "summary.csv": summary_df,
    }

    for name, df in paths.items():
        p = os.path.join(args.out, name)
        index_flag = name in [
            "transition_counts_train.csv",
            "transition_probabilities_train.csv",
            "confusion_matrix_train.csv",
            "confusion_matrix_test.csv",
            "hmm_A.csv",
            "hmm_B.csv",
        ]
        df.to_csv(p, index=index_flag)

    if t_churn_df is not None:
        t_churn_df.to_csv(os.path.join(args.out, "expected_time_to_churn.csv"), index=False)

    db_path = os.path.join(cwd, args.db)
    write_df(db_path, "periods", g)
    write_df(db_path, "train_periods", g_train)
    write_df(db_path, "test_periods", g_test)
    write_df(db_path, "transition_counts_train", Cdf_train.reset_index().rename(columns={"index": "from_state"}))
    write_df(db_path, "transition_probabilities_train", Pdf_train.reset_index().rename(columns={"index": "from_state"}))
    write_df(db_path, "steady_state", ss_df)
    write_df(db_path, "forecast_kstep", forecast_df)
    write_df(db_path, "evaluation_train", eval_train_df)
    write_df(db_path, "evaluation_test", eval_test_df)
    write_df(db_path, "confusion_matrix_train", cm_train_df.reset_index().rename(columns={"index": "true_state"}))
    write_df(db_path, "confusion_matrix_test", cm_test_df.reset_index().rename(columns={"index": "true_state"}))
    write_df(db_path, "state_durations", duration_df)
    write_df(db_path, "transition_drift", drift_df)
    write_df(db_path, "hmm_fit", hmm_fit_df)
    write_df(db_path, "hmm_A", hmm_A_df.reset_index().rename(columns={"index": "from_state"}))
    write_df(db_path, "hmm_B", hmm_B_df.reset_index().rename(columns={"index": "hidden_state"}))
    write_df(db_path, "summary", summary_df)
    if t_churn_df is not None:
        write_df(db_path, "expected_time_to_churn", t_churn_df)

    # Auto-generate graphs
    graph_result = generate_graphs(args.out, state_order, include_churn=False if args.enable_churn else True)
    if not graph_result.get("ok", False):
        print("Graphs not generated:", graph_result.get("reason"))

    print("Ran in:", cwd)
    print("Detected mode:", mode)
    print("Aggregated rows:", len(g))
    print("Train rows:", len(g_train))
    print("Test rows:", len(g_test))
    print("State order:", ",".join(state_order))
    print("Laplace alpha:", float(args.alpha))
    if args.enable_churn:
        print("Churn enabled. Inactive streak k:", int(args.churn_k))
    print("Forecast steps:", ",".join([str(int(k)) for k in k_steps]))
    print("Train-window drift L1:", float(drift_score))

if __name__ == "__main__":
    main()
