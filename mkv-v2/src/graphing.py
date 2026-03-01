
import os
import pandas as pd

def _safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return None

def generate_graphs(out_dir: str, state_order, include_churn: bool = True):
    """
    Generates PNGs into out_dir. Uses matplotlib if available.
    No seaborn. No custom colors. One plot per figure.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return {"ok": False, "reason": "matplotlib_not_installed"}

    os.makedirs(out_dir, exist_ok=True)

    # 1) Loyalty score over time
    ts_path = os.path.join(out_dir, "time_series_states.csv")
    df = _safe_read_csv(ts_path)
    if df is not None and "period" in df.columns and "loyalty_score" in df.columns:
        try:
            x = pd.to_datetime(df["period"], errors="coerce")
            y = pd.to_numeric(df["loyalty_score"], errors="coerce").fillna(0.0)
            plt.figure()
            plt.plot(x, y)
            plt.title("Loyalty Score Over Time")
            plt.xlabel("Period")
            plt.ylabel("Loyalty Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "graph_loyalty_over_time.png"))
            plt.close()
        except Exception:
            pass

    # 2) State proportions over time (Excel-style line chart)
    if df is not None and "period" in df.columns and "state_obs" in df.columns:
        try:
            dfx = df.copy()
            dfx["period"] = pd.to_datetime(dfx["period"], errors="coerce")
            dfx = dfx.dropna(subset=["period"]).sort_values("period")

            for s in state_order:
                dfx[f"p_{s}"] = (dfx["state_obs"].astype(str) == s).astype(int)

            win = min(6, max(2, len(dfx)//10)) if len(dfx) > 0 else 2
            cols = [f"p_{s}" for s in state_order]
            dfx[cols] = dfx[cols].rolling(window=win, min_periods=1).mean()

            plt.figure(figsize=(10, 6))
            plt.grid(True, linestyle='--', linewidth=0.5)

            colors = {
                "NEW": "blue",
                "ENGAGED": "red",
                "LOYAL": "green",
                "INACTIVE": "orange"
            }

            for s in state_order:
                plt.plot(
                    dfx["period"],
                    dfx[f"p_{s}"],
                    label=s,
                    linewidth=2,
                    color=colors.get(s, None)
                )

            plt.title("State Proportions Over Time")
            plt.xlabel("Period")
            plt.ylabel("Proportion")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "graph_state_proportions.png"))
            plt.close()

        except Exception:
            pass

    # 3) Transition heatmap
    P_path = os.path.join(out_dir, "transition_probabilities_train.csv")
    Pdf = _safe_read_csv(P_path, index_col=0)
    if Pdf is not None:
        try:
            plt.figure()
            plt.imshow(Pdf.values)
            plt.xticks(range(len(Pdf.columns)), Pdf.columns, rotation=45)
            plt.yticks(range(len(Pdf.index)), Pdf.index)
            plt.title("Transition Matrix Heatmap (Train)")
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "graph_transition_heatmap.png"))
            plt.close()
        except Exception:
            pass

    # 4) Steady-state bar chart (handles long or wide format; excludes CHURN by default)
    ss_path = os.path.join(out_dir, "steady_state.csv")
    ss = _safe_read_csv(ss_path)
    if ss is not None:
        try:
            labels, values = [], []
            if "state" in ss.columns and ("steady_prob" in ss.columns or ss.shape[1] >= 2):
                prob_col = "steady_prob" if "steady_prob" in ss.columns else ss.columns[-1]
                labels = ss["state"].astype(str).tolist()
                values = pd.to_numeric(ss[prob_col], errors="coerce").fillna(0.0).tolist()
            else:
                # wide
                cols = list(ss.columns)
                labels = cols
                values = pd.to_numeric(ss.iloc[0], errors="coerce").fillna(0.0).tolist()

            if not include_churn:
                filt = [(l, v) for l, v in zip(labels, values) if str(l) != "CHURN"]
                labels, values = [x[0] for x in filt], [x[1] for x in filt]

            if labels:
                plt.figure()
                plt.bar(labels, values)
                plt.title("Steady State Distribution")
                plt.xlabel("State")
                plt.ylabel("Probability")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "graph_steady_state.png"))
                plt.close()
        except Exception:
            pass

    # 5) Forecast LOYAL probability
    fc_path = os.path.join(out_dir, "forecast_kstep.csv")
    fc = _safe_read_csv(fc_path)
    if fc is not None and "k" in fc.columns:
        try:
            loyal_cols = [c for c in fc.columns if c.upper().endswith("LOYAL") or "prob_LOYAL" in c]
            if "prob_LOYAL" in fc.columns:
                ycol = "prob_LOYAL"
            else:
                ycol = "prob_LOYAL" if "prob_LOYAL" in fc.columns else None
                if ycol is None:
                    # common format: prob_LOYAL exists from run.py
                    cand = [c for c in fc.columns if c.lower() == "prob_loyal"]
                    ycol = cand[0] if cand else None
            if ycol:
                plt.figure()
                plt.plot(fc["k"], fc[ycol])
                plt.title("Forecast Probability of LOYAL State")
                plt.xlabel("Steps (k)")
                plt.ylabel("Probability")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "graph_forecast_loyal.png"))
                plt.close()
        except Exception:
            pass

    return {"ok": True}
