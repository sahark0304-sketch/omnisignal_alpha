"""
dashboard/app.py — OmniSignal Alpha v2.0
Pillar 10: Institutional-Grade Dashboard

New vs v1.1:
  - Equity curve chart (Altair/Pandas)
  - Rolling drawdown chart
  - Source Leaderboard with tier badges (Alpha Ranker)
  - Portfolio heatmap (currency exposure Pillar 11)
  - Latency monitor panel (Pillar 13)
  - Black Box decision log with full reasoning (Pillar 12)
  - AI correction history (Pillar 14)
  - Backtest quick-runner
"""

import streamlit as st
import pandas as pd
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import db_manager
from mt5_executor import mt5_executor
from risk_guard import risk_guard
import config

st.set_page_config(
    page_title="OmniSignal Alpha v2.0",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --primary: #00ff88; --warn: #ffaa00; --danger: #ff4444; --bg: #0e0e1a; }
.main-header { font-size:2rem; font-weight:700; color:var(--primary); margin-bottom:0; }
.version-tag { font-size:0.75rem; color:#888; }
div[data-testid="metric-container"] { background:#1a1a2e; border-radius:10px; padding:10px; border:1px solid #2a2a3e; }
.tier-s { background:#ffd700; color:#000; padding:2px 8px; border-radius:4px; font-weight:bold; }
.tier-a { background:#00ff88; color:#000; padding:2px 8px; border-radius:4px; font-weight:bold; }
.tier-b { background:#4488ff; color:#fff; padding:2px 8px; border-radius:4px; font-weight:bold; }
.tier-c { background:#ff8800; color:#fff; padding:2px 8px; border-radius:4px; font-weight:bold; }
.tier-f { background:#ff4444; color:#fff; padding:2px 8px; border-radius:4px; font-weight:bold; }
.lat-normal { color:#00ff88; }
.lat-warn { color:#ffaa00; }
.lat-safety { color:#ff8800; }
.lat-critical { color:#ff4444; }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ OmniSignal Alpha v2.0")
    mode_color = "#00ff88" if config.OPERATING_MODE == "LIVE" else "#ffaa00"
    st.markdown(f'<span style="color:{mode_color}; font-weight:bold;">● {config.OPERATING_MODE}</span>',
                unsafe_allow_html=True)
    st.divider()

    # Latency widget
    st.markdown("### 📡 Network Status")
    try:
        from quant.latency_monitor import get_state as lat_state
        ls = lat_state()
        mode_css = {
            "NORMAL": "lat-normal", "WARNING": "lat-warn",
            "SAFETY": "lat-safety", "CRITICAL": "lat-critical",
            "UNREACHABLE": "lat-critical", "INIT": "lat-warn",
        }.get(ls.mode, "lat-warn")
        st.markdown(
            f'<span class="{mode_css}">● {ls.mode}</span> '
            f'`{ls.avg_ms:.0f}ms avg` | `±{ls.jitter_ms:.0f}ms jitter`',
            unsafe_allow_html=True,
        )
    except Exception:
        st.caption("Latency monitor not running")

    st.divider()
    st.markdown("### 🚨 Emergency Controls")

    if st.button("🛑 PANIC — CLOSE ALL", type="primary", use_container_width=True):
        with st.spinner("Closing all positions..."):
            try:
                mt5_executor.ensure_connected()
                closed = mt5_executor.emergency_close_all()
                risk_guard.halt_trading("Manual panic from dashboard")
                st.error(f"✅ {closed} positions closed. Trading HALTED.")
            except Exception as e:
                st.error(f"Error: {e}")

    halted, halt_reason = risk_guard.is_halted()
    if halted:
        st.error(f"🛑 HALTED\n{halt_reason}")
        if st.button("▶️ Resume Trading", use_container_width=True):
            risk_guard.resume_trading()
            st.success("Resumed.")
            st.rerun()
    else:
        st.success("✅ System Active")

    st.divider()
    refresh = st.slider("Refresh (s)", 5, 60, 15)
    page    = st.radio("Page", ["📊 Overview", "📈 Performance KPIs", "🏆 Alpha Ranker",
                                 "🔍 Black Box", "🌡 Exposure", "⚙️ Self-Correction"])


# ── HEADER ────────────────────────────────────────────────────────────────────

st.markdown('<div class="main-header">⚡ OmniSignal Alpha</div>', unsafe_allow_html=True)
st.markdown(f'<div class="version-tag">v2.0 | {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</div>',
            unsafe_allow_html=True)
st.divider()


# ── LIVE METRICS ──────────────────────────────────────────────────────────────

try:
    mt5_executor.ensure_connected()
    equity    = mt5_executor.get_account_equity()
    balance   = mt5_executor.get_account_balance()
    daily_pnl = db_manager.get_daily_pnl()
    positions = mt5_executor.get_all_positions()
    dd_limit  = balance * (config.DAILY_DRAWDOWN_LIMIT_PCT / 100)
    dd_pct    = abs(daily_pnl) / dd_limit * 100 if dd_limit > 0 and daily_pnl < 0 else 0.0
    mt5_ok    = True
except Exception as e:
    equity = balance = daily_pnl = 0
    positions = []
    dd_pct = 0
    mt5_ok = False
    st.warning(f"⚠️ MT5 issue: {e}")

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("💰 Balance",    f"${balance:,.2f}")
c2.metric("📊 Equity",     f"${equity:,.2f}")
c3.metric("📈 Daily PnL",  f"${daily_pnl:+,.2f}")
c4.metric("🔢 Open",       len(positions))
c5.metric("⚠️ DD Used",   f"{dd_pct:.1f}%")
dd_emoji = "🟢" if dd_pct < 50 else ("🟡" if dd_pct < 80 else "🔴")
c6.metric(f"{dd_emoji} DD Limit", f"{config.DAILY_DRAWDOWN_LIMIT_PCT}%")

st.progress(min(dd_pct / 100, 1.0),
            text=f"Daily Drawdown: {dd_pct:.1f}% / {config.DAILY_DRAWDOWN_LIMIT_PCT}%")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "📊 Overview":

    # ── Equity Curve ──────────────────────────────────────────────────────────
    st.subheader("📈 Equity Curve")
    curve_data = db_manager.get_equity_curve(limit=288)  # ~24h at 5min intervals
    if curve_data:
        df_eq = pd.DataFrame(curve_data)
        df_eq["ts"] = pd.to_datetime(df_eq["ts"])
        df_eq = df_eq.set_index("ts")

        try:
            import altair as alt
            base  = alt.Chart(df_eq.reset_index()).mark_area(
                line={"color": "#00ff88"}, color=alt.Gradient(
                    gradient="linear",
                    stops=[alt.GradientStop(color="#00ff8833", offset=0),
                           alt.GradientStop(color="#00ff8800", offset=1)],
                    x1=1, x2=1, y1=1, y2=0,
                )
            ).encode(
                x=alt.X("ts:T", title="Time"),
                y=alt.Y("equity:Q", title="Equity ($)",
                        scale=alt.Scale(zero=False)),
                tooltip=["ts:T", "equity:Q", "daily_pnl:Q"],
            )
            st.altair_chart(base, use_container_width=True)
        except ImportError:
            st.line_chart(df_eq["equity"])
    else:
        st.info("Equity curve populates after first snapshot (every 5 min).")

    # ── Rolling Drawdown ──────────────────────────────────────────────────────
    if curve_data and len(curve_data) > 2:
        df_eq2 = pd.DataFrame(curve_data)
        df_eq2["ts"] = pd.to_datetime(df_eq2["ts"])
        eq_vals = df_eq2["equity"].values
        peak    = pd.Series(eq_vals).cummax()
        dd_vals = (eq_vals - peak) / peak * 100
        df_dd   = pd.DataFrame({"ts": df_eq2["ts"], "drawdown": dd_vals})
        st.subheader("📉 Rolling Drawdown (%)")
        st.area_chart(df_dd.set_index("ts")["drawdown"])

    st.divider()

    # ── Live Positions ────────────────────────────────────────────────────────
    st.subheader("📋 Live Positions")
    if positions:
        df_p = pd.DataFrame(positions)
        df_p["PnL"] = df_p["profit"].apply(lambda x: f"{'🟢' if x >= 0 else '🔴'} ${x:+,.2f}")
        st.dataframe(
            df_p.rename(columns={"ticket":"Ticket","symbol":"Symbol","type":"Side",
                                  "volume":"Lots","price_open":"Entry",
                                  "price_current":"Current","sl":"SL","profit":"Profit"}
                        )[["Ticket","Symbol","Side","Lots","Entry","Current","SL","PnL"]],
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No open positions.")
    st.divider()

    # ── Recent Signals ────────────────────────────────────────────────────────
    st.subheader("📡 Recent Signals (last 50)")
    emoji = {"EXECUTED": "✅", "REJECTED": "❌", "PENDING": "⏳"}
    try:
        with db_manager.get_connection() as conn:
            rows = conn.execute("""
                SELECT received_at, source, status, ai_confidence,
                       is_high_conviction, reject_reason,
                       json_extract(parsed_json,'$.symbol') AS symbol,
                       json_extract(parsed_json,'$.action') AS action
                FROM signals ORDER BY received_at DESC LIMIT 50
            """).fetchall()
        if rows:
            df_s = pd.DataFrame([dict(r) for r in rows])
            df_s["received_at"] = pd.to_datetime(df_s["received_at"]).dt.strftime("%H:%M:%S")
            df_s["status"]   = df_s["status"].apply(lambda s: f"{emoji.get(s,'?')} {s}")
            df_s["HC"]       = df_s["is_high_conviction"].apply(lambda x: "🔥" if x else "")
            st.dataframe(
                df_s[["received_at","symbol","action","status","HC","ai_confidence","source","reject_reason"]]
                .rename(columns={"received_at":"Time","ai_confidence":"Conf"}),
                use_container_width=True, hide_index=True,
            )
    except Exception as e:
        st.error(f"DB error: {e}")


elif page == "📈 Performance KPIs":

    @st.cache_data(ttl=30)
    def _fetch_kpi_trades():
        return db_manager.get_recent_closed_trades_with_session(500)

    kpi_trades = _fetch_kpi_trades()

    if not kpi_trades:
        st.info("No closed trades yet. KPIs will populate after the first trade closes.")
    else:
        kpi_trades_chrono = list(reversed(kpi_trades))

        # Row 1: Core KPI metric cards
        st.subheader("Core KPIs")
        k1, k2, k3, k4, k5, k6 = st.columns(6)

        all_pnls = [t["pnl"] for t in kpi_trades_chrono if t.get("pnl") is not None]
        wins_all = [p for p in all_pnls if p > 0]
        losses_all = [p for p in all_pnls if p <= 0]

        last20 = all_pnls[-20:] if len(all_pnls) >= 20 else all_pnls
        wr20 = sum(1 for p in last20 if p > 0) / len(last20) * 100 if last20 else 0
        prev20 = all_pnls[-40:-20] if len(all_pnls) >= 40 else []
        wr20_prev = sum(1 for p in prev20 if p > 0) / len(prev20) * 100 if prev20 else None
        wr20_delta = f"{wr20 - wr20_prev:+.1f}%" if wr20_prev is not None else None

        last100 = all_pnls[-100:] if len(all_pnls) >= 100 else all_pnls
        wr100 = sum(1 for p in last100 if p > 0) / len(last100) * 100 if last100 else 0

        pf_num = sum(wins_all) if wins_all else 0
        pf_den = abs(sum(losses_all)) if losses_all else 0
        pf = pf_num / pf_den if pf_den > 0 else float("inf")
        pf_str = f"{pf:.2f}" if pf != float("inf") else "∞"

        avg_win = sum(wins_all) / len(wins_all) if wins_all else 0
        avg_loss = abs(sum(losses_all) / len(losses_all)) if losses_all else 0
        avg_rr = avg_win / avg_loss if avg_loss > 0 else float("inf")
        rr_str = f"{avg_rr:.2f}" if avg_rr != float("inf") else "∞"

        expectancy = sum(all_pnls) / len(all_pnls) if all_pnls else 0

        consec_losses = 0
        for p in reversed(all_pnls):
            if p <= 0:
                consec_losses += 1
            else:
                break

        k1.metric("Win Rate (20)", f"{wr20:.1f}%", delta=wr20_delta)
        k2.metric("Win Rate (100)", f"{wr100:.1f}%")
        k3.metric("Profit Factor", pf_str)
        k4.metric("Avg R:R", rr_str)
        k5.metric("Expectancy", f"")
        k6.metric("Consec Losses", consec_losses)

        st.divider()

        # Row 2: Rolling charts
        st.subheader("Rolling Performance")
        col_eq, col_wr = st.columns(2)

        with col_eq:
            st.markdown("**Equity Curve**")
            curve_data = db_manager.get_equity_curve(limit=288)
            if curve_data:
                df_eq = pd.DataFrame(curve_data)
                df_eq["ts"] = pd.to_datetime(df_eq["ts"])
                st.line_chart(df_eq.set_index("ts")["equity"])
            else:
                st.caption("Equity snapshots not yet available.")

        with col_wr:
            st.markdown("**Rolling Win Rate (20-trade window)**")
            if len(all_pnls) >= 20:
                rolling_wr = []
                for i in range(19, len(all_pnls)):
                    window = all_pnls[i - 19 : i + 1]
                    wr_val = sum(1 for p in window if p > 0) / 20.0 * 100
                    rolling_wr.append({"Trade": i + 1, "WR%": wr_val})
                df_rwr = pd.DataFrame(rolling_wr)
                st.line_chart(df_rwr.set_index("Trade")["WR%"])
            else:
                st.caption(f"Need 20 trades for rolling WR (have {len(all_pnls)}).")

        st.divider()

        # Row 3: Session Breakdown
        st.subheader("Session Breakdown")
        session_hours = getattr(config, "SESSION_HOURS_UTC", {})

        def _derive_session(open_time_str):
            try:
                from datetime import datetime as _dt
                if isinstance(open_time_str, str):
                    ot = _dt.fromisoformat(open_time_str)
                else:
                    ot = open_time_str
                h = ot.hour
                for name, (s, e) in session_hours.items():
                    if s < e:
                        if s <= h < e:
                            return name
                    else:
                        if h >= s or h < e:
                            return name
                return "OTHER"
            except Exception:
                return "UNKNOWN"

        session_stats = {}
        for t in kpi_trades_chrono:
            sess = t.get("session") or _derive_session(t.get("open_time"))
            if sess not in session_stats:
                session_stats[sess] = {"trades": 0, "wins": 0, "pnls": [], "best": None, "worst": None}
            s = session_stats[sess]
            pnl_val = t.get("pnl") or 0
            s["trades"] += 1
            if pnl_val > 0:
                s["wins"] += 1
            s["pnls"].append(pnl_val)
            if s["best"] is None or pnl_val > s["best"]:
                s["best"] = pnl_val
            if s["worst"] is None or pnl_val < s["worst"]:
                s["worst"] = pnl_val

        sess_rows = []
        for name in ["LONDON", "OVERLAP", "NY", "ASIA", "OTHER", "UNKNOWN"]:
            if name not in session_stats:
                continue
            s = session_stats[name]
            wr_pct = s["wins"] / s["trades"] * 100 if s["trades"] > 0 else 0
            net = sum(s["pnls"])
            avg = net / s["trades"] if s["trades"] > 0 else 0
            sess_rows.append({
                "Session": name,
                "Trades": s["trades"],
                "Wins": s["wins"],
                "WR%": f"{wr_pct:.1f}%",
                "Net PnL": f"",
                "Avg PnL": f"",
                "Best": f"" if s["best"] is not None else "-",
                "Worst": f"" if s["worst"] is not None else "-",
            })

        if sess_rows:
            st.dataframe(pd.DataFrame(sess_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No session data available.")

        st.divider()

        # Row 4: Upcoming Gold News Events
        st.subheader("Upcoming Gold-Sensitive News (24h)")
        try:
            import asyncio as _aio
            from news_filter.news_filter import NewsFilter as _NF

            @st.cache_data(ttl=300)
            def _fetch_gold_news():
                nf = _NF()
                _aio.run(nf.refresh_if_needed())
                return nf.get_upcoming_gold_events(24)

            gold_events = _fetch_gold_news()
            if gold_events:
                st.dataframe(pd.DataFrame(gold_events), use_container_width=True, hide_index=True)
            else:
                st.caption("No upcoming gold-sensitive events in the next 24 hours.")
        except Exception as e:
            st.caption(f"News data unavailable: {e}")

        st.divider()

        # Row 5: ML Model Comparison
        st.subheader("ML Model Comparison")
        try:
            from quant.win_model import win_model
            comparison = win_model.get_shadow_comparison()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("RF Status", comparison["rf_status"])
            mc2.metric("RF CV Accuracy", f"{comparison['rf_cv']:.1%}" if comparison['rf_cv'] > 0 else "N/A")
            if comparison["xgb_available"]:
                mc3.metric("XGB CV Accuracy", f"{comparison['xgb_cv']:.1%}")
                winner = "XGBoost" if comparison["xgb_cv"] > comparison["rf_cv"] else "RandomForest"
                if comparison["rf_cv"] > 0 and comparison["xgb_cv"] > 0:
                    st.info(f"Shadow comparison winner: **{winner}** (RF={comparison['rf_cv']:.4f} vs XGB={comparison['xgb_cv']:.4f}) | Samples: {comparison['n_samples']}")
            else:
                mc3.metric("XGB Shadow", "Not available")
                st.caption("XGBoost shadow model activates after 100+ training samples and nightly optimization.")
        except Exception as e:
            st.caption(f"ML comparison unavailable: {e}")

        st.divider()

        # Row 6: SHAP Feature Impact
        st.subheader("SHAP Feature Impact (Last RF Retrain)")
        try:
            from quant.win_model import win_model as _wm
            if _wm._shap_explainer is not None and _wm._model is not None:
                import shap as _shap
                feat_imp = _wm._state.feature_importances
                if feat_imp:
                    df_shap = pd.DataFrame([
                        {"Feature": k, "Importance": v}
                        for k, v in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:15]
                    ])
                    st.bar_chart(df_shap.set_index("Feature")["Importance"])
                    st.caption("SHAP TreeExplainer is active. Per-trade SHAP values are appended to trade forensics.")
                else:
                    st.info("Feature importances not yet computed.")
            else:
                st.info("SHAP explainer not yet initialized (activates after first RF retrain with shap installed).")
        except ImportError:
            st.caption("shap package not installed \u2014 feature impact visualization unavailable.")
        except Exception as e:
            st.caption(f"SHAP visualization error: {e}")





# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ALPHA RANKER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🏆 Alpha Ranker":
    st.subheader("🏆 Signal Source Leaderboard")

    TIER_BADGE = {
        "S": "🥇 S", "A": "🥈 A", "B": "🥉 B",
        "C": "⚠️ C", "F": "🔇 F", "UNRATED": "❓ —",
    }
    TIER_MULT = {"S": "1.5×", "A": "1.2×", "B": "1.0×", "C": "0.5×", "F": "MUTED"}

    perf = db_manager.get_source_performance()
    if perf:
        try:
            from quant.alpha_ranker import alpha_ranker
            profiles = {p.source: p for p in alpha_ranker.get_all_profiles()}
        except Exception:
            profiles = {}

        rows = []
        for rank, s in enumerate(perf, 1):
            src     = s["source"]
            profile = profiles.get(src)
            tier    = profile.tier if profile else "UNRATED"
            mult    = TIER_MULT.get(tier, "1.0×")
            wr      = s["wins"] / max(s["total"], 1) * 100
            rows.append({
                "Rank":      rank,
                "Source":    src,
                "Trades":    s["total"],
                "Wins":      s["wins"],
                "Win Rate":  f"{wr:.1f}%",
                "Net PnL":   f"${s['net_pnl']:+,.2f}",
                "Avg/Trade": f"${s['avg_pnl']:+,.2f}",
                "Best":      f"${s['best_trade']:+,.2f}",
                "Worst":     f"${s['worst_trade']:+,.2f}",
                "Tier":      TIER_BADGE.get(tier, tier),
                "Multiplier": mult,
            })
        df_lb = pd.DataFrame(rows)
        st.dataframe(df_lb, use_container_width=True, hide_index=True)

        # Win rate bar chart
        if len(rows) > 1:
            df_wr = pd.DataFrame([{"Source": r["Source"], "WR": float(r["Win Rate"].replace("%", ""))}
                                   for r in rows])
            st.bar_chart(df_wr.set_index("Source")["WR"])
    else:
        st.info("Leaderboard populates after the first closed trade.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: BLACK BOX
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Black Box":
    st.subheader("🔍 Black Box Decision Audit")

    try:
        from quant.black_box import query_decisions, get_rejection_breakdown, get_noise_rate

        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            bb_limit  = st.number_input("Records", 10, 500, 50, step=10)
        with col_f2:
            bb_source = st.text_input("Filter Source", "")
        with col_f3:
            bb_dec    = st.selectbox("Filter Decision", ["", "EXECUTED", "REJECTED", "NOISE"])

        decisions = query_decisions(
            limit=bb_limit,
            source=bb_source or None,
            decision=bb_dec or None,
        )

        noise_rate = get_noise_rate(24)
        rej_break  = get_rejection_breakdown(200)

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Records", len(decisions))
        m2.metric("24h Noise Rate", f"{noise_rate:.1%}")
        m3.metric("Top Reject Reason",
                  max(rej_break, key=rej_break.get) if rej_break else "N/A")

        if decisions:
            df_bb = pd.DataFrame(decisions)
            show_cols = [c for c in [
                "ts","source","ai_symbol","ai_action","ai_confidence",
                "confluence_score","confluence_max","final_decision",
                "ai_reasoning","risk_reject_reason","alpha_tier","lot_size",
            ] if c in df_bb.columns]
            st.dataframe(df_bb[show_cols], use_container_width=True, hide_index=True)

            # Rejection breakdown chart
            if rej_break:
                st.subheader("Rejection Breakdown")
                df_rej = pd.DataFrame(
                    [{"Reason": k[:50], "Count": v} for k, v in rej_break.items()]
                ).sort_values("Count", ascending=False)
                st.bar_chart(df_rej.set_index("Reason")["Count"])
        else:
            st.info("No black box records yet.")
    except Exception as e:
        st.error(f"Black box error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EXPOSURE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🌡 Exposure":
    st.subheader("🌡️ Currency Exposure Heatmap (% Equity at Risk)")

    try:
        from quant.exposure_guard import get_portfolio_heatmap
        heatmap = get_portfolio_heatmap(equity or 10000)
        if heatmap:
            df_heat = pd.DataFrame(
                [{"Currency": k, "Risk %": round(v, 2)} for k, v in heatmap.items()]
            ).sort_values("Risk %", ascending=False)

            max_exp = config.MAX_CURRENCY_EXPOSURE_PCT
            for _, row in df_heat.iterrows():
                used = row["Risk %"] / max_exp * 100
                color = "🟢" if used < 50 else ("🟡" if used < 80 else "🔴")
                st.progress(
                    min(used / 100, 1.0),
                    text=f"{color} {row['Currency']}: {row['Risk %']:.2f}% / {max_exp}% limit"
                )
            st.bar_chart(df_heat.set_index("Currency")["Risk %"])
        else:
            st.info("No open positions — exposure is zero.")

        # Latency history
        st.subheader("📡 Latency History (last 50)")
        with db_manager.get_connection() as conn:
            lat_rows = conn.execute(
                "SELECT ts, avg_ms, jitter_ms, mode FROM latency_log ORDER BY ts DESC LIMIT 50"
            ).fetchall()
        if lat_rows:
            df_lat = pd.DataFrame([dict(r) for r in lat_rows])
            df_lat["ts"] = pd.to_datetime(df_lat["ts"])
            st.dataframe(df_lat, use_container_width=True, hide_index=True)
            st.line_chart(df_lat.set_index("ts")[["avg_ms","jitter_ms"]])
        else:
            st.info("No latency records yet.")
    except Exception as e:
        st.error(f"Exposure page error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SELF-CORRECTION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Self-Correction":
    st.subheader("🧠 AI Self-Correction Engine (Pillar 14)")

    try:
        import json as _json
        corrections_path = config.PROMPT_CORRECTIONS_FILE
        if os.path.exists(corrections_path):
            with open(corrections_path) as f:
                corr_data = _json.load(f)
            st.success(f"Last updated: {corr_data.get('last_updated', 'Unknown')}")
            st.markdown("**Last AI Summary:**")
            st.info(corr_data.get("last_summary", "No summary yet."))
            st.markdown("**Active Correction Rules:**")
            rules = corr_data.get("corrections", [])
            if rules:
                for i, rule in enumerate(rules, 1):
                    st.markdown(f"`{i}.` {rule}")
            else:
                st.info("No correction rules yet.")
        else:
            st.info("No corrections file yet — engine runs after enough closed trades.")

        # Manual trigger
        if st.button("🔄 Run Self-Correction Review Now"):
            with st.spinner("Running AI review..."):
                import asyncio
                from quant.self_correction import self_correction as sc
                asyncio.run(sc._run_review())
            st.success("Review complete — check corrections file.")
    except Exception as e:
        st.error(f"Self-correction page error: {e}")

    # Audit log
    with st.expander("📋 Audit Log (last 30)"):
        try:
            with db_manager.get_connection() as conn:
                rows = conn.execute(
                    "SELECT ts, event_type, details FROM audit_log ORDER BY ts DESC LIMIT 30"
                ).fetchall()
            if rows:
                df_a = pd.DataFrame([dict(r) for r in rows])
                df_a["ts"] = pd.to_datetime(df_a["ts"]).dt.strftime("%H:%M:%S")
                st.dataframe(df_a, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Audit error: {e}")


# ── AUTO-REFRESH ──────────────────────────────────────────────────────────────
time.sleep(refresh)
st.rerun()
