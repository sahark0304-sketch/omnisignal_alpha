"""
quant/win_model.py -- Random Forest Win-Probability Model with Explainable AI.
Core ML brain of OmniSignal Alpha: predicts P(win), explains decisions, and self-optimizes nightly.
"""

import os
import pickle
import asyncio
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import config
from database import db_manager
from utils.logger import get_logger

logger = get_logger(__name__)

_sklearn_available = False
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    _sklearn_available = True
except ImportError:
    logger.warning("[ML-Brain] scikit-learn not installed -- model will stay DORMANT")

_xgb_available = False
try:
    import xgboost as xgb
    _xgb_available = True
except ImportError:
    logger.info("[ML-Brain] xgboost not installed -- shadow model disabled")

_shap_available = False
try:
    import shap
    _shap_available = True
except ImportError:
    logger.info("[ML-Brain] shap not installed -- explainability disabled")


# ---------------------------------------------------------------------------
#  Canonical feature vector (18 features)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "vol_regime_ratio",
    "realized_vol_rank",
    "rsi_m5",
    "rsi_m15",
    "rsi_h1",
    "rsi_divergence_score",
    "tick_density_ratio",
    "bid_ask_imbalance",
    "volume_surge",
    "momentum_m15",
    "mean_rev_z",
    "minutes_into_session",
    "is_session_open_30min",
    "hurst_50",
    "confluence_score",
    "ai_confidence",
    "source_win_rate",
    "spread_pips",
    "time_since_last_trade_mins",
    "current_dd_pct",
]

SESSION_MAP = {"LONDON": 0, "NY": 1, "OVERLAP": 2, "ASIA": 3}


def _bayesian_win_rate(source_stats: Dict) -> float:
    """
    Beta-Binomial Bayesian smoothing for source win rate.

    Solves the cold-start problem: new sources (especially AUTO_*) have zero
    history, which tanks the ML score.  Instead of raw wins/total, we use a
    Beta(3, 2) prior centered at 60%.  With 0 trades this returns 0.60; as
    real data accumulates the prior washes out:
        0 trades -> 0.60  |  5 trades (3W) -> 0.60  |  20 trades (10W) -> 0.52
    """
    ALPHA = 3.0
    BETA = 2.0
    wins = float(source_stats.get("wins", 0))
    total = float(source_stats.get("total", 0))
    return (wins + ALPHA) / (total + ALPHA + BETA)


PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 6},
    {"n_estimators": 200, "max_depth": 8},
    {"n_estimators": 300, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 10},
    {"n_estimators": 150, "max_depth": 12, "min_samples_leaf": 3},
]


@dataclass
class ModelState:
    status: str = "DORMANT"
    n_samples: int = 0
    n_positive: int = 0
    accuracy_cv: float = 0.0
    last_train_ts: Optional[datetime] = None
    feature_importances: Dict[str, float] = field(default_factory=dict)


class WinProbabilityModel:

    def __init__(self):
        self._model: Optional[object] = None
        self._scaler: Optional[object] = None
        self._state = ModelState()
        self._model_path = Path(config.DATA_DIR) / "win_model.pkl"
        self._min_samples = getattr(config, "WIN_MODEL_MIN_SAMPLES", 50)
        self._retrain_every = getattr(config, "WIN_MODEL_RETRAIN_EVERY", 20)
        self._min_prob = getattr(config, "WIN_MODEL_MIN_PROB", 0.55)
        self._last_train_count = 0
        self._active_feature_names: List[str] = list(FEATURE_NAMES)
        self._xgb_model: Optional[object] = None
        self._xgb_cv_score: float = 0.0
        self._shap_explainer: Optional[object] = None
        self._best_params: Dict = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_leaf": 5,
        }

        if not _sklearn_available:
            self._state.status = "DORMANT"
            logger.warning(
                "[ML-Brain] DORMANT -- scikit-learn not installed. "
                "Run: pip install scikit-learn pandas"
            )
            return

        if not getattr(config, "WIN_MODEL_ENABLED", False):
            self._state.status = "OBSERVATION"
            logger.info(
                "[ML-Brain] Master switch OFF -- OBSERVATION mode "
                "(shadow predictions active, model will not block trades)"
            )
            self._try_load_model()
            return

        self._try_load_model()

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def state(self) -> ModelState:
        return self._state

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def predict(self, features: Dict) -> Tuple[Optional[float], str]:
        """Return (P(win), explanation_string) or (None, '') if not ready."""
        if self._state.status != "READY" or self._model is None:
            return None, ""
        try:
            X = self._engineer_single(features)
            if X is None:
                return None, ""
            X_scaled = self._scaler.transform(X.reshape(1, -1))
            prob = float(self._model.predict_proba(X_scaled)[0, 1])
            prob = round(prob, 4)

            explanation = self._build_explanation(prob, features)
            threshold = self._min_prob

            if prob >= threshold:
                logger.info(
                    "[ML-Brain] Trade Approved: "
                    + f"{prob:.0%} P(win). {self._top_drivers_str()}"
                )
            else:
                logger.info(
                    "[ML-Brain] Trade Rejected: "
                    + f"{prob:.0%} P(win) < "
                    + f"{threshold:.0%} threshold. {self._top_drivers_str()}"
                )

            return prob, explanation
        except Exception as e:
            logger.error(f"[ML-Brain] Predict error: {e}")
            return None, ""

    def shadow_predict(self, features: Dict) -> Optional[float]:
        """Run predict_proba() regardless of WIN_MODEL_ENABLED flag.
        Used for silent auditing -- never blocks trades."""
        if self._model is None or self._scaler is None:
            return None
        try:
            X = self._engineer_single(features)
            if X is None:
                return None
            X_scaled = self._scaler.transform(X.reshape(1, -1))
            prob = float(self._model.predict_proba(X_scaled)[0, 1])
            return round(prob, 4)
        except Exception:
            return None

    def shadow_predict_xgb(self, features: Dict) -> Optional[float]:
        """XGBoost shadow prediction — strictly logging, never blocks trades."""
        if self._xgb_model is None or self._scaler is None:
            return None
        try:
            X = self._engineer_single(features)
            if X is None:
                return None
            X_scaled = self._scaler.transform(X.reshape(1, -1))
            prob = float(self._xgb_model.predict_proba(X_scaled)[0, 1])
            return round(prob, 4)
        except Exception:
            return None

    def get_shadow_comparison(self) -> Dict:
        """Return RF vs XGBoost CV scores for dashboard display."""
        return {
            "rf_cv": round(self._state.accuracy_cv, 4),
            "xgb_cv": round(self._xgb_cv_score, 4),
            "xgb_available": _xgb_available and self._xgb_model is not None,
            "rf_status": self._state.status,
            "n_samples": self._state.n_samples,
        }

    def explain_prediction(self, features: Dict) -> Optional[Dict]:
        """Return per-feature SHAP values for a single prediction, sorted by |impact|.
        Used by post-trade forensic callback for Explainable AI."""
        if not _shap_available or self._model is None or self._scaler is None:
            return None
        try:
            if self._shap_explainer is None:
                self._shap_explainer = shap.TreeExplainer(self._model)

            X = self._engineer_single(features)
            if X is None:
                return None
            X_scaled = self._scaler.transform(X.reshape(1, -1))
            shap_values = self._shap_explainer.shap_values(X_scaled)

            if isinstance(shap_values, list):
                sv = shap_values[1][0]  # class=1 (WIN)
            else:
                sv = shap_values[0]

            feat_names = self._active_feature_names
            pairs = list(zip(feat_names, sv))
            pairs.sort(key=lambda kv: abs(kv[1]), reverse=True)

            return {
                "shap_drivers": [
                    {"feature": name, "shap_value": round(float(val), 4)}
                    for name, val in pairs[:10]
                ],
                "top5_summary": ", ".join(
                    f"{name}={'+' if val>0 else ''}{val:.3f}"
                    for name, val in pairs[:5]
                ),
            }
        except Exception as e:
            logger.debug("[ML-Brain] SHAP explain_prediction failed: %s", e)
            return None

    def check_and_retrain(self):
        """Called after a trade closes.  Retrains if enough new data.
        Runs even when WIN_MODEL_ENABLED=False to support shadow predictions."""
        if not _sklearn_available:
            return

        current_count = self._count_labeled_trades()
        self._state.n_samples = current_count

        if current_count < self._min_samples:
            self._state.status = "OBSERVATION"
            logger.info(
                f"[ML-Brain] OBSERVATION: {current_count}/{self._min_samples} "
                + "labeled trades -- waiting for more data"
            )
            return

        if self._model is None:
            logger.info(
                f"[ML-Brain] Threshold reached ({current_count}/{self._min_samples}) "
                + "-- triggering initial training -> READY"
            )
            self._train()
            return

        if current_count - self._last_train_count >= self._retrain_every:
            self._train()

    def get_feature_importance(self) -> Dict[str, float]:
        return dict(self._state.feature_importances)

    def get_current_stats(self) -> Dict:
        """Structured stats for dashboard and terminal display."""
        n_labeled = self._count_labeled_trades()
        top_features = list(self._state.feature_importances.items())[:3]

        return {
            "status": self._state.status,
            "n_samples": n_labeled,
            "n_required": self._min_samples,
            "progress_pct": round(
                min(n_labeled / max(self._min_samples, 1), 1.0) * 100, 1
            ),
            "n_positive": self._state.n_positive,
            "n_negative": self._state.n_samples - self._state.n_positive,
            "accuracy_cv": self._state.accuracy_cv,
            "last_train": (
                self._state.last_train_ts.strftime("%Y-%m-%d %H:%M")
                if self._state.last_train_ts else None
            ),
            "top_features": [
                {"name": name, "importance": imp} for name, imp in top_features
            ],
            "model_type": "RandomForest",
            "best_params": dict(self._best_params),
        }

    def get_prediction_explanation(self, features: Dict) -> Dict:
        """Full structured explanation for logging."""
        if self._model is None or self._scaler is None:
            return {}
        try:
            X = self._engineer_single(features)
            if X is None:
                return {}
            X_scaled = self._scaler.transform(X.reshape(1, -1))
            prob = float(self._model.predict_proba(X_scaled)[0, 1])

            importances = self._model.feature_importances_
            feat_names = self._active_feature_names
            driver_list = sorted(
                zip(feat_names, importances),
                key=lambda kv: kv[1],
                reverse=True,
            )
            top_drivers = []
            for name, imp in driver_list[:5]:
                idx = self._feature_index(name)
                val = float(X[idx]) if idx is not None else 0.0
                top_drivers.append({
                    "name": name,
                    "importance_pct": round(float(imp) * 100, 1),
                    "value": round(val, 4),
                })

            if prob >= 0.70:
                confidence_band = "HIGH"
            elif prob >= self._min_prob:
                confidence_band = "MEDIUM"
            else:
                confidence_band = "LOW"

            return {
                "p_win": round(prob, 4),
                "decision": "APPROVE" if prob >= self._min_prob else "REJECT",
                "top_drivers": top_drivers,
                "regime": features.get("vol_regime_label", "UNKNOWN"),
                "confidence_band": confidence_band,
            }
        except Exception as e:
            logger.error(f"[ML-Brain] Explanation error: {e}")
            return {}

    # ------------------------------------------------------------------
    #  Nightly Optimization
    # ------------------------------------------------------------------

    async def nightly_optimization(self) -> str:
        """
        Run once per day during low-activity hours.
        1. Retrain on all available data
        2. Test multiple hyperparameter configs (n_estimators, max_depth)
        3. Feature importance pruning: drop features with importance < 1%
        4. Cross-validation with multiple folds
        5. Generate a text report of discoveries
        Returns the report as a string.
        """
        if not _sklearn_available:
            return "[ML-Brain] Nightly skipped -- scikit-learn not available."

        logger.info("[ML-Brain] === Nightly Optimization Started ===")
        report_lines = ["=== OmniSignal Nightly ML Report ===", ""]
        t0 = datetime.now()

        rows = await asyncio.to_thread(db_manager.get_win_model_training_data)
        if not rows or len(rows) < self._min_samples:
            count = len(rows) if rows else 0
            msg = f"[ML-Brain] Nightly: insufficient data ({count} rows)"
            logger.info(msg)
            return msg

        X_list, y_list, row_metas = [], [], []
        for row in rows:
            x = self._engineer_row(row)
            if x is None:
                continue
            label = 1 if (row.get("pnl") or 0) > 0 else 0
            X_list.append(x)
            y_list.append(label)
            row_metas.append(row)

        if len(X_list) < self._min_samples:
            msg = f"[ML-Brain] Nightly: usable rows {len(X_list)} < {self._min_samples}"
            logger.info(msg)
            return msg

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int32)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        overall_wr = float(np.mean(y))
        report_lines.append(f"Total samples: {len(X_list)}")
        report_lines.append(f"Overall win rate: {overall_wr:.1%}")
        report_lines.append("")

        report_lines.append("--- Hyperparameter Search ---")
        best_score = -1.0
        best_cfg = PARAM_GRID[1]

        for cfg in PARAM_GRID:
            rf = RandomForestClassifier(
                n_estimators=cfg.get("n_estimators", 200),
                max_depth=cfg.get("max_depth", 8),
                min_samples_leaf=cfg.get("min_samples_leaf", 5),
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
            try:
                n_folds = min(5, max(2, len(X_list) // 10))
                skf = StratifiedKFold(
                    n_splits=n_folds, shuffle=True, random_state=42
                )
                scores = cross_val_score(
                    rf, X_scaled, y, cv=skf, scoring="accuracy"
                )
                mean_score = float(np.mean(scores))
            except Exception:
                mean_score = 0.0

            line = (
                f"  n_est={cfg.get('n_estimators')}, "
                + f"depth={cfg.get('max_depth')}, "
                + f"leaf={cfg.get('min_samples_leaf', 5)} "
                + f"-> CV={mean_score:.4f}"
            )
            report_lines.append(line)
            logger.info(f"[ML-Brain] Nightly HP: {line.strip()}")

            if mean_score > best_score:
                best_score = mean_score
                best_cfg = dict(cfg)

        best_cfg.setdefault("min_samples_leaf", 5)
        self._best_params = best_cfg
        report_lines.append(f"  WINNER: {best_cfg} -> CV={best_score:.4f}")
        report_lines.append("")

        rf_best = RandomForestClassifier(
            n_estimators=best_cfg["n_estimators"],
            max_depth=best_cfg["max_depth"],
            min_samples_leaf=best_cfg["min_samples_leaf"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_best.fit(X_scaled, y)

        importances = rf_best.feature_importances_
        feat_imp_pairs = list(zip(self._active_feature_names, importances))
        feat_imp_pairs.sort(key=lambda kv: kv[1], reverse=True)

        # --- XGBoost Shadow Model ---
        xgb_cv_score = 0.0
        if _xgb_available and len(X_list) >= 100:
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    eval_metric="logloss",
                    random_state=42, n_jobs=-1,
                )
                n_folds_xgb = min(5, max(2, len(X_list) // 10))
                skf_xgb = StratifiedKFold(n_splits=n_folds_xgb, shuffle=True, random_state=42)
                xgb_scores = cross_val_score(xgb_model, X_scaled, y, cv=skf_xgb, scoring="accuracy")
                xgb_cv_score = float(np.mean(xgb_scores))
                xgb_model.fit(X_scaled, y)
                self._xgb_model = xgb_model
                self._xgb_cv_score = xgb_cv_score
                winner = "XGBoost" if xgb_cv_score > best_score else "RandomForest"
                report_lines.append("--- XGBoost Shadow ---")
                report_lines.append(f"  XGB CV={xgb_cv_score:.4f} vs RF CV={best_score:.4f}")
                report_lines.append(f"  Shadow winner: {winner}")
                report_lines.append("")
                logger.info(
                    "[ML-Brain] XGBoost shadow: CV=%.4f vs RF CV=%.4f -> %s",
                    xgb_cv_score, best_score, winner,
                )
            except Exception as e:
                report_lines.append(f"--- XGBoost Shadow FAILED: {e} ---")
                report_lines.append("")
                logger.warning("[ML-Brain] XGBoost shadow training failed: %s", e)
        elif _xgb_available:
            report_lines.append("--- XGBoost Shadow: skipped (need >=100 samples) ---")
            report_lines.append("")

        report_lines.append("--- Feature Importances ---")
        pruned_features = []
        for name, imp in feat_imp_pairs:
            marker = " [PRUNED]" if imp < 0.01 else ""
            report_lines.append(
                f"  {name}: {imp:.4f} ({imp*100:.1f}%){marker}"
            )
            if imp < 0.01:
                pruned_features.append(name)

        if pruned_features:
            report_lines.append(
                "  Features below 1%: " + ", ".join(pruned_features)
            )
        report_lines.append("")

        self._model = rf_best
        self._scaler = scaler
        self._last_train_count = len(X_list)

        imp_dict = {}
        for name, imp in feat_imp_pairs:
            imp_dict[name] = round(float(imp), 4)

        self._state.status = "READY"
        self._state.n_samples = len(X_list)
        self._state.n_positive = int(np.sum(y))
        self._state.accuracy_cv = round(best_score, 4)
        self._state.last_train_ts = datetime.now()
        self._state.feature_importances = imp_dict

        self._save_model()

        # Recompute SHAP explainer after retrain
        if _shap_available and self._model is not None:
            try:
                self._shap_explainer = shap.TreeExplainer(self._model)
                report_lines.append("--- SHAP Explainer: recomputed successfully ---")
                report_lines.append("")
                logger.info("[ML-Brain] SHAP TreeExplainer recomputed after nightly retrain.")
            except Exception as e:
                self._shap_explainer = None
                report_lines.append(f"--- SHAP Explainer: FAILED ({e}) ---")
                report_lines.append("")
                logger.warning("[ML-Brain] SHAP explainer recompute failed: %s", e)

        # Breakthrough #2: Check for feature importance drift
        try:
            self._detect_feature_drift(X_scaled, y)
        except Exception as e:
            logger.debug(f'[ML-Brain] Drift check skipped: {e}')

        # --- Correlation discovery ---
        self._nightly_session_analysis(report_lines, row_metas, y_list, overall_wr)
        self._nightly_regime_analysis(report_lines, row_metas, y_list, overall_wr)
        self._nightly_source_analysis(report_lines, row_metas, y_list, overall_wr)

        elapsed = (datetime.now() - t0).total_seconds()
        report_lines.append(f"Completed in {elapsed:.1f}s")
        report_lines.append("=== End Report ===")

        full_report = "\n".join(report_lines)

        for line in report_lines:
            if line.strip():
                logger.info(f"[ML-Brain] {line}")

        try:
            db_manager.set_system_state("nightly_report", full_report)
        except Exception as e:
            logger.error(f"[ML-Brain] Failed to store nightly report: {e}")

        logger.info("[ML-Brain] === Nightly Optimization Complete ===")
        return full_report

    def _nightly_session_analysis(self, report_lines, row_metas, y_list, overall_wr):
        report_lines.append("--- Win Rate by Session ---")
        session_stats = {}
        for row, label in zip(row_metas, y_list):
            sess = (row.get("session") or "UNKNOWN").upper()
            if sess not in session_stats:
                session_stats[sess] = {"total": 0, "wins": 0}
            session_stats[sess]["total"] += 1
            session_stats[sess]["wins"] += label

        for sess in ["ASIA", "LONDON", "NY", "OVERLAP"]:
            st = session_stats.get(sess, {"total": 0, "wins": 0})
            if st["total"] > 0:
                wr = st["wins"] / st["total"]
                flag = ""
                if st["total"] >= 10 and abs(wr - overall_wr) > 0.10:
                    flag = " ** SIGNIFICANT DEVIATION **"
                report_lines.append(
                    f"  {sess}: {st['wins']}/{st['total']} = {wr:.1%}{flag}"
                )
            else:
                report_lines.append(f"  {sess}: no data")
        report_lines.append("")

    def _nightly_regime_analysis(self, report_lines, row_metas, y_list, overall_wr):
        report_lines.append("--- Win Rate by Volatility Regime ---")
        vol_stats = {}
        for row, label in zip(row_metas, y_list):
            regime = (
                row.get("vol_regime_label")
                or row.get("hurst_regime")
                or "UNKNOWN"
            ).upper()
            if regime not in vol_stats:
                vol_stats[regime] = {"total": 0, "wins": 0}
            vol_stats[regime]["total"] += 1
            vol_stats[regime]["wins"] += label

        for regime in ["LOW", "NORMAL", "HIGH", "UNKNOWN"]:
            st = vol_stats.get(regime, {"total": 0, "wins": 0})
            if st["total"] > 0:
                wr = st["wins"] / st["total"]
                flag = ""
                if st["total"] >= 10 and abs(wr - overall_wr) > 0.10:
                    flag = " ** SIGNIFICANT DEVIATION **"
                report_lines.append(
                    f"  {regime}: {st['wins']}/{st['total']} = {wr:.1%}{flag}"
                )
        report_lines.append("")

    def _nightly_source_analysis(self, report_lines, row_metas, y_list, overall_wr):
        report_lines.append("--- Win Rate by Source ---")
        src_stats = {}
        for row, label in zip(row_metas, y_list):
            src = row.get("source") or "UNKNOWN"
            if src not in src_stats:
                src_stats[src] = {"total": 0, "wins": 0}
            src_stats[src]["total"] += 1
            src_stats[src]["wins"] += label

        for src, st in sorted(
            src_stats.items(), key=lambda kv: kv[1]["total"], reverse=True
        ):
            if st["total"] > 0:
                wr = st["wins"] / st["total"]
                flag = ""
                if st["total"] >= 10 and abs(wr - overall_wr) > 0.10:
                    flag = " ** SIGNIFICANT DEVIATION **"
                report_lines.append(
                    f"  {src}: {st['wins']}/{st['total']} = {wr:.1%}{flag}"
                )
        report_lines.append("")

    # ------------------------------------------------------------------
    #  Training
    # ------------------------------------------------------------------

    def _train(self):
        """Fetch labeled data, engineer features, train Random Forest."""
        logger.info("[ML-Brain] Training started...")
        t0 = datetime.now()

        rows = db_manager.get_win_model_training_data()
        if not rows or len(rows) < self._min_samples:
            self._state.status = "OBSERVATION"
            self._state.n_samples = len(rows) if rows else 0
            logger.info(f"[ML-Brain] Not enough data: {self._state.n_samples}")
            return

        X_list, y_list = [], []
        skipped = 0

        for row in rows:
            x = self._engineer_row(row)
            if x is None:
                skipped += 1
                continue
            label = 1 if (row.get("pnl") or 0) > 0 else 0
            X_list.append(x)
            y_list.append(label)

        if len(X_list) < self._min_samples:
            self._state.status = "OBSERVATION"
            logger.info(
                f"[ML-Brain] Usable rows {len(X_list)} < "
                + f"{self._min_samples} after cleaning"
            )
            return

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.int32)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        model = RandomForestClassifier(
            n_estimators=self._best_params.get("n_estimators", 200),
            max_depth=self._best_params.get("max_depth", 8),
            min_samples_leaf=self._best_params.get("min_samples_leaf", 5),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_scaled, y)

        cv_score = 0.0
        if len(X_list) >= 20:
            n_folds = min(5, max(2, len(X_list) // 10))
            try:
                skf = StratifiedKFold(
                    n_splits=n_folds, shuffle=True, random_state=42
                )
                scores = cross_val_score(
                    model, X_scaled, y, cv=skf, scoring="accuracy"
                )
                cv_score = float(np.mean(scores))
            except Exception:
                cv_score = 0.0

        self._model = model
        self._last_train_count = len(X_list)

        importances = {}
        if hasattr(model, "feature_importances_"):
            fi = model.feature_importances_
            feat_names = self._active_feature_names
            if len(fi) == len(feat_names):
                for name, imp in zip(feat_names, fi):
                    importances[name] = round(float(imp), 4)

        self._state.status = "READY"
        self._state.n_samples = len(X_list)
        self._state.n_positive = int(np.sum(y))
        self._state.accuracy_cv = round(cv_score, 4)
        self._state.last_train_ts = datetime.now()
        self._state.feature_importances = dict(
            sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
        )

        elapsed = (datetime.now() - t0).total_seconds()
        logger.info(
            f"[ML-Brain] READY | {len(X_list)} samples "
            + f"({int(np.sum(y))} wins, "
            + f"{len(X_list) - int(np.sum(y))} losses) | "
            + f"CV accuracy: {cv_score:.1%} | trained in {elapsed:.1f}s"
        )

        top3 = list(self._state.feature_importances.items())[:3]
        for name, imp in top3:
            logger.info(
                f"[ML-Brain]   Top feature: {name} = "
                + f"{imp:.4f} ({imp*100:.1f}%)"
            )

        self._save_model()

        if _shap_available and self._model is not None:
            try:
                self._shap_explainer = shap.TreeExplainer(self._model)
                logger.info("[ML-Brain] SHAP TreeExplainer recomputed after retrain.")
            except Exception as e:
                self._shap_explainer = None
                logger.debug("[ML-Brain] SHAP explainer failed: %s", e)

        if getattr(config, "WIN_MODEL_AUTO_ENABLE", False) and not getattr(config, "WIN_MODEL_ENABLED", False):
            min_acc = getattr(config, "WIN_MODEL_AUTO_ENABLE_MIN_ACC", 0.55)
            if cv_score >= min_acc:
                config.WIN_MODEL_ENABLED = True
                db_manager.set_system_state("win_model_enabled", "1")
                db_manager.log_audit("ML_AUTO_ENABLED", {
                    "cv_accuracy": round(cv_score, 4),
                    "threshold": min_acc,
                    "n_samples": len(X_list),
                })
                logger.info(
                    f"[ML-Brain] AUTO-ENABLED: CV accuracy {cv_score:.1%} >= "
                    f"{min_acc:.0%} threshold. ML gate is now ACTIVE."
                )
            else:
                logger.info(
                    f"[ML-Brain] Auto-enable deferred: CV accuracy {cv_score:.1%} < "
                    f"{min_acc:.0%} threshold. Model stays advisory-only."
                )

    # ------------------------------------------------------------------
    #  Feature engineering
    # ------------------------------------------------------------------

    def _engineer_row(self, row: Dict) -> Optional[np.ndarray]:
        """Transform a DB row into an 18-element feature vector matching
        FEATURE_NAMES order.  Handles both new-format rows (with
        vol_regime_ratio, rsi_m5, etc.) and legacy 14-feature rows by
        filling missing columns with sensible defaults."""
        try:
            has_new_features = row.get("vol_regime_ratio") is not None

            if has_new_features:
                vec = self._extract_new_features(row)
            else:
                vec = self._extract_legacy_features(row)

            if vec is None:
                return None

            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
            return vec

        except Exception as e:
            logger.debug(f"[ML-Brain] Feature engineering failed for row: {e}")
            return None

    def _extract_new_features(self, row: Dict) -> Optional[np.ndarray]:
        """Extract the full 18-feature vector from a new-format row."""
        source = row.get("source") or ""
        source_stats = db_manager.get_source_win_rate(source)
        source_wr = _bayesian_win_rate(source_stats)

        return np.array([
            float(row.get("vol_regime_ratio") or 0.0),
            float(row.get("realized_vol_rank") or 0.5),
            float(row.get("rsi_m5") or 50.0),
            float(row.get("rsi_m15") or 50.0),
            float(row.get("rsi_h1") or 50.0),
            float(row.get("rsi_divergence_score") or 0.0),
            float(row.get("tick_density_ratio") or 1.0),
            float(row.get("bid_ask_imbalance") or 0.0),
            float(row.get("volume_surge") or 1.0),
            float(row.get("momentum_m15") or 0.0),
            float(row.get("mean_rev_z") or 0.0),
            float(row.get("minutes_into_session") or 0.0),
            float(row.get("is_session_open_30min") or 0.0),
            float(row.get("hurst_50") or 0.5),
            float(row.get("confluence_score") or 0.0),
            float(row.get("ai_confidence") or 7.0),
            source_wr,
            float(row.get("spread_pips") or 0.0),
            float(row.get("time_since_last_trade_mins") or 999.0),
            float(row.get("current_dd_pct") or 0.0),
        ], dtype=np.float64)

    def _extract_legacy_features(self, row: Dict) -> Optional[np.ndarray]:
        """Map legacy 14-feature rows to the 18-feature vector with defaults."""
        hurst = row.get("hurst_50")
        if hurst is None:
            return None

        vol_z = float(row.get("vol_z_score") or 0.0)
        spread = float(row.get("spread_pips") or 0.0)
        conf_score = float(row.get("confluence_score") or 0.0)
        ai_conf = float(row.get("ai_confidence") or 7.0)

        source = row.get("source") or ""
        source_stats = db_manager.get_source_win_rate(source)
        source_wr = _bayesian_win_rate(source_stats)

        session_str = (row.get("session") or "LONDON").upper()
        session_code = float(SESSION_MAP.get(session_str, 0))
        minutes_approx = session_code * 120.0

        return np.array([
            abs(vol_z) if vol_z != 0.0 else 1.0,     # vol_regime_ratio approx
            0.5,                                       # realized_vol_rank default
            50.0,                                      # rsi_m5 neutral
            50.0,                                      # rsi_m15 neutral
            50.0,                                      # rsi_h1 neutral
            0.0,                                       # rsi_divergence_score
            1.0,                                       # tick_density_ratio neutral
            0.0,                                       # bid_ask_imbalance neutral
            1.0,                                       # volume_surge neutral
            0.0,                                       # momentum_m15 neutral
            vol_z,                                     # mean_rev_z from vol_z_score
            minutes_approx,                            # minutes_into_session approx
            0.0,                                       # is_session_open_30min unknown
            float(hurst),                              # hurst_50
            conf_score,                                # confluence_score
            ai_conf,                                   # ai_confidence
            source_wr,                                 # source_win_rate
            spread,                                    # spread_pips
            999.0,                                     # time_since_last_trade_mins default
            0.0,                                       # current_dd_pct default
        ], dtype=np.float64)

    def _engineer_single(self, features: Dict) -> Optional[np.ndarray]:
        """Engineer features from a live signal context dict."""
        return self._engineer_row(features)

    # ------------------------------------------------------------------
    #  Explanation helpers
    # ------------------------------------------------------------------

    def _build_explanation(self, prob: float, features: Dict) -> str:
        """Build human-readable explanation from feature importances."""
        if not hasattr(self._model, "feature_importances_"):
            return f"P(win)={prob:.2f}"

        importances = self._model.feature_importances_
        feat_names = self._active_feature_names

        if len(importances) != len(feat_names):
            return f"P(win)={prob:.2f}"

        pairs = sorted(
            zip(feat_names, importances),
            key=lambda kv: kv[1],
            reverse=True,
        )

        top3 = pairs[:3]
        parts = [f"{name}({imp*100:.0f}%)" for name, imp in top3]
        return f"P(win)={prob:.2f} | Key: " + ", ".join(parts)

    def _top_drivers_str(self) -> str:
        """Format top-3 feature importances for log output."""
        if not self._state.feature_importances:
            return ""
        top3 = list(self._state.feature_importances.items())[:3]
        parts = [f"{name} ({imp*100:.0f}%)" for name, imp in top3]
        return "Key drivers: " + ", ".join(parts)

    def _detect_feature_drift(self, X: "np.ndarray", y: "np.ndarray"):
        """
        Breakthrough #2: Online Feature Drift Detection.

        After each retrain, compare feature importances from the last 10
        trades vs the full dataset.  When a feature's importance shifts
        significantly (>2x), log a regime drift alert.  This enables the
        system to adapt to changing market conditions.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np

            if len(X) < 15:
                return

            recent_n = min(10, len(X) // 3)
            X_recent = X[-recent_n:]
            y_recent = y[-recent_n:]

            if len(np.unique(y_recent)) < 2:
                return

            rf_recent = RandomForestClassifier(
                n_estimators=50, max_depth=5,
                min_samples_leaf=2, random_state=42, n_jobs=-1,
            )
            rf_recent.fit(X_recent, y_recent)
            recent_imp = rf_recent.feature_importances_

            if not hasattr(self._model, "feature_importances_"):
                return
            full_imp = self._model.feature_importances_

            feat_names = self._active_feature_names
            if len(recent_imp) != len(feat_names) or len(full_imp) != len(feat_names):
                return

            drifts = []
            for i, name in enumerate(feat_names):
                if full_imp[i] < 0.01:
                    continue
                ratio = recent_imp[i] / max(full_imp[i], 0.001)
                if ratio > 2.5 or ratio < 0.3:
                    drifts.append((name, full_imp[i], recent_imp[i], ratio))

            if drifts:
                for name, old_imp, new_imp, ratio in drifts:
                    direction = "SURGING" if ratio > 1 else "FADING"
                    logger.warning(
                        f"[ML-Brain] FEATURE DRIFT: {name} {direction} | "
                        f"full={old_imp:.3f} recent={new_imp:.3f} "
                        f"(ratio={ratio:.1f}x) | last {recent_n} trades"
                    )
                from database import db_manager
                db_manager.log_audit("FEATURE_DRIFT", {
                    "drifts": [
                        {"feature": d[0], "full_imp": round(d[1], 4),
                         "recent_imp": round(d[2], 4), "ratio": round(d[3], 2)}
                        for d in drifts
                    ],
                    "recent_n": recent_n,
                    "total_n": len(X),
                })
            else:
                logger.debug(
                    f"[ML-Brain] Feature drift check: no significant drift "
                    f"(last {recent_n} vs {len(X)} trades)"
                )
        except Exception as e:
            logger.debug(f"[ML-Brain] Feature drift check error: {e}")


    def _feature_index(self, name: str) -> Optional[int]:
        """Get index of a feature name in the active feature list."""
        try:
            return self._active_feature_names.index(name)
        except ValueError:
            return None

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------

    def _save_model(self):
        try:
            os.makedirs(os.path.dirname(self._model_path), exist_ok=True)
            with open(self._model_path, "wb") as f:
                pickle.dump({
                    "model": self._model,
                    "scaler": self._scaler,
                    "state": self._state,
                    "last_train_count": self._last_train_count,
                    "feature_names": list(self._active_feature_names),
                    "best_params": dict(self._best_params),
                    "xgb_model": self._xgb_model,
                }, f)
            logger.info(f"[ML-Brain] Model saved to {self._model_path}")
        except Exception as e:
            logger.error(f"[ML-Brain] Save failed: {e}")

    def _try_load_model(self):
        if not self._model_path.exists():
            self._state.status = "OBSERVATION"
            return
        try:
            with open(self._model_path, "rb") as f:
                data = pickle.load(f)
            self._model = data.get("model")
            self._scaler = data.get("scaler")
            saved_state = data.get("state")
            self._last_train_count = data.get("last_train_count", 0)
            self._xgb_model = data.get("xgb_model")
            self._shap_explainer = None  # recomputed lazily
            loaded_names = data.get("feature_names")
            if loaded_names and len(loaded_names) != len(FEATURE_NAMES):
                logger.warning(
                    "[ML-Brain] Feature count mismatch (%d saved vs %d current). "
                    "Invalidating cached model for retrain.",
                    len(loaded_names), len(FEATURE_NAMES),
                )
                self._model = None
                self._scaler = None
                self._active_feature_names = list(FEATURE_NAMES)
                self._state.status = "OBSERVATION"
                return
            if loaded_names:
                self._active_feature_names = loaded_names
            loaded_params = data.get("best_params")
            if loaded_params:
                self._best_params = loaded_params

            if saved_state and self._model is not None:
                self._state = saved_state
                self._state.status = "READY"
                logger.info(
                    f"[ML-Brain] Loaded from disk | "
                    + f"{self._state.n_samples} samples | "
                    + f"CV: {self._state.accuracy_cv:.1%} | "
                    + f"params: {self._best_params}"
                )
            else:
                self._state.status = "OBSERVATION"
        except Exception as e:
            logger.warning(f"[ML-Brain] Could not load saved model: {e}")
            self._state.status = "OBSERVATION"

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _count_labeled_trades(self) -> int:
        try:
            from database.db_manager import get_connection
            with get_connection() as conn:
                r = conn.execute(
                    "SELECT COUNT(*) FROM market_features "
                    "WHERE trade_ticket IS NOT NULL "
                    "AND label_pips_pnl IS NOT NULL"
                ).fetchone()
            return int(r[0]) if r else 0
        except Exception:
            return 0


# Module-level singleton
win_model = WinProbabilityModel()
