from pathlib import Path

p = Path("quant/catcd_engine.py")
t = p.read_text(encoding="utf-8")
t = t.replace("z_threshold: float = 2.0,", "z_threshold: float = 2.5,", 1)
t = t.replace("if len(self._ref_correlations) < 30:", "if len(self._ref_correlations) < 100:", 1)
t = t.replace(
    "        sl_dist = max(1.0 * atr, 25 * self._pip_size)\n"
    "        sl_dist = min(sl_dist, 2.0 * atr)",
    "        sl_dist = max(0.8 * atr, 20 * self._pip_size)\n"
    "        sl_dist = min(sl_dist, 1.5 * atr)",
    1,
)
t = t.replace(
    "        tp_dist = max(sl_dist * 1.5, 15 * self._pip_size)",
    "        tp_dist = max(sl_dist * 2.0, 20 * self._pip_size)",
    1,
)
old = """                direction = self._determine_direction(gold_ticks, dxy_ticks, z_score)
                if direction is None:
                    continue

                await self._generate_signal(direction, z_score, corr)"""
new = """                direction = self._determine_direction(gold_ticks, dxy_ticks, z_score)
                if direction is None:
                    continue

                # v4.4: CVD confirmation -- verify flow supports the signal direction
                gold_ret_dir = self._compute_tick_returns(gold_ticks)
                if len(gold_ret_dir) >= 20:
                    cvd = np.cumsum(gold_ret_dir)
                    half = len(cvd) // 2
                    cvd_slope = float(np.mean(cvd[half:]) - np.mean(cvd[:half]))
                    if direction == "BUY" and cvd_slope < -0.0001:
                        logger.debug("[CATCD] CVD opposes BUY direction, skipping")
                        continue
                    if direction == "SELL" and cvd_slope > 0.0001:
                        logger.debug("[CATCD] CVD opposes SELL direction, skipping")
                        continue

                await self._generate_signal(direction, z_score, corr)"""
if old not in t:
    raise SystemExit("catcd block not found")
t = t.replace(old, new, 1)
p.write_text(t, encoding="utf-8")
print("catcd_engine.py OK")

ce = Path("quant/convergence_engine.py")
ct = ce.read_text(encoding="utf-8")
ct = ct.replace('"CATCD":     0.10,', '"CATCD":     0.22,', 1)
ce.write_text(ct, encoding="utf-8")
print("convergence_engine.py OK")
