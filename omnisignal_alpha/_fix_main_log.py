path = "main.py"
with open(path, encoding="utf-8") as f:
    text = f.read()

bad = (
    "    logger.info('[Main] Adaptive Optimizer: Active (Sharpe-maximizing parameter tuning)')"
    "\\n    logger.info('[Main] Kelly Engine: Active (Bayesian Kelly sizing + concave DD dampener)')"
)
good = (
    "    logger.info('[Main] Adaptive Optimizer: Active (Sharpe-maximizing parameter tuning)')\n"
    "    logger.info('[Main] Kelly Engine: Active (Bayesian Kelly sizing + concave DD dampener)')\n"
)

if bad not in text:
    raise SystemExit("bad fragment not found")
text = text.replace(bad, good, 1)

lines = text.splitlines(keepends=True)
kelly = "    logger.info('[Main] Kelly Engine: Active (Bayesian Kelly sizing + concave DD dampener)')\n"
out = []
skip_next_kelly_dupes = False
for i, line in enumerate(lines):
    if "Adaptive Optimizer: Active" in line and "\\n" not in line:
        out.append(line)
        skip_next_kelly_dupes = True
        continue
    if skip_next_kelly_dupes and line == kelly:
        out.append(line)
        skip_next_kelly_dupes = False
        continue
    if line == kelly and out and out[-1].strip() == kelly.strip():
        continue
    out.append(line)

with open(path, "w", encoding="utf-8", newline="") as f:
    f.write("".join(out))
print("OK")
