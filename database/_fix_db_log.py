from pathlib import Path

p = Path(__file__).resolve().parent / "db_manager.py"
t = p.read_text(encoding="utf-8")
old = '    logger.info("[DB] v2.0 schema initialized.")  # placeholder'
new = '    logger.info("[DB] v2.0 schema initialized.")'
if old in t:
    p.write_text(t.replace(old, new, 1), encoding="utf-8")
    print("Fixed logger line")
else:
    print("Logger line already OK or different")
