"""Verifie la syntaxe JS des pages statiques (scripts inline + common.js) avec node. Usage: python scripts_check.py"""
import json, re, subprocess, sys, tempfile, pathlib
root = pathlib.Path(__file__).parent / "web" / "public"
bad = 0
for f in [root / "index.html", root / "match/index.html", root / "common.js"]:
    src = f.read_text(encoding="utf-8")
    parts = [src] if f.suffix == ".js" else re.findall(r"<script>(.*?)</script>", src, re.S)
    for i, js in enumerate(parts):
        tmp = pathlib.Path(tempfile.gettempdir()) / f"chk_{f.stem}_{i}.js"; tmp.write_text(js, encoding="utf-8")
        r = subprocess.run(["node", "--check", str(tmp)], capture_output=True, text=True, encoding='utf-8', errors='replace')
        if r.returncode:
            bad += 1; print("SYNTAX ERROR", f, "\n", (r.stderr or '')[:600])
def _no_const(c): raise ValueError("constante invalide en JSON : " + c)
for f in sorted(list(root.glob("data/*.json")) + list(root.glob("model/*.json"))):
    try: json.loads(f.read_text(encoding="utf-8"), parse_constant=_no_const)
    except ValueError as e:
        bad += 1; print("JSON INVALIDE", f, e)
print("OK" if not bad else f"{bad} erreur(s)"); sys.exit(1 if bad else 0)
