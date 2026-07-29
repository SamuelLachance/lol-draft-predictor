"""Genere les donnees de l'app de draft (web/public/draft/data/champions.json).

Contenu : les 172 champions avec leurs attributs (l'adversaire a un pool INFINI)
+ le champ pool de NOTRE equipe par role (team/champ-pool.md).
"""
from __future__ import annotations
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # draftwork/
LOL = ROOT.parent                                    # lolcoach/
SRC = ROOT / "data" / "processed" / "champions_db.csv"
OUT = ROOT / "web" / "public" / "draft" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# Champ pool de l'equipe (source: tier list drafting.gg, confirme par l'utilisateur)
POOL = {
    "top": ["Sion", "Gnar", "Renekton", "K'Sante", "Gragas", "Garen", "Jax"],
    "jng": ["Maokai", "Jarvan IV", "Nasus", "Poppy", "Nunu & Willump", "Sejuani",
            "Trundle", "Naafiri", "Skarner", "Cho'Gath"],
    "mid": ["Mel", "Orianna", "Galio", "Veigar", "Smolder", "Ziggs", "Viktor", "Syndra"],
    "bot": ["Ezreal", "Sivir", "Ashe", "Xayah", "Tristana", "Brand", "Lucian", "Karthus"],
    "sup": ["Karma", "Bard", "Seraphine", "Rakan", "Janna", "Leona", "Braum", "Nautilus"],
}
NUM = ["cc", "engage", "disengage", "mobility", "waveclear", "poke", "scaling",
       "tankiness", "sustain", "burst", "global", "anti_dive", "pick"]


def main() -> None:
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    by_name = {r["champion"]: r for r in rows}

    missing = [c for cs in POOL.values() for c in cs if c not in by_name]
    if missing:
        raise SystemExit(f"Champions du pool absents de la base : {missing}")

    champs = []
    for r in rows:
        c = {"name": r["champion"], "roles": r["roles"].split("|"),
             "damage": r["damage"], "range": r["range_type"], "family": r["family"],
             "teamfight": r["teamfight"], "desc": r["description"]}
        for k in NUM:
            c[k] = int(r[k])
        champs.append(c)
    champs.sort(key=lambda c: c["name"])

    data = {"champions": champs, "pool": POOL,
            "roles": ["top", "jng", "mid", "bot", "sup"],
            "role_labels": {"top": "Top", "jng": "Jungle", "mid": "Mid",
                            "bot": "Bot", "sup": "Support"}}
    (OUT / "champions.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    n_pool = sum(len(v) for v in POOL.values())
    print(f"champions.json : {len(champs)} champions | pool equipe {n_pool} "
          f"({', '.join(f'{k}={len(v)}' for k, v in POOL.items())})")


if __name__ == "__main__":
    main()
