# DraftSight — LoL Pro Draft Predictor

Site live: https://samuellachance.github.io/lol-draft-predictor/

Modèle ML qui estime la probabilité de victoire **Blue vs Red** à partir d’une draft pro 5v5.

## Données

- Source: [Oracle’s Elixir](https://oracleselixir.com/tools/downloads) (2014 → présent)
- **Toutes ligues / divisions** confondues (LCK, LPL, LEC, LCS, LDL, VCS, CBLOL, etc.)
- ~93k games pro avec drafts 5v5 complètes

## Stack

- `ml/` — fetch, construction des drafts, entraînement LightGBM
- `web/` — Next.js (export statique) + inférence navigateur
- Déploiement: GitHub Pages

## Entraîner le modèle

```bash
python -m pip install -r ml/requirements.txt
python ml/fetch_data.py          # télécharge tous les CSV OE
python ml/build_drafts.py        # 1 ligne = 1 draft 5v5
python ml/train.py               # entraîne + exporte web/public/model/draft_model.json
```

## Site local

```bash
cd web
npm install
npm run dev
```

## GitHub Pages

1. Crée le repo `lol-draft-predictor` et push `main`
2. Settings → Pages → Source: **GitHub Actions**
3. Le workflow `.github/workflows/deploy.yml` publie `web/out`

URL attendue: `https://SamuelLachance.github.io/lol-draft-predictor/`

## Notes

La draft seule ne détermine pas une game pro. Un bon modèle draft-only se situe généralement un peu au-dessus de 50% (souvent ~55–60% selon la période / le meta). Les métriques exactes sont dans `models/metrics.json` après entraînement.
