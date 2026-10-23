# DraftSight — LoL Pro Predictor (Worlds 2026)

Site live : https://samuellachance.github.io/lol-draft-predictor/

Prédictions, ratings d'équipes/joueurs, meta champions et **une page dédiée par match des Worlds** (probas, enjeux Swiss, champions contestés, draft attendue, plan de jeu de chaque équipe, duels de rôles, draft/résultats une fois joué, et le match en direct).

## Modèle (v3)

`p(blue gagne) = σ( w0 + w_bt·bt + w_ts·ts + w_draft·draft )`

| Brique | Rôle | Fichier |
|---|---|---|
| `bt` | Force d'équipe **inter-ligues** : Bradley-Terry ridge, `rating = décalage de ligue + écart d'équipe`, pondéré par la récence | `ml/strength.py` |
| `ts` | TrueSkill des 5 joueurs du roster | `ml/build_board.py` |
| `draft` | Valeur champion×rôle, matchups de lane, synergies (ridge logistique antisymétrique, décroissance temporelle) | `ml/draftmodel.py` |

Les décalages de ligue ne sont appris que sur les matchs inter-ligues (Worlds/MSI/FST/EWC) : une équipe de division inférieure (TCL, LFL, EM, académies…) ne peut plus passer devant une équipe de ligue majeure. Seules les ligues majeures actives sont classées.

**Validation honnête (walk-forward, refit mensuel, aucune fuite)** — mêmes ~2 900 matchs majeurs/internationaux depuis 09/2025 :

| Modèle | Log-loss | Précision |
|---|---|---|
| Ancien Elo séquentiel | 0.657 | 61.5 % |
| BT inter-ligues | 0.644 | 63.7 % |
| + TrueSkill roster | 0.641 | 63.7 % |
| + draft (modèle final) | **0.639** | **63.8 %** |

La draft seule vaut ~54 % : elle n'apporte qu'un petit ajustement (−0.002 de log-loss) au-dessus de la force d'équipe. Probabilités bien calibrées (voir `ml/eval_strength.py`, `ml/eval_draft.py`). Les hyperparamètres (demi-vie 180 j, ridge…) viennent d'une grille walk-forward.

## Pipeline

```bash
python -m pip install -r ml/requirements.txt trueskill
python ml/fetch_leaguepedia.py --since 2026-10-01   # joueurs/équipes (ratings)  — source fraîche du jour
python ml/fetch_rich.py --since 2025-01-01          # picks, bans, objectifs + calendrier Worlds
python ml/build_drafts.py && python ml/build_drafts_team.py
python ml/build_board.py                            # ratings, modèle, teams/players/champions/meta.json  (~5 min)
python ml/build_worlds.py                           # pages de match, simulation Swiss, meta  (~10 s)
```

`build_worlds.py` ne dépend que de fichiers déjà publiés (+ `games_rich.parquet`) : il est donc rejoué toutes les 10 min par le job « live ».

## Mise à jour en temps réel

1. **Navigateur → API lolesports** (officielle, CORS ouvert) : statuts/scores toutes les 30 s ; pendant un match, l'état du game (or, kills, tours, objectifs, KDA des joueurs) toutes les 10 s (`web/public/common.js`).
2. **`.github/workflows/live-update.yml`** (toutes les ~10 min) : relit Leaguepedia, applique un ajustement de rating « en tournoi » (+0.10 logit × résultat surprise par game Worlds jouée), republie `worlds.json` + `live.json`. Le navigateur recharge dès que `live.json` change.
3. **`.github/workflows/daily-update.yml`** (13:00 UTC) : rebuild complet (ratings, modèle, draft), déploiement `gh-pages`.

Le workflow quotidien publie aussi l'asset `games_rich.parquet` de la Release `board-data`, dont dépend le job live (premier run = backfill ~3 min).

En local (IP résidentielle) : `update_board.ps1` fait la même chose et déploie.

## Site local

```bash
cd web/public && python -m http.server 8765      # site statique : index.html, match/, draft/
```

(`web/src` = ancienne appli Next.js, non utilisée par le site publié.)

## Limites assumées

- Pas de flux de draft *pendant* la sélection : l'API officielle n'expose les picks qu'une fois le game lancé.
- Appariements Swiss futurs simulés au hasard au sein d'un même bilan (hors revanche) ; bracket des quarts tiré au hasard.
- « Picks probables » = pool récent du joueur ; l'effet des bans/counters adverses n'est pas modélisé.
- Analyse esport, pas un conseil de pari.
