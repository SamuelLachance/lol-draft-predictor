# DraftSight — LoL Pro Predictor (Worlds 2026)

Site live : https://samuellachance.github.io/lol-draft-predictor/

Prédictions, ratings d'équipes/joueurs, meta champions et **une page dédiée par match des Worlds** (probas, enjeux Swiss, champions contestés, draft attendue, plan de jeu de chaque équipe, duels de rôles, draft/résultats une fois joué, et le match en direct).

## Modèle (v4)

`p(blue gagne) = σ( w0 + w_bt·bt + w_pbt·pbt + w_ts·ts + w_fw·Δforme + w_h2h·h2h [+ w_draft·draft + w_c·maîtrise] )`

| Signal | Rôle | Fichier |
|---|---|---|
| `bt` | Force d'équipe **inter-ligues** : Bradley-Terry ridge, `niveau = décalage de ligue + écart d'équipe`, pondéré par la récence | `ml/strength.py` |
| `pbt` | Bradley-Terry au niveau **joueur** (somme des 5 joueurs) : réagit tout de suite aux changements de roster | `ml/strength.py` |
| `ts` | TrueSkill des 5 joueurs | `ml/features.py` |
| forme, h2h | forme pondérée (10 derniers matchs), confrontations directes sur 2 ans | `ml/features.py` |
| `draft`, maîtrise | valeur champion×rôle + matchups de lane + synergies ; historique joueur×champion (winrate shrinké vers la meta) | `ml/draftmodel.py`, `ml/features.py` |

Les décalages de ligue ne sont appris que sur les matchs inter-ligues : une équipe de division inférieure (TCL, LFL, EM, académies…) ne peut plus passer devant une équipe de ligue majeure. Seules les ligues majeures actives sont classées.

**Validation honnête** — walk-forward (refit mensuel, aucune fuite), 2 783 matchs officiels des ligues majeures depuis 09/2025 :

| Modèle | Log-loss | Précision |
|---|---|---|
| Ancien Elo séquentiel | 0.652 | 62.2 % |
| Bradley-Terry d'équipe seul | 0.639 | 64.2 % |
| v3 (équipe + TrueSkill + draft) | 0.633 | 64.8 % |
| **v4 avant draft** (pour les matchs à venir) | **0.632** | **64.5 %** |
| **v4 après draft** (games dont on connaît la draft) | **0.628** | **65.2 %** |

Probabilités bien calibrées à tous les niveaux de confiance (60 % annoncé ⇒ ~60 % observé, 80-90 % ⇒ ~84 %).

### Pourquoi pas 75-80 % ?
Un match pro est très bruité : avec uniquement l'information d'avant-match, ~65 % est proche du plafond. Je l'ai testé : un LightGBM sur ~60 variables causales (meta glissante par champion×rôle, attributs de composition — engage, CC, mobilité, poke, scaling, frontline, AD/AP —, repos/charge, série, TrueSkill par rôle, Elo séquentiel…) fait **moins bien** que le stack linéaire ci-dessus (0.636 vs 0.628) ; ces variables n'apportent rien une fois la force d'équipe connue (`ml/eval_v4.py`, `ml/exp_stack.py`). La draft seule vaut ~54 %. Les probabilités qui dépassent nettement 70 % ne viennent donc que d'écarts de niveau réels entre équipes. La sélection des variables s'est faite sur la même période de test : prévoir ~0.1-0.2 pt de moins en conditions réelles.

## Approches « valeur cachée » testées (RAPM, embeddings, …)

Test hors échantillon (walk-forward, 7 329 matchs majeurs 2024-2026, refit trimestriel ; `ml/exp_rapm.py`, `ml/eval_rapm.py`) de méthodes inspirées du basket (RAPM) et des GNN :
- **RAPM / modèle hiérarchique** (`ml/rapm.py`) : ridge logistique `y ~ force d'équipe + niveau des 10 joueurs + champion×rôle + champion×maîtrise` (maîtrise = games déjà jouées par ce joueur sur ce champion) → effet *net* du champion. Seul, il estime mieux la valeur de draft que notre ancien modèle (log-loss à force d'équipe égale **0,6276 vs 0,6306**, précision 64,6 % vs 63,7 %). Ajouté au stack complet : gain réel mais minime (−0,0010 [−0,0019 ; −0,0001]), car la maîtrise et le niveau des joueurs y sont déjà captés → **non intégré au stack** (complexité > gain), mais **utilisé pour l'onglet Meta → « 💎 Valeur cachée »** : effet ajusté vs winrate brut, courbe d'apprentissage (gain après 15 games), présence. Exemple : Nautilus support 43,8 % de winrate brut sur 981 games, mais effet ajusté ≈ neutre/positif — le winrate sous-estime le champion.
- **Embeddings de champions + synergies/matchups latents** (réseau torch, équivalent léger d'un GNN sur le graphe de composition) : **aucun gain** (coefficient ~0, surajustement).
- **Niveau joueur du RAPM** : n'apporte rien au-delà du Bradley-Terry par joueur déjà présent.
- **Non testables ici** : chaînes de Markov / analyse de survie sur les objectifs (pas de données en cours de partie), RL et CFR/équilibre de Nash (pas de simulateur du jeu).

## Études spécifiques aux Worlds (résultats négatifs, documentés)

Toutes reproductibles (`ml/fetch_history.py` puis `ml/study_scaling.py`, `ml/study_worlds.py`, `ml/study_worlds2.py`, `ml/study_intl.py`) sur 42 000 games pro 2014-2026 avec durée et compositions.

**Théorie « aux Worlds le scaling gagne beaucoup plus souvent »** — verdict : **non démontrée, et sans gain pour le modèle.**
- Les games de Worlds ne sont pas plus longues : écart moyen −0,02 min vs LCK/LPL/LEC/LCS la même année (plus longues 5 éditions sur 10). Les équipes ne draftent pas plus de scaling aux Worlds (écart de score de scaling : +0,002).
- Indice de scaling *mesuré* (effet de la durée sur le winrate de chaque champion, ridge, à force d'équipe égale, estimé hors des games testées) : valide (l'équipe la plus « scaling » gagne 44 % des games < 28 min et 53 % des > 38 min en domestique) mais **instable d'une époque à l'autre** (corrélation 0,06 entre ≤2021 et ≥2022). Effet aux Worlds +0,075 logit/écart-type [−0,05 ; +0,24] contre −0,02 en domestique : supplément non significatif.
- Avec l'attribut « scaling » (0-3) écrit à la main : effet significatif en rétrospectif (Worlds +0,14/écart-type [+0,01 ; +0,32] ; 2021-2025 : +0,32 [+0,10 ; +0,51] ; domestique ≈ 0 ; l'équipe la plus « scaling » gagne 53,8 % ±4,4 aux Worlds contre 49,4 % en domestique). Mais l'effet change de signe selon l'édition (négatif 2016-2018, positif 2019-2025 sauf 2024) et 7 sous-ensembles ont été regardés : à prendre comme un indice, pas une preuve.
- **Test hors échantillon** (9 éditions, 726 games, entraînement uniquement sur le passé) : ajouter le scaling (attribut, hypercarry, indice mesuré ; seul ou × Worlds) ne change pas le log-loss (−0,0007 [−0,005 ; +0,003] au mieux). Rien n'a été ajouté au modèle.

**Autres diagnostics Worlds** : côté bleu 54,0 % aux Worlds (55,3 % sur 2021-2025) vs 53,3 % en domestique (déjà capté par le biais de side) ; le favori gagne 65,8 % aux Worlds vs 62,7 % en domestique. Une pente de calibration < 1 (favoris surestimés) apparaît sur MSI/play-ins avec l'ancien Bradley-Terry seul, mais **disparaît dans le stack de production** (ajouter un terme « × international » dégrade de 0,01 de log-loss) : pas de correction spécifique.

## Pipeline

```bash
python -m pip install -r ml/requirements.txt trueskill
python ml/fetch_leaguepedia.py --since 2026-10-01   # joueurs/équipes (ratings)  — source fraîche du jour
python ml/fetch_rich.py --since 2025-01-01          # picks, bans, objectifs + calendrier Worlds
python ml/build_drafts.py && python ml/build_drafts_team.py
python ml/build_board.py                            # ratings, modèle v4, teams/players/champions/meta/h2h.json  (~20 min)
python ml/build_worlds.py                           # pages de match, simulation Swiss, meta  (~10 s)
```

`build_worlds.py` ne dépend que de fichiers déjà publiés (+ `games_rich.parquet`) : il est donc rejoué toutes les 10 min par le job « live ».

## Mise à jour en temps réel

1. **Navigateur → API lolesports** (officielle, CORS ouvert) : statuts/scores toutes les 30 s ; pendant un match, l'état du game (or, kills, tours, objectifs, KDA des joueurs) toutes les 10 s (`web/public/common.js`).
2. **`.github/workflows/live-update.yml`** (toutes les ~10 min) : relit Leaguepedia, ajoute les games du jour et **refait l'ajustement complet** de la force d'équipe, du niveau des joueurs, de la forme et des h2h (`ml/refresh_ratings.py`, ~40 s, mêmes fonctions que le build quotidien), puis republie `worlds.json` + `live.json`. Le navigateur recharge dès que `live.json` change.
3. **Simulateur dans le navigateur** (`web/public/sim.js`) : dès qu'un résultat officiel arrive par l'API lolesports (avant le serveur), bilans, chances de qualification/titre et enjeux victoire-défaite sont recalculés localement avec les mêmes règles et le même modèle.
4. **`.github/workflows/daily-update.yml`** (13:00 UTC) : rebuild complet (ratings, modèle, poids du stack, draft), déploiement `gh-pages`.

Le workflow quotidien publie aussi l'asset `games_rich.parquet` de la Release `board-data`, dont dépend le job live (premier run = backfill ~3 min).

En local (IP résidentielle) : `update_board.ps1` fait la même chose et déploie.

## Site local

```bash
cd web/public && python -m http.server 8765      # site statique : index.html, match/, draft/
```

(`web/src` = ancienne appli Next.js, non utilisée par le site publié.)

## Limites assumées

- Pas de flux de draft *pendant* la sélection : l'API officielle n'expose les picks qu'une fois le game lancé.
- Swiss simulé avec les **règles officielles Riot** : R1 connu, puis tirage au hasard parmi les équipes du même bilan sans revanche ; quarts : les 3-0 contre des 3-2 tirés au sort, les 4 autres au hasard ; demi-finales : appariement tiré au hasard (règle non publiée).
- Les Bo5 d'élimination utilisent la proba d'un game répétée (pas de modélisation du fearless) ; les cotes de titre des gros favoris sont donc à lire comme des ordres de grandeur.
- « Picks probables » = pool récent du joueur ; l'effet des bans/counters adverses n'est pas modélisé.
- Analyse esport, pas un conseil de pari.
