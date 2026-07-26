export type Role = "top" | "jng" | "mid" | "bot" | "sup";

export type DraftSides = {
  blue: Record<Role, string>;
  red: Record<Role, string>;
};

export type ModelBundle = {
  version: number;
  mode?: string;
  metrics: {
    accuracy: number;
    baseline_accuracy?: number;
    accuracy_confident?: number | null;
    confident_coverage?: number;
    roc_auc: number;
    n_games_total: number;
    n_test: number;
    n_champions: number;
    best_iteration?: number;
  };
  champions: string[];
  champion_index: Record<string, number>;
  roles: Role[];
  feature_names: string[];
  stats: {
    champ_wr: Record<string, number>;
    matchup_wr: Record<string, number>;
    synergy_wr: Record<string, number>;
    patch_champ_wr?: Record<string, number>;
    duo_wr?: Record<string, number>;
  };
  lgbm_model: LgbmModel;
  best_iteration: number;
  calibrator?: { coef: number; intercept: number };
  default_year: number;
  default_patch: string;
};

type LgbmNode = {
  split_feature?: number;
  threshold?: number;
  decision_type?: string;
  default_left?: boolean;
  left_child?: number | LgbmNode;
  right_child?: number | LgbmNode;
  leaf_value?: number;
};

type LgbmTree = {
  tree_structure: LgbmNode;
};

type LgbmModel = {
  feature_names?: string[];
  tree_info: LgbmTree[];
};

function mean(xs: number[]) {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}

function pairKey(a: string, b: string) {
  return [a, b].sort().join("||");
}

function buildMultiHotFeatures(model: ModelBundle, draft: DraftSides): number[] {
  const k = model.champions.length;
  const x = new Array(k * 3 + 10 + 5).fill(0);
  const roles = model.roles;
  const blue = roles.map((r) => draft.blue[r]);
  const red = roles.map((r) => draft.red[r]);

  blue.forEach((champ, i) => {
    const idx = model.champion_index[champ];
    if (idx == null) return;
    x[idx] = 1;
    x[2 * k + idx] += 1;
    x[3 * k + i] = idx;
  });
  red.forEach((champ, i) => {
    const idx = model.champion_index[champ];
    if (idx == null) return;
    x[k + idx] = 1;
    x[2 * k + idx] -= 1;
    x[3 * k + 5 + i] = idx;
  });

  const champWr = model.stats.champ_wr;
  const matchups = model.stats.matchup_wr;
  const synergies = model.stats.synergy_wr;
  const bWr = mean(blue.map((c) => champWr[c] ?? 0.5));
  const rWr = mean(red.map((c) => champWr[c] ?? 0.5));
  const mu = mean(
    roles.map((role, i) => matchups[`${role}::${blue[i]}||${red[i]}`] ?? 0.5),
  );
  const syn = (team: string[]) => {
    const vals: number[] = [];
    for (let a = 0; a < 5; a++) {
      for (let b = a + 1; b < 5; b++) {
        vals.push(synergies[pairKey(team[a], team[b])] ?? 0.5);
      }
    }
    return mean(vals);
  };
  const base = k * 3 + 10;
  x[base + 0] = bWr - rWr;
  x[base + 1] = mu - 0.5;
  x[base + 2] = syn(blue) - syn(red);
  x[base + 3] = bWr;
  x[base + 4] = rWr;
  return x;
}

function walkTree(node: LgbmNode, x: number[]): number {
  if (node.leaf_value != null && node.left_child == null) {
    return node.leaf_value;
  }
  if (typeof node.left_child !== "object" || typeof node.right_child !== "object") {
    return node.leaf_value ?? 0;
  }
  const fi = node.split_feature ?? 0;
  const thr = node.threshold ?? 0;
  const v = x[fi];
  const missing = v == null || Number.isNaN(v);
  const goLeft = missing ? Boolean(node.default_left) : v <= thr;
  return walkTree((goLeft ? node.left_child : node.right_child) as LgbmNode, x);
}

function sigmoid(z: number) {
  if (z >= 0) {
    const e = Math.exp(-z);
    return 1 / (1 + e);
  }
  const e = Math.exp(z);
  return e / (1 + e);
}

export function predictBlueWin(
  model: ModelBundle,
  draft: DraftSides,
): { blueWinProb: number; redWinProb: number; confidence: number } {
  const x = buildMultiHotFeatures(model, draft);
  const trees = model.lgbm_model.tree_info;
  const n = Math.min(model.best_iteration || trees.length, trees.length);
  let raw = 0;
  for (let i = 0; i < n; i++) {
    raw += walkTree(trees[i].tree_structure, x);
  }
  let blueWinProb = sigmoid(raw);
  if (model.calibrator) {
    const z = model.calibrator.coef * blueWinProb + model.calibrator.intercept;
    blueWinProb = sigmoid(z);
  }
  return {
    blueWinProb,
    redWinProb: 1 - blueWinProb,
    confidence: Math.abs(blueWinProb - 0.5) * 2,
  };
}

export function draftComplete(draft: DraftSides, roles: Role[]) {
  return roles.every((r) => draft.blue[r] && draft.red[r]);
}
