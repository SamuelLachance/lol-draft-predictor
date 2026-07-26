"use client";

import { useEffect, useMemo, useState } from "react";
import { DraftBoard } from "@/components/DraftBoard";
import { PredictionPanel } from "@/components/PredictionPanel";
import { getDdragonVersion } from "@/lib/ddragon";
import {
  draftComplete,
  predictBlueWin,
  type DraftSides,
  type ModelBundle,
  type Role,
} from "@/lib/predict";

const emptyDraft = (roles: Role[]): DraftSides => ({
  blue: Object.fromEntries(roles.map((r) => [r, ""])) as Record<Role, string>,
  red: Object.fromEntries(roles.map((r) => [r, ""])) as Record<Role, string>,
});

export default function Home() {
  const [model, setModel] = useState<ModelBundle | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [ddragonVersion, setDdragonVersion] = useState("15.1.1");
  const [draft, setDraft] = useState<DraftSides>(emptyDraft(["top", "jng", "mid", "bot", "sup"]));

  useEffect(() => {
    Promise.all([
      fetch("model/draft_model.json").then((r) => {
        if (!r.ok) throw new Error("Modèle introuvable");
        return r.json();
      }),
      getDdragonVersion().catch(() => "15.1.1"),
    ])
      .then(([bundle, version]) => {
        setModel(bundle);
        setDdragonVersion(version);
        setDraft(emptyDraft(bundle.roles));
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const prediction = useMemo(() => {
    if (!model || !draftComplete(draft, model.roles)) return null;
    return predictBlueWin(model, draft);
  }, [model, draft]);

  function randomize() {
    if (!model) return;
    const pool = [...model.champions];
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    const next = emptyDraft(model.roles);
    model.roles.forEach((role, i) => {
      next.blue[role] = pool[i];
      next.red[role] = pool[i + 5];
    });
    setDraft(next);
  }

  function reset() {
    if (!model) return;
    setDraft(emptyDraft(model.roles));
  }

  return (
    <main className="relative mx-auto flex w-full max-w-7xl flex-1 flex-col px-4 py-8 md:px-8 md:py-12">
      <header className="rise mb-8 md:mb-10">
        <p className="mb-3 text-xs uppercase tracking-[0.28em] text-[var(--gold)]">
          League of Legends · Pro 5v5
        </p>
        <h1 className="font-[family-name:var(--font-display)] text-5xl leading-none tracking-wide md:text-7xl">
          DraftSight
        </h1>
        <p className="mt-4 max-w-2xl text-base text-[var(--muted)] md:text-lg">
          Qui gagne selon la draft ? Modèle entraîné sur l&apos;historique pro — toutes ligues,
          toutes divisions.
        </p>
        <div className="mt-6 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={randomize}
            className="rounded-full px-5 py-2.5 text-sm font-medium text-black transition hover:brightness-110"
            style={{ background: "linear-gradient(90deg, #d7b56d, #f0d79a)" }}
          >
            Draft aléatoire
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded-full border px-5 py-2.5 text-sm text-[var(--ink)] transition hover:bg-white/5"
            style={{ borderColor: "var(--line)" }}
          >
            Réinitialiser
          </button>
        </div>
      </header>

      {error ? (
        <p className="rounded-xl border border-red-400/40 bg-red-500/10 p-4 text-red-200">
          {error}. Lance <code>python ml/train.py</code> puis rebuild le site.
        </p>
      ) : null}

      {!model ? (
        <p className="text-[var(--muted)]">Chargement du modèle...</p>
      ) : (
        <div className="space-y-6">
          <div className="rise rise-delay-1">
            <DraftBoard
              model={model}
              ddragonVersion={ddragonVersion}
              draft={draft}
              onChange={setDraft}
            />
          </div>
          <PredictionPanel
            ready={Boolean(prediction)}
            blueWinProb={prediction?.blueWinProb ?? 0.5}
            redWinProb={prediction?.redWinProb ?? 0.5}
            confidence={prediction?.confidence ?? 0}
            metrics={model.metrics}
          />
        </div>
      )}

      <footer className="mt-12 border-t pt-6 text-xs text-[var(--muted)]" style={{ borderColor: "var(--line)" }}>
        Données: Oracle&apos;s Elixir (2014–présent). La draft explique une partie seulement du
        résultat — l&apos;exécution en game reste décisive.
      </footer>
    </main>
  );
}
