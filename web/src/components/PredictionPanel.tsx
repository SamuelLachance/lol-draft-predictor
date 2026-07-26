"use client";

type Props = {
  ready: boolean;
  blueWinProb: number;
  redWinProb: number;
  confidence: number;
  metrics?: {
    accuracy: number;
    accuracy_confident?: number | null;
    roc_auc: number;
    n_games_total: number;
  };
};

export function PredictionPanel({
  ready,
  blueWinProb,
  redWinProb,
  confidence,
  metrics,
}: Props) {
  const bluePct = Math.round(blueWinProb * 1000) / 10;
  const redPct = Math.round(redWinProb * 1000) / 10;
  const confPct = Math.round(confidence * 100);

  return (
    <section
      className="rise rise-delay-2 rounded-2xl border p-5 md:p-7"
      style={{ borderColor: "var(--line)", background: "var(--panel)" }}
    >
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="font-[family-name:var(--font-display)] text-2xl tracking-wide">
            Prédiction
          </h2>
          <p className="text-sm text-[var(--muted)]">
            Probabilité de victoire selon la draft seule (pro 5v5)
          </p>
        </div>
        {metrics ? (
          <div className="text-right text-xs text-[var(--muted)]">
            <div>
              Holdout: {(metrics.accuracy * 100).toFixed(1)}% · AUC{" "}
              {metrics.roc_auc.toFixed(3)}
            </div>
            {metrics.accuracy_confident ? (
              <div>
                Quand confiant: {(metrics.accuracy_confident * 100).toFixed(1)}%
              </div>
            ) : null}
            <div>{metrics.n_games_total.toLocaleString("fr-FR")} drafts pro</div>
          </div>
        ) : null}
      </div>

      {!ready ? (
        <p className="text-[var(--muted)]">Complète les 10 picks pour lancer la prédiction.</p>
      ) : (
        <>
          <div className="mb-3 flex items-center justify-between font-[family-name:var(--font-display)] text-3xl md:text-5xl">
            <span style={{ color: "var(--blue-glow)" }}>{bluePct}%</span>
            <span className="text-base tracking-[0.2em] text-[var(--muted)] md:text-lg">VS</span>
            <span style={{ color: "var(--red-glow)" }}>{redPct}%</span>
          </div>
          <div
            className="prob-bar mb-4 flex h-4 overflow-hidden rounded-full border"
            style={{ borderColor: "var(--line)" }}
          >
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${bluePct}%`,
                background: "linear-gradient(90deg, #1d5fd0, #4ea1ff)",
              }}
            />
            <div
              className="h-full transition-all duration-500"
              style={{
                width: `${redPct}%`,
                background: "linear-gradient(90deg, #ff6b76, #c42230)",
              }}
            />
          </div>
          <div className="flex flex-wrap gap-4 text-sm text-[var(--muted)]">
            <span>
              Favori:{" "}
              <strong className="text-[var(--ink)]">
                {blueWinProb >= redWinProb ? "Blue Side" : "Red Side"}
              </strong>
            </span>
            <span>
              Confiance modèle: <strong className="text-[var(--ink)]">{confPct}%</strong>
            </span>
          </div>
        </>
      )}
    </section>
  );
}
