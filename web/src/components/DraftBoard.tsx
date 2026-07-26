"use client";

import { useMemo, useState } from "react";
import type { DraftSides, ModelBundle, Role } from "@/lib/predict";
import { championIconUrlSync } from "@/lib/ddragon";

const ROLE_LABEL: Record<Role, string> = {
  top: "Top",
  jng: "Jungle",
  mid: "Mid",
  bot: "Bot",
  sup: "Support",
};

type Props = {
  model: ModelBundle;
  ddragonVersion: string;
  draft: DraftSides;
  onChange: (draft: DraftSides) => void;
};

export function DraftBoard({ model, ddragonVersion, draft, onChange }: Props) {
  const [query, setQuery] = useState("");
  const [active, setActive] = useState<{ side: "blue" | "red"; role: Role }>({
    side: "blue",
    role: "top",
  });

  const picked = useMemo(() => {
    const set = new Set<string>();
    for (const role of model.roles) {
      if (draft.blue[role]) set.add(draft.blue[role]);
      if (draft.red[role]) set.add(draft.red[role]);
    }
    return set;
  }, [draft, model.roles]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return model.champions
      .filter((c) => !q || c.toLowerCase().includes(q))
      .slice(0, 80);
  }, [model.champions, query]);

  function pick(champ: string) {
    if (picked.has(champ) && draft[active.side][active.role] !== champ) return;
    const next: DraftSides = {
      blue: { ...draft.blue },
      red: { ...draft.red },
    };
    next[active.side][active.role] = champ;
    onChange(next);

    // auto-advance slot
    const order: Array<{ side: "blue" | "red"; role: Role }> = [];
    for (const role of model.roles) order.push({ side: "blue", role });
    for (const role of model.roles) order.push({ side: "red", role });
    const idx = order.findIndex(
      (s) => s.side === active.side && s.role === active.role,
    );
    for (let i = 1; i <= order.length; i++) {
      const slot = order[(idx + i) % order.length];
      if (!next[slot.side][slot.role]) {
        setActive(slot);
        break;
      }
    }
  }

  function clearSlot(side: "blue" | "red", role: Role) {
    const next: DraftSides = {
      blue: { ...draft.blue },
      red: { ...draft.red },
    };
    next[side][role] = "";
    onChange(next);
    setActive({ side, role });
  }

  function Slot({
    side,
    role,
  }: {
    side: "blue" | "red";
    role: Role;
  }) {
    const champ = draft[side][role];
    const selected = active.side === side && active.role === role;
    const accent = side === "blue" ? "var(--blue)" : "var(--red)";
    return (
      <button
        type="button"
        onClick={() => setActive({ side, role })}
        className="group relative flex w-full items-center gap-3 rounded-xl border px-3 py-2 text-left transition"
        style={{
          borderColor: selected ? accent : "var(--line)",
          background: selected ? "rgba(255,255,255,0.04)" : "transparent",
          boxShadow: selected ? `0 0 0 1px ${accent}` : undefined,
        }}
      >
        <div
          className="flex h-12 w-12 shrink-0 items-center justify-center overflow-hidden rounded-lg border"
          style={{ borderColor: "var(--line)", background: "rgba(0,0,0,0.25)" }}
        >
          {champ ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img
              src={championIconUrlSync(champ, ddragonVersion)}
              alt={champ}
              className="h-full w-full object-cover"
            />
          ) : (
            <span className="text-xs text-[var(--muted)]">{ROLE_LABEL[role][0]}</span>
          )}
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-[11px] uppercase tracking-[0.16em] text-[var(--muted)]">
            {ROLE_LABEL[role]}
          </div>
          <div className="truncate font-[family-name:var(--font-display)] text-base">
            {champ || "Choisir"}
          </div>
        </div>
        {champ ? (
          <span
            role="button"
            tabIndex={0}
            onClick={(e) => {
              e.stopPropagation();
              clearSlot(side, role);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.stopPropagation();
                clearSlot(side, role);
              }
            }}
            className="rounded px-2 py-1 text-xs text-[var(--muted)] opacity-0 transition group-hover:opacity-100 hover:text-white"
          >
            ✕
          </span>
        ) : null}
      </button>
    );
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr_1.1fr_1fr]">
      <section className="space-y-2 rounded-2xl border p-4" style={{ borderColor: "var(--line)", background: "var(--panel)" }}>
        <h2 className="mb-3 font-[family-name:var(--font-display)] text-xl tracking-wide text-[var(--blue-glow)]">
          Blue Side
        </h2>
        {model.roles.map((role) => (
          <Slot key={`blue-${role}`} side="blue" role={role} />
        ))}
      </section>

      <section className="rounded-2xl border p-4" style={{ borderColor: "var(--line)", background: "var(--panel)" }}>
        <div className="mb-3 flex items-end justify-between gap-3">
          <div>
            <h2 className="font-[family-name:var(--font-display)] text-xl tracking-wide">
              Champions
            </h2>
            <p className="text-sm text-[var(--muted)]">
              Slot actif: {active.side === "blue" ? "Blue" : "Red"} · {ROLE_LABEL[active.role]}
            </p>
          </div>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Rechercher..."
            className="w-44 rounded-lg border bg-black/20 px-3 py-2 text-sm outline-none focus:border-[var(--gold)]"
            style={{ borderColor: "var(--line)" }}
          />
        </div>
        <div className="grid max-h-[420px] grid-cols-4 gap-2 overflow-y-auto pr-1 sm:grid-cols-5">
          {filtered.map((champ) => {
            const disabled = picked.has(champ);
            return (
              <button
                key={champ}
                type="button"
                disabled={disabled}
                onClick={() => pick(champ)}
                className="rounded-xl border p-1.5 text-center transition hover:-translate-y-0.5 disabled:cursor-not-allowed disabled:opacity-30"
                style={{ borderColor: "var(--line)", background: "rgba(0,0,0,0.2)" }}
                title={champ}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={championIconUrlSync(champ, ddragonVersion)}
                  alt={champ}
                  className="mx-auto mb-1 h-12 w-12 rounded-lg object-cover"
                />
                <div className="truncate text-[10px] text-[var(--muted)]">{champ}</div>
              </button>
            );
          })}
        </div>
      </section>

      <section className="space-y-2 rounded-2xl border p-4" style={{ borderColor: "var(--line)", background: "var(--panel)" }}>
        <h2 className="mb-3 font-[family-name:var(--font-display)] text-xl tracking-wide text-[var(--red-glow)]">
          Red Side
        </h2>
        {model.roles.map((role) => (
          <Slot key={`red-${role}`} side="red" role={role} />
        ))}
      </section>
    </div>
  );
}
