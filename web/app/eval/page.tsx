"use client";

import { useEffect, useState } from "react";
import { evalResults } from "@/lib/api";

// The eval dashboard: every run in eval/results with its 95% bootstrap interval, so a
// difference that sits inside the intervals reads as noise at a glance.

type CI = [number, number, number];
type Run = {
  run: string;
  split: string;
  n: number;
  commit: string;
  config: Record<string, unknown>;
  ci?: Record<string, CI>;
  usd?: number;
  judge?: { judge_usd: number };
};
type Results = { retrieval: Run[]; generation: Run[] };

const RETRIEVAL = ["recall@5", "recall@10", "mrr", "ndcg@10"];
const GENERATION = [
  ["correctness", "correctness"],
  ["faithfulness", "faithfulness"],
  ["citation_hit", "citation hit"],
  ["unanswerable_refused", "declines unanswerable"],
  ["false_refusal", "false refusal ↓"],
];

function Interval({ ci, best }: { ci?: CI; best?: number }) {
  if (!ci) return <td className="px-3 py-2 text-muted">—</td>;
  const [mean, lo, hi] = ci;
  return (
    <td className="px-3 py-2 text-right tabular-nums">
      <div className={mean === best ? "font-semibold" : ""}>{mean.toFixed(3)}</div>
      <div className="text-xs text-muted">
        [{lo.toFixed(3)}, {hi.toFixed(3)}]
      </div>
      <div className="relative mt-1 h-1 rounded bg-line">
        <div
          className="absolute h-1 rounded bg-accent/40"
          style={{ left: `${lo * 100}%`, width: `${Math.max((hi - lo) * 100, 1)}%` }}
        />
        <div className="absolute -top-0.5 h-2 w-0.5 bg-accent" style={{ left: `${mean * 100}%` }} />
      </div>
    </td>
  );
}

function describe(run: Run): string {
  const c = run.config as Record<string, unknown>;
  const retrieval = (c.retrieval as Record<string, unknown> | undefined) ?? c;
  const parts = [
    String(retrieval.method ?? "vector"),
    retrieval.rerank ? `rerank ${retrieval.rerank} over ${retrieval.candidates}` : "no rerank",
  ];
  if (c.query_mode && c.query_mode !== "original") parts.push(`query: ${c.query_mode}`);
  if (c.version_filter === false) parts.push("no version filter");
  if (c.model && String(c.model).startsWith("claude")) parts.unshift(String(c.model));
  else if (c.model && c.model !== "voyage-4") parts.unshift(`embed: ${c.model}`);
  if (c.index_name && c.index_name !== "heading-plain") parts.push(`index: ${c.index_name}`);
  if (c.sample) parts.push(`${c.sample}-item sample`);
  return parts.join(" · ");
}

function Table({ title, runs, metrics }: { title: string; runs: Run[]; metrics: string[][] }) {
  const best = Object.fromEntries(
    metrics.map(([m]) => [m, Math.max(...runs.map((r) => r.ci?.[m]?.[0] ?? -1))]),
  );
  return (
    <section className="space-y-2">
      <h2 className="text-lg font-semibold">{title}</h2>
      <div className="overflow-x-auto rounded-lg border border-line bg-surface">
        <table className="w-full text-sm">
          <thead className="border-b border-line text-left text-xs uppercase tracking-wider text-muted">
            <tr>
              <th className="px-3 py-2">run</th>
              <th className="px-3 py-2 text-right">n</th>
              {metrics.map(([m, label]) => (
                <th key={m} className="min-w-28 px-3 py-2 text-right">
                  {label}
                </th>
              ))}
              <th className="px-3 py-2 text-right">cost</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run} className="border-b border-line last:border-0 align-top">
                <td className="px-3 py-2">
                  <div className="font-mono text-xs">{r.run}</div>
                  <div className="text-xs text-muted">{describe(r)}</div>
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{r.n}</td>
                {metrics.map(([m]) => (
                  <Interval key={m} ci={r.ci?.[m]} best={m.startsWith("false") ? undefined : best[m]} />
                ))}
                <td className="px-3 py-2 text-right tabular-nums text-muted">
                  {r.usd !== undefined ? `$${(r.usd + (r.judge?.judge_usd ?? 0)).toFixed(2)}` : "$0"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function EvalPage() {
  const [results, setResults] = useState<Results | null>(null);
  const [error, setError] = useState("");
  const [split, setSplit] = useState<"dev" | "test">("dev");

  useEffect(() => {
    evalResults()
      .then(setResults)
      .catch((e) => setError(String(e.message)));
  }, []);

  const of = (runs: Run[]) => runs.filter((r) => r.split === split);

  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">Eval results</h1>
        <p className="max-w-3xl text-sm text-muted">
          Each cell is the mean over the split with its 95% bootstrap interval (bar). Answer
          quality is scored by a calibrated Claude judge (rubric v2); retrieval metrics need no
          LLM. Dev is where settings were chosen; the test split is held out and scored once per
          milestone.
        </p>
        <div className="flex gap-1 text-sm">
          {(["dev", "test"] as const).map((s) => (
            <button
              key={s}
              onClick={() => setSplit(s)}
              className={`rounded px-3 py-1 ${split === s ? "bg-accent text-white" : "bg-surface border border-line"}`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      {error && <div className="rounded-md bg-warn-bg p-3 text-sm text-warn-text">{error}</div>}
      {!results && !error && <div className="text-sm text-muted">Loading…</div>}
      {results && (
        <>
          <Table
            title="Retrieval"
            runs={of(results.retrieval)}
            metrics={RETRIEVAL.map((m) => [m, m])}
          />
          <Table title="Answers" runs={of(results.generation)} metrics={GENERATION} />
        </>
      )}
    </div>
  );
}
