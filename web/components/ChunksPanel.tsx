import type { Hit } from "@/lib/api";

// The retrieval debug panel: what the reranker kept, in order, with its score and the
// chunk's feature states. Cited chunks are marked so misses are easy to spot.

export function ChunksPanel({ hits, cited }: { hits: Hit[]; cited: Set<string> }) {
  return (
    <ol className="space-y-2">
      {hits.map((h, i) => (
        <li key={h.chunk_id} className="rounded-md border border-line bg-surface p-3 text-sm">
          <div className="flex items-baseline gap-2">
            <span className="w-5 shrink-0 text-right tabular-nums text-muted">{i + 1}</span>
            <a
              href={h.url}
              target="_blank"
              rel="noreferrer"
              className="min-w-0 flex-1 truncate font-medium hover:underline"
              title={h.url}
            >
              {h.heading_path.filter(Boolean).join(" › ") || h.title}
            </a>
            {cited.has(h.chunk_id) && (
              <span className="rounded bg-accent-soft px-1.5 text-xs text-accent">cited</span>
            )}
            {h.score !== null && (
              <span className="font-mono text-xs tabular-nums text-muted">{h.score.toFixed(3)}</span>
            )}
          </div>
          {h.feature_states.length > 0 && (
            <div className="ml-7 mt-1 text-xs text-muted">
              {h.feature_states
                .map((s) => `${s.feature_gate ?? "feature"}: ${s.state} since v${s.since}`)
                .join(" · ")}
            </div>
          )}
          <p className="ml-7 mt-1 line-clamp-3 text-muted">{h.text}</p>
        </li>
      ))}
    </ol>
  );
}
