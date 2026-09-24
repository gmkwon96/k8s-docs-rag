"use client";

import { useEffect, useMemo, useState } from "react";
import { AnswerView } from "@/components/AnswerView";
import { ChunksPanel } from "@/components/ChunksPanel";
import {
  type Answer,
  type Example,
  type Hit,
  type VersionChoice,
  LiveDisabled,
  STATIC,
  ask,
  examples as loadExamples,
  health,
  retrieve,
} from "@/lib/api";

const CATEGORY_LABELS: Record<string, string> = {
  fact: "Fact",
  howto: "How-to",
  version: "Version",
  multihop: "Multi-hop",
  unanswerable: "Not in docs",
  false_premise: "False premise",
};

type Shown = {
  question: string;
  version: VersionChoice | null;
  answer: Answer | null;
  hits: Hit[] | null;
  fromExample: string | null;
};

export default function AskPage() {
  const [versions, setVersions] = useState<string[]>([]);
  const [version, setVersion] = useState<string>("");
  const [question, setQuestion] = useState("");
  const [exampleList, setExampleList] = useState<Example[]>([]);
  const [shown, setShown] = useState<Shown | null>(null);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [liveOff, setLiveOff] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    health()
      .then((h) => setVersions(h.versions))
      .catch((e) => setError(String(e.message ?? e)));
    loadExamples()
      .then((r) => setExampleList(r.examples))
      .catch(() => {});
  }, []);

  const grouped = useMemo(() => {
    const out = new Map<string, Example[]>();
    for (const ex of exampleList) out.set(ex.category, [...(out.get(ex.category) ?? []), ex]);
    return [...out.entries()];
  }, [exampleList]);

  function showExample(ex: Example) {
    setError("");
    setQuestion(ex.question);
    setVersion(ex.version);
    setShown({
      question: ex.question,
      version: { version: ex.version, requested: ex.version, note: "" },
      answer: {
        text: ex.answer,
        found: ex.found,
        sources: ex.sources.map((s, i) => ({ ...s, number: i + 1 })),
      },
      hits: ex.hits ?? null,
      fromExample: ex.id,
    });
  }

  async function showChunks() {
    if (!shown) return;
    setBusy(true);
    setStatus("Retrieving…");
    try {
      const r = await retrieve(shown.question, shown.version?.version ?? (version || null));
      setShown({ ...shown, hits: r.hits, version: shown.version ?? r.version });
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
      setStatus("");
    }
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (q.length < 3) return;
    setBusy(true);
    setError("");
    const current: Shown = { question: q, version: null, answer: null, hits: null, fromExample: null };
    setShown(current);
    setStatus("Searching the docs…");
    try {
      for await (const ev of ask(q, version || null)) {
        if (ev.event === "version") current.version = ev.data;
        if (ev.event === "hits") {
          current.hits = ev.data.hits;
          setStatus("Writing the answer…");
        }
        if (ev.event === "answer") current.answer = ev.data;
        if (ev.event === "error") setError(ev.data.message);
        setShown({ ...current });
      }
    } catch (err) {
      if (err instanceof LiveDisabled) {
        setLiveOff(true);
        // Still show what retrieval finds; only the answer needs Claude.
        try {
          const r = await retrieve(q, version || null);
          setShown({ ...current, version: r.version, hits: r.hits });
        } catch (e2) {
          setError(String((e2 as Error).message));
        }
      } else setError(String((err as Error).message));
    } finally {
      setBusy(false);
      setStatus("");
    }
  }

  const cited = new Set(shown?.answer?.sources.map((s) => s.chunk_id) ?? []);

  return (
    <div className="grid gap-6 lg:grid-cols-[18rem_1fr]">
      <aside className="order-2 lg:order-1">
        <h2 className="mb-2 text-xs font-semibold uppercase tracking-wider text-muted">
          Example questions
        </h2>
        <p className="mb-3 text-xs text-muted">
          From the dev eval set, with the answers the pipeline gave in its latest run.
        </p>
        <div className="max-h-[70vh] space-y-4 overflow-y-auto pr-1">
          {grouped.map(([category, items]) => (
            <div key={category}>
              <div className="mb-1 text-sm font-medium">{CATEGORY_LABELS[category] ?? category}</div>
              <ul className="space-y-1">
                {items.slice(0, 6).map((ex) => (
                  <li key={ex.id}>
                    <button
                      onClick={() => showExample(ex)}
                      className={`w-full rounded px-2 py-1 text-left text-sm hover:bg-accent-soft ${
                        shown?.fromExample === ex.id ? "bg-accent-soft" : ""
                      }`}
                    >
                      <span className="line-clamp-2">{ex.question}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </aside>

      <section className="order-1 min-w-0 space-y-4 lg:order-2">
        {STATIC && (
          <div className="rounded-md border border-line bg-surface p-4 text-sm">
            <p className="font-medium">Static demo</p>
            <p className="mt-1 text-muted">
              Pick an example question to see the answer the pipeline generated in its latest
              eval run, the sources it cited, and the chunks it was given. Live questions need
              the API and a database; see the{" "}
              <a
                className="text-accent hover:underline"
                href="https://github.com/gmkwon96/k8s-docs-rag#local-setup-macos"
              >
                local setup
              </a>
              .
            </p>
          </div>
        )}
        <form onSubmit={submit} className="space-y-2" hidden={STATIC}>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            maxLength={1000}
            placeholder="Ask about Kubernetes, e.g. How do I pause a Deployment rollout?"
            className="w-full resize-y rounded-md border border-line bg-surface p-3 outline-none focus:border-accent"
          />
          <div className="flex flex-wrap items-center gap-3">
            <label className="text-sm text-muted">
              Version{" "}
              <select
                value={version}
                onChange={(e) => setVersion(e.target.value)}
                className="rounded border border-line bg-surface px-2 py-1 text-text"
              >
                <option value="">auto (latest or from the question)</option>
                {versions.map((v) => (
                  <option key={v} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </label>
            <button
              type="submit"
              disabled={busy}
              className="ml-auto rounded-md bg-accent px-4 py-1.5 text-sm font-medium text-white disabled:opacity-50"
            >
              Ask
            </button>
          </div>
        </form>

        {liveOff && (
          <div className="rounded-md bg-warn-bg p-3 text-sm text-warn-text">
            Live answers are off on this server (they call Claude on the operator&apos;s key).
            Retrieval still runs; pick an example question to see a generated answer.
          </div>
        )}
        {error && <div className="rounded-md bg-warn-bg p-3 text-sm text-warn-text">{error}</div>}
        {status && <div className="text-sm text-muted">{status}</div>}

        {shown && (
          <div className="space-y-4">
            <article className="rounded-lg border border-line bg-surface p-5">
              <div className="mb-3 flex flex-wrap items-baseline gap-2 text-xs text-muted">
                {shown.version && (
                  <span className="rounded bg-accent-soft px-1.5 py-0.5 font-medium text-accent">
                    Kubernetes {shown.version.version}
                  </span>
                )}
                {shown.fromExample && <span>stored answer · {shown.fromExample}</span>}
                {shown.answer && !shown.answer.found && <span>not found in the docs</span>}
              </div>
              {shown.version?.note && (
                <p className="mb-3 text-sm text-warn-text">{shown.version.note}</p>
              )}
              {shown.answer ? (
                <AnswerView text={shown.answer.text} sources={shown.answer.sources} />
              ) : (
                <p className="text-sm text-muted">
                  {liveOff ? "No live answer; the retrieved chunks are below." : "…"}
                </p>
              )}
            </article>

            <details open={STATIC || (!!shown.hits && !shown.answer)} className="group">
              <summary
                className="cursor-pointer text-sm font-medium"
                onClick={() => !shown.hits && !busy && showChunks()}
              >
                Retrieved chunks {shown.hits ? `(${shown.hits.length})` : "(click to load)"}
              </summary>
              <div className="mt-2">
                {shown.hits && <ChunksPanel hits={shown.hits} cited={cited} />}
              </div>
            </details>
          </div>
        )}
      </section>
    </div>
  );
}
