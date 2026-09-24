import type { ReactNode } from "react";
import type { Source } from "@/lib/api";

// A small renderer for the answers' markdown subset: fenced code, paragraphs, bullet
// lists, `code`, **bold**, and the [n] citation markers, which link to their source.

function inline(text: string, sources: Map<number, Source>, key: string): ReactNode[] {
  const parts = text.split(/(`[^`]+`|\*\*[^*]+\*\*|\[\d+\])/g);
  return parts.map((part, i) => {
    const k = `${key}-${i}`;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 1)
      return (
        <code key={k} className="rounded bg-code px-1 py-0.5 font-mono text-[0.85em]">
          {part.slice(1, -1)}
        </code>
      );
    if (part.startsWith("**") && part.endsWith("**") && part.length > 3)
      return <strong key={k}>{part.slice(2, -2)}</strong>;
    const m = part.match(/^\[(\d+)\]$/);
    if (m) {
      const source = sources.get(Number(m[1]));
      return source ? (
        <a
          key={k}
          href={source.url}
          target="_blank"
          rel="noreferrer"
          title={source.title ?? source.url}
          className="mx-0.5 rounded bg-accent-soft px-1 align-super text-[0.7em] font-medium text-accent no-underline hover:underline"
        >
          {m[1]}
        </a>
      ) : (
        <span key={k}>{part}</span>
      );
    }
    return <span key={k}>{part}</span>;
  });
}

export function AnswerView({ text, sources }: { text: string; sources: Source[] }) {
  const byNumber = new Map(sources.map((s) => [s.number, s]));
  const blocks: ReactNode[] = [];
  const chunks = text.split(/(```[\s\S]*?```)/g);
  chunks.forEach((chunk, ci) => {
    if (chunk.startsWith("```")) {
      const body = chunk.replace(/^```[^\n]*\n?/, "").replace(/```$/, "");
      blocks.push(
        <pre key={`c${ci}`} className="overflow-x-auto rounded-md bg-code p-3 font-mono text-sm">
          {body}
        </pre>,
      );
      return;
    }
    chunk
      .split(/\n{2,}/)
      .filter((p) => p.trim())
      .forEach((para, pi) => {
        const lines = para.split("\n");
        const key = `p${ci}-${pi}`;
        if (lines.every((l) => /^\s*([-*]|\d+\.)\s/.test(l))) {
          blocks.push(
            <ul key={key} className="ml-5 list-disc space-y-1">
              {lines.map((l, li) => (
                <li key={li}>{inline(l.replace(/^\s*([-*]|\d+\.)\s/, ""), byNumber, `${key}-${li}`)}</li>
              ))}
            </ul>,
          );
        } else {
          blocks.push(
            <p key={key}>
              {lines.flatMap((l, li) => [
                ...(li ? [<br key={`br${li}`} />] : []),
                ...inline(l.replace(/^#+\s/, ""), byNumber, `${key}-${li}`),
              ])}
            </p>,
          );
        }
      });
  });
  return (
    <div className="space-y-3 leading-relaxed">
      {blocks}
      {sources.length > 0 && (
        <ol className="mt-4 space-y-1 border-t border-line pt-3 text-sm">
          {sources.map((s) => (
            <li key={s.number} className="flex gap-2">
              <span className="w-5 shrink-0 text-right font-medium text-accent">{s.number}</span>
              <a href={s.url} target="_blank" rel="noreferrer" className="break-all hover:underline">
                {s.title || s.url.replace(/^https:\/\//, "")}
              </a>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
