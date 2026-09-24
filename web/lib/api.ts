// Types and calls for the FastAPI service, proxied under /api (see next.config.ts).

export type VersionChoice = { version: string; requested: string | null; note: string };

export type Hit = {
  content_id: number;
  score: number;
  text: string;
  version: string;
  chunk_id: string;
  url: string;
  title: string;
  heading_path: string[];
  feature_states: { feature_gate: string | null; state: string; since: string }[];
};

export type Source = {
  number: number;
  url: string;
  title?: string;
  chunk_id: string;
  cited_texts?: string[];
};

export type Answer = { text: string; found: boolean; sources: Source[]; usd?: number };

export type Example = {
  id: string;
  category: string;
  version: string;
  question: string;
  answer: string;
  found: boolean;
  sources: Source[];
};

export async function health(): Promise<{ versions: string[] }> {
  const r = await fetch("/api/health");
  if (!r.ok) throw new Error(`API unavailable (${r.status})`);
  return r.json();
}

export async function examples(): Promise<{ run: string; examples: Example[] }> {
  const r = await fetch("/api/examples");
  if (!r.ok) throw new Error(`Could not load examples (${r.status})`);
  return r.json();
}

export async function retrieve(
  question: string,
  version: string | null,
): Promise<{ version: VersionChoice; hits: Hit[]; latency: Record<string, number> }> {
  const r = await fetch("/api/retrieve", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, version }),
  });
  if (!r.ok) throw new Error(`Retrieval failed (${r.status})`);
  return r.json();
}

export type AskEvent =
  | { event: "version"; data: VersionChoice }
  | { event: "hits"; data: { hits: Hit[]; latency: Record<string, number> } }
  | { event: "answer"; data: Answer & { latency: Record<string, number> } }
  | { event: "error"; data: { type: string; message: string } };

/** POST /ask and yield its Server-Sent Events. Throws LiveDisabled on 403. */
export class LiveDisabled extends Error {}

export async function* ask(question: string, version: string | null): AsyncGenerator<AskEvent> {
  const r = await fetch("/api/ask", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question, version }),
  });
  if (r.status === 403) throw new LiveDisabled();
  if (!r.ok || !r.body) throw new Error(`Ask failed (${r.status})`);
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const fields = Object.fromEntries(
        block.split("\n").map((line) => {
          const i = line.indexOf(": ");
          return [line.slice(0, i), line.slice(i + 2)];
        }),
      );
      yield { event: fields.event, data: JSON.parse(fields.data) } as AskEvent;
    }
  }
}
