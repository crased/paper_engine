// Typed fetch client for the Paper Engine API.
// All calls go through /api (Vite proxies to uvicorn in dev).

import type {
  AnnotateRequest,
  DatasetStats,
  InferRequest,
  InferResponse,
  JobInfo,
  ModelInfo,
  ReportContent,
  ReportInfo,
} from "@/types";

const BASE = "/api";

async function http<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => http<{ status: string; service: string }>("/health"),

  models: () => http<ModelInfo[]>("/models"),

  datasetStats: (yaml = "dataset.yaml") =>
    http<DatasetStats>(`/dataset/stats?yaml=${encodeURIComponent(yaml)}`),

  reports: () => http<ReportInfo[]>("/reports"),
  report: (name: string) => http<ReportContent>(`/reports/${encodeURIComponent(name)}`),

  sessions: () => http<string[]>("/sessions"),

  infer: (req: InferRequest) =>
    http<InferResponse>("/infer", { method: "POST", body: JSON.stringify(req) }),

  jobs: () => http<JobInfo[]>("/jobs"),
  job: (id: string) => http<JobInfo>(`/jobs/${id}`),
  startTrain: () => http<JobInfo>("/jobs/train", { method: "POST", body: "{}" }),
  startAnnotate: (req: AnnotateRequest) =>
    http<JobInfo>("/jobs/annotate", { method: "POST", body: JSON.stringify(req) }),
  cancelJob: (id: string) => http<JobInfo>(`/jobs/${id}/cancel`, { method: "POST" }),

  // SSE log stream for a job. Returns an EventSource the caller wires up.
  streamJob(id: string): EventSource {
    return new EventSource(`${BASE}/jobs/${id}/stream`);
  },
};
