// Mirrors api/schemas.py. (Later this can be generated from /openapi.json.)

export interface ModelInfo {
  name: string;
  weights_path: string;
  size_mb: number;
  modified: number;
  active: boolean;
}

export interface Detection {
  cls_id: number;
  cls_name: string;
  confidence: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface FrameDetections {
  image: string;
  width: number;
  height: number;
  detections: Detection[];
  error?: string | null;
}

export interface InferRequest {
  paths?: string[] | null;
  directory?: string | null;
  session?: string | null;
  model?: string | null;
  conf?: number;
  limit?: number;
}

export interface InferResponse {
  model: string;
  classes: Record<number, string>;
  conf: number;
  frames: FrameDetections[];
  total_detections: number;
  frames_with_detection: number;
}

export interface ClassCount {
  cls_id: number;
  cls_name: string;
  instances: number;
}

export interface DatasetStats {
  yaml: string;
  classes: ClassCount[];
  train_images: number;
  val_images: number;
  train_labels: number;
  val_labels: number;
  total_instances: number;
}

export interface ReportInfo {
  name: string;
  size_bytes: number;
  modified: number;
}

export interface ReportContent {
  name: string;
  content: string;
}

export type JobStatus = "pending" | "running" | "done" | "error";

export interface JobInfo {
  id: string;
  kind: string;
  status: JobStatus;
  cmd: string[];
  return_code?: number | null;
  started: number;
  ended?: number | null;
  line_count: number;
}

export interface AnnotateRequest {
  session: string;
  dataset?: string | null;
  rpm?: number;
  max_frames?: number;
  no_examples?: boolean;
  dry_run?: boolean;
}
