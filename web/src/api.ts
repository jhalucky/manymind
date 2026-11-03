export type Mind = "sentinel" | "classifier" | "guard";

export type SentinelResult = {
  index?: number;
  amount?: number;
  value?: number;
  prediction: string;
  score: number;
  threshold: number;
};

export type SentinelResponse = {
  mind: "sentinel";
  threshold: number;
  results: SentinelResult[];
  disclaimer: string;
};

export type ClassifierRow = {
  index: number;
  tenure: number;
  prediction: "churn" | "retain";
  churn_probability: number;
};

export type ClassifierResponse = {
  mind: "classifier";
  prediction?: "churn" | "retain";
  label?: number;
  churn_probability?: number;
  features?: Record<string, string | number>;
  results?: ClassifierRow[];
  source: string;
  disclaimer: string;
};

async function post<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = (await response.json()) as T & { error?: string };
  if (!response.ok) {
    throw new Error(data.error || `Request failed (${response.status})`);
  }
  return data;
}

export type StatusResponse = {
  name: string;
  default_threshold: number;
  sentinel: { available: boolean; artifact: string };
  classifier: {
    available: boolean;
    artifact: string;
    source: string;
    features: string[];
  };
  guard?: { available: boolean; artifact: string };
  disclaimer: string;
};

export type GuardResponse = {
  mind: "guard";
  text: string;
  prediction: "bait" | "ok";
  bait_probability: number;
  threshold: number;
  flags: string[];
  advice: string;
  disclaimer: string;
};

export function getStatus() {
  return fetch("/api/status").then(async (response) => {
    const data = (await response.json()) as StatusResponse & { error?: string };
    if (!response.ok) {
      throw new Error(data.error || `Status request failed (${response.status})`);
    }
    return data;
  });
}

export function predictSentinel(values: number[], threshold: number) {
  return post<SentinelResponse>("/api/sentinel", { values, threshold });
}

export function predictClassifier(features: Record<string, number>) {
  return post<ClassifierResponse>("/api/classifier", features);
}

export function scoreSentinelSample() {
  return fetch("/api/sentinel/sample").then(async (response) => {
    const data = (await response.json()) as SentinelResponse & { error?: string };
    if (!response.ok) {
      throw new Error(data.error || `Sentinel sample failed (${response.status})`);
    }
    return data;
  });
}

export function scoreClassifierSample() {
  return fetch("/api/classifier/sample").then(async (response) => {
    const data = (await response.json()) as ClassifierResponse & { error?: string };
    if (!response.ok) {
      throw new Error(data.error || `Classifier sample failed (${response.status})`);
    }
    return data;
  });
}

export function scoreSentinelCsv(csv: string) {
  return post<SentinelResponse>("/api/sentinel", { csv });
}

export function scoreClassifierCsv(csv: string) {
  return post<ClassifierResponse>("/api/classifier", { csv });
}

export function scoreGuard(text: string) {
  return post<GuardResponse>("/api/guard", { text });
}
