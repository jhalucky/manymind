import { ChangeEvent, useEffect, useState } from "react";
import {
  getStatus,
  scoreClassifierCsv,
  scoreClassifierSample,
  scoreGuard,
  scoreSentinelCsv,
  scoreSentinelSample,
  type ClassifierResponse,
  type GuardResponse,
  type Mind,
  type SentinelResponse,
  type StatusResponse,
} from "./api";

function App() {
  const [mind, setMind] = useState<Mind>("sentinel");
  const [sentinel, setSentinel] = useState<SentinelResponse | null>(null);
  const [classifier, setClassifier] = useState<ClassifierResponse | null>(null);
  const [guard, setGuard] = useState<GuardResponse | null>(null);
  const [guardText, setGuardText] = useState("");
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch(() =>
        setError("API is offline. Start it with: python -m manymind.web.app"),
      );
  }, []);

  async function runSentinel() {
    setLoading(true);
    setError("");
    try {
      setSentinel(await scoreSentinelSample());
      setClassifier(null);
      setGuard(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sentinel sample failed.");
    } finally {
      setLoading(false);
    }
  }

  async function runClassifier() {
    setLoading(true);
    setError("");
    try {
      setClassifier(await scoreClassifierSample());
      setSentinel(null);
      setGuard(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Classifier sample failed.");
    } finally {
      setLoading(false);
    }
  }

  async function onUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    setLoading(true);
    setError("");
    try {
      const csv = await file.text();
      if (mind === "sentinel") {
        setSentinel(await scoreSentinelCsv(csv));
        setClassifier(null);
        setGuard(null);
      } else {
        setClassifier(await scoreClassifierCsv(csv));
        setSentinel(null);
        setGuard(null);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "CSV upload failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative min-h-screen">
      <header className="flex items-center justify-between px-6 py-4">
        <div className="flex items-center gap-3">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-gradient-to-br from-accent to-teal-700 font-heading text-sm font-medium text-black">
            M
          </span>
          <span className="font-heading text-[17px] font-medium tracking-tight">ManyMind</span>
        </div>
        <p className="hidden font-subhead text-[11px] uppercase tracking-[0.18em] text-zinc-500 sm:block">
          {status
            ? `Sentinel ${status.sentinel.available ? "ready" : "missing"} · Classifier ${status.classifier.source || "loading"} · Guard ${status.guard?.available ? "ready" : "missing"}`
            : "Loading minds"}
        </p>
      </header>

      <main className="mx-auto w-full max-w-2xl px-4 pb-16 pt-16">
        <section className="mb-8 text-center">
          <h1 className="font-heading text-5xl font-medium tracking-tighter sm:text-6xl">
            ManyMind
          </h1>
          <p className="mx-auto mt-4 max-w-xl font-subhead text-sm leading-relaxed text-zinc-400">
            {mind === "sentinel"
              ? "Sentinel is a fraud-review scorer for a payments desk. Upload a CSV with V1–V28 and Amount."
              : mind === "classifier"
                ? "Classifier is a churn-review scorer for a telecom desk. Upload a CSV with IBM Telco columns."
                : "Guard checks a KYC / bait SMS. Paste the message. Do not paste OTP, PIN, or Aadhaar."}
          </p>
        </section>

        <div className="mb-5 flex justify-center gap-2">
          {(["sentinel", "classifier", "guard"] as const).map((item) => (
            <button
              key={item}
              type="button"
              onClick={() => {
                setMind(item);
                setError("");
              }}
              className={`rounded-full px-4 py-1.5 font-subhead text-xs tracking-wide ${
                mind === item
                  ? "bg-accent/15 text-zinc-100"
                  : "border border-white/10 text-zinc-500"
              }`}
            >
              {item === "sentinel" ? "Sentinel" : item === "classifier" ? "Classifier" : "Guard"}
            </button>
          ))}
        </div>

        {mind === "guard" ? (
          <form
            className="space-y-3"
            onSubmit={async (event) => {
              event.preventDefault();
              setLoading(true);
              setError("");
              try {
                setGuard(await scoreGuard(guardText));
                setSentinel(null);
                setClassifier(null);
              } catch (err) {
                setError(err instanceof Error ? err.message : "Guard failed.");
              } finally {
                setLoading(false);
              }
            }}
          >
            <textarea
              value={guardText}
              onChange={(event) => setGuardText(event.target.value)}
              rows={5}
              placeholder="Paste the SMS or WhatsApp here"
              className="w-full rounded-2xl border border-white/10 bg-panel/90 p-4 font-sans text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
            />
            <button
              type="submit"
              disabled={loading || !guardText.trim()}
              className="h-12 w-full rounded-full bg-accent font-subhead text-xs tracking-wide text-black disabled:opacity-50"
            >
              {loading ? "Checking…" : "Check with Guard"}
            </button>
            <p className="px-1 font-subhead text-[11px] leading-relaxed text-zinc-600">
              Seed templates, not a bank dump. Never paste OTP, UPI PIN, or Aadhaar.
            </p>
          </form>
        ) : (
          <div className="space-y-3">
            {mind === "sentinel" ? (
              <button
                type="button"
                onClick={runSentinel}
                disabled={loading}
                className="h-12 w-full rounded-full bg-accent font-subhead text-xs tracking-wide text-black disabled:opacity-50"
              >
                {loading ? "Scoring cards…" : "Score sample credit-card CSV"}
              </button>
            ) : (
              <button
                type="button"
                onClick={runClassifier}
                disabled={loading}
                className="h-12 w-full rounded-full bg-accent font-subhead text-xs tracking-wide text-black disabled:opacity-50"
              >
                {loading ? "Scoring customers…" : "Score sample telco CSV"}
              </button>
            )}
            <label className="flex h-12 w-full cursor-pointer items-center justify-center rounded-full border border-white/15 font-subhead text-xs tracking-wide text-zinc-300">
              {mind === "sentinel" ? "Upload card CSV" : "Upload telco CSV"}
              <input
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                disabled={loading}
                onChange={onUpload}
              />
            </label>
            <p className="px-1 font-subhead text-[11px] leading-relaxed text-zinc-600">
              {mind === "sentinel"
                ? "Headers must include V1…V28 and Amount. First 200 rows are scored."
                : "Headers must match IBM Telco (tenure, Contract, MonthlyCharges, …). First 200 rows are scored."}
            </p>
          </div>
        )}

        {error ? <p className="mt-5 font-subhead text-sm text-red-400">{error}</p> : null}

        <section className="mt-6 space-y-3">
          {sentinel?.results.map((row) => (
            <article
              key={`${row.index}-${row.score}`}
              className="rounded-2xl border border-white/10 bg-panel/80 p-4"
            >
              <div className="flex items-baseline justify-between gap-3">
                <h2 className="font-heading text-lg font-medium tracking-tight">
                  row {row.index} · ${row.amount?.toFixed(2)}
                </h2>
                <span
                  className={`rounded-full px-2 py-0.5 font-subhead text-[11px] uppercase tracking-wider ${
                    row.prediction === "fraud"
                      ? "bg-red-500/10 text-red-300"
                      : "bg-emerald-500/10 text-emerald-300"
                  }`}
                >
                  {row.prediction}
                </span>
              </div>
              <p className="mt-2 font-subhead text-xs text-zinc-500">
                score {row.score.toFixed(4)} · threshold {row.threshold.toFixed(4)} · fraud
                if score &lt; threshold
              </p>
            </article>
          ))}

          {classifier?.results?.map((row) => (
            <article
              key={row.index}
              className="rounded-2xl border border-white/10 bg-panel/80 p-4"
            >
              <div className="flex items-baseline justify-between gap-3">
                <h2 className="font-heading text-lg font-medium tracking-tight">
                  row {row.index} · tenure {row.tenure}
                </h2>
                <span
                  className={`rounded-full px-2 py-0.5 font-subhead text-[11px] uppercase tracking-wider ${
                    row.prediction === "churn"
                      ? "bg-red-500/10 text-red-300"
                      : "bg-emerald-500/10 text-emerald-300"
                  }`}
                >
                  {row.prediction}
                </span>
              </div>
              <p className="mt-2 font-subhead text-xs text-zinc-500">
                churn probability {(row.churn_probability * 100).toFixed(1)}%
              </p>
            </article>
          ))}

          {guard ? (
            <article className="rounded-2xl border border-white/10 bg-panel/80 p-4">
              <div className="flex items-baseline justify-between gap-3">
                <h2 className="font-heading text-lg font-medium tracking-tight">Guard</h2>
                <span
                  className={`rounded-full px-2 py-0.5 font-subhead text-[11px] uppercase tracking-wider ${
                    guard.prediction === "bait"
                      ? "bg-red-500/10 text-red-300"
                      : "bg-emerald-500/10 text-emerald-300"
                  }`}
                >
                  {guard.prediction}
                </span>
              </div>
              <p className="mt-2 font-subhead text-xs text-zinc-500">
                bait probability {(guard.bait_probability * 100).toFixed(0)}%
                {guard.flags.length ? ` · ${guard.flags.join(", ")}` : ""}
              </p>
              <p className="mt-2 font-subhead text-xs leading-relaxed text-zinc-400">
                {guard.advice}
              </p>
            </article>
          ) : null}

          {sentinel || classifier || guard ? (
            <p className="font-subhead text-[11px] leading-relaxed text-zinc-600">
              {(sentinel ?? classifier ?? guard)?.disclaimer}
            </p>
          ) : null}
        </section>
      </main>

      <footer className="pb-8 text-center font-subhead text-[11px] text-zinc-600">
        Public ULB cards, IBM Telco, and a KYC seed. Not a live bank or carrier.
      </footer>
    </div>
  );
}

export default App;
