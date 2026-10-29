import { useState } from "react";
import { AppShell } from "../components/layout/AppShell";
import api from "../services/api";
import type { Prediction } from "../types";
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function ComparePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<Prediction | null>(null);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const { data } = await api.post("/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(data);
    } catch (error) {
      console.error(error);
      alert("Analysis failed.");
    } finally {
      setLoading(false);
    }
  };

  const renderComparisonCard = (
    title: string,
    data: any,
    colorClass: string,
    strokeColor: string,
    description: string,
  ) => {
    if (!data) return null;
    return (
      <div className="glass-panel overflow-hidden rounded-[32px] p-6 flex flex-col gap-4">
        <div>
          <h3 className="font-display text-xl font-bold">{title}</h3>
          <div className="mt-2 flex items-center justify-between">
            <span className={`text-2xl font-bold ${data.prediction === "FAKE" ? "text-alert" : "text-emerald-400"}`}>
              {data.prediction}
            </span>
            <span className="text-sm text-slate-400">{data.confidence.toFixed(2)}% Conf.</span>
          </div>
          <p className="mt-3 text-xs leading-relaxed text-slate-400">{description}</p>
        </div>
        <div className="h-40 w-full mt-4">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data.timeline}>
              <XAxis dataKey="frameIndex" hide />
              <YAxis domain={[0, 1]} hide />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px" }}
                itemStyle={{ color: strokeColor }}
              />
              <Area
                type="monotone"
                dataKey="score"
                stroke={strokeColor}
                fill={strokeColor}
                fillOpacity={0.2}
                strokeWidth={2}
                isAnimationActive={true}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  return (
    <AppShell>
      <div className="space-y-6">
        <div className="glass-panel rounded-[32px] p-8">
          <p className="text-sm uppercase tracking-[0.3em] text-cyber">Deep Analysis</p>
          <h2 className="mt-3 font-display text-3xl font-bold">Model Comparison Dashboard</h2>
          <p className="mt-2 text-slate-300">
            Compare our trained CNN-LSTM model against a pretrained, independently-trained forensics model (Hugging Face) on the same input media. CNN (Spatial Only) is a derived estimate, not a separately trained model.
          </p>

          <form onSubmit={onSubmit} className="mt-8 flex flex-col md:flex-row md:items-center gap-4">
            <input
              type="file"
              accept="video/mp4,video/quicktime"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
              className="block w-full max-w-sm rounded-xl border border-white/10 bg-slate-900/50 px-4 py-3 text-sm text-slate-300 file:mr-4 file:rounded-full file:border-0 file:bg-cyber/10 file:px-4 file:py-2 file:text-sm file:font-semibold file:text-cyber hover:file:bg-cyber/20"
            />
            <button
              type="submit"
              disabled={!file || loading}
              className="rounded-xl bg-cyber px-6 py-3 font-semibold text-slate-950 disabled:opacity-50"
            >
              {loading ? "Analyzing Models..." : "Run Comparison"}
            </button>
          </form>
        </div>

        {result && result.modelsComparison && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {renderComparisonCard(
              "CNN (Spatial Only, derived)",
              result.modelsComparison.CNN,
              "text-emerald-400",
              "#10b981",
              "A lightweight spatial-only estimate derived from the primary model's score, used as a quick illustrative reference point rather than a separately trained architecture.",
            )}
            {renderComparisonCard(
              "Pretrained ViT (Hugging Face)",
              result.modelsComparison.LSTM,
              "text-amber-400",
              "#f59e0b",
              "A deepfake-forensics transformer trained independently by a third party on 2.7M images across 4,803 generators — not fine-tuned by this project. Validated strong on diverse, internet-sourced fakes (AUC 0.82) but near chance-level on Celeb-DF-style studio footage (AUC 0.60); reliability depends on how close your video is to real-world \"in the wild\" content.",
            )}
            {renderComparisonCard(
              "CNN-LSTM (Our Model)",
              result.modelsComparison.CNN_LSTM,
              "text-cyber",
              "#06b6d4",
              "This app's primary model: EfficientNet-B0 + BiLSTM, trained on Celeb-DF-v2. Deterministic and the one used for the main verdict elsewhere in the app, but identity-disjoint testing showed it struggles to generalize beyond that dataset's specific domain.",
            )}
          </div>
        )}
      </div>
    </AppShell>
  );
}
