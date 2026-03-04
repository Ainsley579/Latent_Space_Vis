"use client";

import { useMemo, useState } from "react";

type SyncControlPanelProps = {
  dataset: string;
  onDatasetChange: (dataset: string) => void;
  datasetOptions: string[];
  dataSplit?: "test" | "train";
  onDataSplitChange?: (split: "test" | "train") => void;
  substructureMaxMembers?: number;
  onSubstructureMaxMembersChange?: (value: number) => void;
  clustering?: string;
  clusteringOptions?: Array<{ value: string; label: string }>;
  onClusteringChange?: (clustering: string) => void;
  cellBgMetric?: string;
  cellBgMetricOptions?: Array<{ value: string; label: string }>;
  onCellBgMetricChange?: (metric: string) => void;
  syncOn: boolean;
  onToggleSync: () => void;
  largePreviewOn: boolean;
  onToggleLargePreview: () => void;
  largePreviewSize: number;
  onLargePreviewSizeChange: (size: number) => void;
  onGenerateRange?: (minValue: number, maxValue: number) => void;
  onGenerateAgglomerative?: (config: {
    k: number | null;
    linkage: "ward" | "complete" | "average";
  }) => void;
  generateEnabled?: boolean;
  generateBusy?: boolean;
  generateMessage?: string;
  onExportBookmarks?: () => void;
  bookmarkCount?: number;
};

export default function SyncControlPanel({
  dataset,
  onDatasetChange,
  datasetOptions,
  dataSplit,
  onDataSplitChange,
  substructureMaxMembers,
  onSubstructureMaxMembersChange,
  clustering,
  clusteringOptions = [],
  onClusteringChange,
  cellBgMetric = "cluster_count",
  cellBgMetricOptions = [],
  onCellBgMetricChange,
  syncOn,
  onToggleSync,
  largePreviewOn,
  onToggleLargePreview,
  largePreviewSize,
  onLargePreviewSizeChange,
  onGenerateRange,
  onGenerateAgglomerative,
  generateEnabled = true,
  generateBusy = false,
  generateMessage = "",
  onExportBookmarks,
  bookmarkCount = 0,
}: SyncControlPanelProps) {
  const [minValue, setMinValue] = useState(2);
  const [maxValue, setMaxValue] = useState(10);
  const [agglomerativeKMode, setAgglomerativeKMode] = useState<"auto" | "manual">("auto");
  const [agglomerativeK, setAgglomerativeK] = useState(6);
  const [agglomerativeLinkage, setAgglomerativeLinkage] =
    useState<"ward" | "complete" | "average">("ward");
  const isAgglomerative = clustering === "agglomerative";
  const isPgmeans = clustering === "pgmeans";

  const minOptions = useMemo(() => {
    const list = Array.from({ length: 19 }, (_, i) => i + 2).filter((v) => v <= maxValue);
    return list.length ? list : [2];
  }, [maxValue]);

  const maxOptions = useMemo(() => {
    const list = Array.from({ length: 19 }, (_, i) => i + 2).filter((v) => v >= minValue);
    return list.length ? list : [20];
  }, [minValue]);

  return (
    <div className="w-full h-full bg-white flex flex-col overflow-hidden">
      <div className="shrink-0 px-2 py-1 text-sm font-semibold text-zinc-700 border-b border-zinc-200">
        Control Panel
      </div>
      <div className="flex-1 min-h-0 px-2 py-2 flex flex-col gap-2 overflow-y-auto">
      <div className="flex items-center justify-between gap-2 min-w-0 mb-2">
        <label className="text-[10px] text-zinc-700 shrink-0">DATASET</label>
        <select
          className="h-6 min-w-0 flex-1 px-1 text-xs border border-zinc-300 rounded bg-white"
          value={dataset}
          onChange={(e) => onDatasetChange(e.target.value)}
        >
          {datasetOptions.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </div>
      {onDataSplitChange && (
        <div className="flex items-center justify-between gap-2 min-w-0 mb-2">
          <label className="text-[10px] text-zinc-700 shrink-0">DATA SPLIT</label>
          <select
            className="h-6 min-w-0 flex-1 px-1 text-xs border border-zinc-300 rounded bg-white"
            value={dataSplit ?? "test"}
            onChange={(e) => onDataSplitChange(e.target.value as "test" | "train")}
          >
            <option value="test">test (default)</option>
            <option value="train">train</option>
          </select>
        </div>
      )}
      {onSubstructureMaxMembersChange && (
        <div className="flex items-center justify-between gap-2 min-w-0 mb-2">
          <label className="text-[10px] text-zinc-700 shrink-0">SUBSTRUCTURE M</label>
          <input
            type="number"
            min={1}
            max={200}
            step={1}
            className="h-6 w-16 px-1 text-xs border border-zinc-300 rounded bg-white"
            value={substructureMaxMembers ?? 32}
            onChange={(e) => {
              const val = Number(e.target.value) || 0;
              const clamped = Math.max(1, Math.min(200, val));
              onSubstructureMaxMembersChange(clamped);
            }}
          />
        </div>
      )}
      {clustering && clusteringOptions.length > 0 && onClusteringChange && (
        <div className="flex items-center justify-between gap-2 min-w-0 mb-2">
          <label className="text-[10px] text-zinc-700 shrink-0">CLUSTERING</label>
          <select
            className="h-6 min-w-0 flex-1 px-1 text-xs border border-zinc-300 rounded bg-white"
            value={clustering}
            onChange={(e) => onClusteringChange(e.target.value)}
          >
            {clusteringOptions.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </div>
      )}
      {onCellBgMetricChange && cellBgMetricOptions.length > 0 && (
        <div className="flex items-center justify-between gap-2 min-w-0 mb-2">
          <label className="text-[10px] text-zinc-700 shrink-0">CELL BG</label>
          <select
            className="h-6 min-w-0 flex-1 px-1 text-xs border border-zinc-300 rounded bg-white"
            value={cellBgMetric}
            onChange={(e) => onCellBgMetricChange(e.target.value)}
          >
            {cellBgMetricOptions.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
        </div>
      )}
      <div className="flex items-center justify-start gap-2 min-w-0 mb-2">
        <span className="text-[10px] text-zinc-700">SYNC</span>
        <button
          type="button"
          onClick={onToggleSync}
          className={
            "relative inline-flex h-5 w-10 items-center rounded-full transition-colors " +
            (syncOn ? "bg-emerald-500" : "bg-zinc-300")
          }
          aria-pressed={syncOn}
          title={syncOn ? "Sync ON" : "Sync OFF"}
        >
          <span
            className={
              "inline-block h-3 w-3 transform rounded-full bg-white shadow transition-transform " +
              (syncOn ? "translate-x-6" : "translate-x-1")
            }
          />
        </button>
      </div>

      <div className="flex items-center justify-start gap-2 min-w-0 mb-2">
        <span className="text-[10px] text-zinc-700">LARGE IMAGE</span>
        <button
          type="button"
          onClick={onToggleLargePreview}
          className={
            "relative inline-flex h-5 w-10 items-center rounded-full transition-colors " +
            (largePreviewOn ? "bg-emerald-500" : "bg-zinc-300")
          }
          aria-pressed={largePreviewOn}
          title={largePreviewOn ? "Large preview ON" : "Large preview OFF"}
        >
          <span
            className={
              "inline-block h-3 w-3 transform rounded-full bg-white shadow transition-transform " +
              (largePreviewOn ? "translate-x-6" : "translate-x-1")
            }
          />
        </button>
      </div>
      <div className={"mb-2 " + (largePreviewOn ? "opacity-100" : "opacity-40")}>
        <div className="flex items-center justify-between mb-1">
          <label className="text-[10px] text-zinc-700">SIZE</label>
          <span className="text-[10px] text-zinc-500">{largePreviewSize}px</span>
        </div>
        <input
          type="range"
          min={220}
          max={520}
          step={20}
          value={largePreviewSize}
          disabled={!largePreviewOn}
          onChange={(e) => onLargePreviewSizeChange(Number(e.target.value))}
          className="w-full h-1.5 accent-emerald-500 disabled:cursor-not-allowed"
        />
      </div>

      {isPgmeans ? (
        <>
          <div className="text-[10px] text-zinc-500 text-center mb-0.5">Clustering k range (min-max)</div>
          <div className="flex items-center justify-center gap-3 min-w-0">
            <label className="text-xs text-zinc-600 shrink-0">Min</label>
            <select
              className="h-6 w-14 px-1 text-xs border border-zinc-300 rounded bg-white shrink-0"
              value={minValue}
              onChange={(e) => {
                const next = Number(e.target.value);
                setMinValue(next);
                if (maxValue <= next) setMaxValue(Math.min(20, next + 1));
              }}
            >
              {minOptions.map((v) => (
                <option key={`min-${v}`} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center justify-center gap-3 min-w-0">
            <label className="text-xs text-zinc-600 shrink-0">Max</label>
            <select
              className="h-6 w-14 px-1 text-xs border border-zinc-300 rounded bg-white shrink-0"
              value={maxValue}
              onChange={(e) => {
                const next = Number(e.target.value);
                setMaxValue(next);
                if (minValue >= next) setMinValue(Math.max(2, next - 1));
              }}
            >
              {maxOptions.map((v) => (
                <option key={`max-${v}`} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>
        </>
      ) : isAgglomerative ? (
        <>
          <div className="text-[10px] text-zinc-500 text-center mb-0.5">Agglomerative parameters</div>
          <div className="flex items-center justify-between gap-2 min-w-0">
            <label className="text-xs text-zinc-600 shrink-0">Linkage</label>
            <select
              className="h-6 min-w-0 flex-1 px-1 text-xs border border-zinc-300 rounded bg-white"
              value={agglomerativeLinkage}
              onChange={(e) =>
                setAgglomerativeLinkage(e.target.value as "ward" | "complete" | "average")
              }
            >
              <option value="ward">ward</option>
              <option value="complete">complete</option>
              <option value="average">average</option>
            </select>
          </div>
          <div className="flex items-center justify-between gap-2 min-w-0">
            <label className="text-xs text-zinc-600 shrink-0">k mode</label>
            <select
              className="h-6 min-w-0 flex-1 px-1 text-xs border border-zinc-300 rounded bg-white"
              value={agglomerativeKMode}
              onChange={(e) => setAgglomerativeKMode(e.target.value as "auto" | "manual")}
            >
              <option value="auto">auto (BIC)</option>
              <option value="manual">manual</option>
            </select>
          </div>
          {agglomerativeKMode === "manual" ? (
            <div className="flex items-center justify-between gap-2 min-w-0">
              <label className="text-xs text-zinc-600 shrink-0">k</label>
              <select
                className="h-6 min-w-0 flex-1 px-1 text-xs border border-zinc-300 rounded bg-white"
                value={agglomerativeK}
                onChange={(e) => setAgglomerativeK(Number(e.target.value))}
              >
                {Array.from({ length: 14 }, (_, i) => i + 2).map((v) => (
                  <option key={`agg-k-${v}`} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
        </>
      ) : null}
      {!isPgmeans && !isAgglomerative ? (
        <div className="text-[10px] text-zinc-500 text-center">No extra parameters for this method.</div>
      ) : null}

      <div className="flex items-center justify-center gap-2 min-w-0">
        <button
          type="button"
          className={
            "h-6 px-3 rounded border text-xs shrink-0 " +
            (generateEnabled && !generateBusy
              ? "border-zinc-300 bg-zinc-50 text-zinc-700 hover:bg-zinc-100"
              : "border-zinc-200 bg-zinc-100 text-zinc-400 cursor-not-allowed")
          }
          disabled={!generateEnabled || generateBusy}
          onClick={() => {
            if (isAgglomerative) {
              onGenerateAgglomerative?.({
                k: agglomerativeKMode === "manual" ? agglomerativeK : null,
                linkage: agglomerativeLinkage,
              });
              return;
            }
            onGenerateRange?.(minValue, maxValue);
          }}
        >
          {generateBusy ? "Generating..." : "Generate"}
        </button>
        <button
          type="button"
          className={
            "h-6 px-3 rounded border text-xs shrink-0 " +
            (bookmarkCount > 0
              ? "border-zinc-300 bg-zinc-50 text-zinc-700 hover:bg-zinc-100"
              : "border-zinc-200 bg-zinc-100 text-zinc-400 cursor-not-allowed")
          }
          disabled={bookmarkCount <= 0}
          title={bookmarkCount > 0 ? `Export ${bookmarkCount} bookmarked IDs` : "No bookmarked IDs"}
          onClick={onExportBookmarks}
        >
          Export ({bookmarkCount})
        </button>
      </div>
      {generateMessage ? (
        <div className="text-[10px] text-zinc-500 text-center break-words">{generateMessage}</div>
      ) : null}
      </div>
    </div>
  );
}
