"use client";

import { useEffect, useMemo, useState } from "react";

type SyncControlPanelProps = {
  dataset: string;
  onDatasetChange: (dataset: string) => void;
  datasetOptions: string[];
  dataSplit?: "all" | "test" | "train";
  onDataSplitChange?: (split: "all" | "test" | "train") => void;
  sankeySplits?: Array<"train" | "test" | "val">;
  availableSankeySplits?: Array<"train" | "test" | "val">;
  onSankeySplitsChange?: (splits: Array<"train" | "test" | "val">) => void;
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
  onPgmeansRangeChange?: (minValue: number, maxValue: number) => void;
  onAgglomerativeParamsChange?: (config: {
    k: number | null;
    linkage: "ward" | "complete" | "average";
  }) => void;
  knnK?: number;
  onKnnKChange?: (k: number) => void;
  onExportBookmarks?: () => void;
  bookmarkCount?: number;
};

export default function SyncControlPanel({
  dataset,
  onDatasetChange,
  datasetOptions,
  sankeySplits = ["train", "test", "val"],
  availableSankeySplits = ["train", "test", "val"],
  onSankeySplitsChange,
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
  onPgmeansRangeChange,
  onAgglomerativeParamsChange,
  knnK: knnKProp,
  onKnnKChange,
  onExportBookmarks,
  bookmarkCount = 0,
}: SyncControlPanelProps) {
  const [minValue, setMinValue] = useState(2);
  const [maxValue, setMaxValue] = useState(10);
  const [agglomerativeKMode, setAgglomerativeKMode] = useState<"auto" | "manual">("auto");
  const [agglomerativeK, setAgglomerativeK] = useState(6);
  const [knnKLocal, setKnnKLocal] = useState(2);
  const [agglomerativeLinkage, setAgglomerativeLinkage] =
    useState<"ward" | "complete" | "average">("ward");
  const clusteringKey = String(clustering ?? "").trim().toLowerCase();
  const isAgglomerative = clusteringKey === "agglomerative";
  const isPgmeans = clusteringKey === "pgmeans";
  const isKnn = clusteringKey === "knn";
  const datasetKey = String(dataset ?? "").trim().toLowerCase();
  const isChemeleon = datasetKey === "chemeleon";
  const knnK = knnKProp ?? knnKLocal;
  const setKnnK = onKnnKChange ?? setKnnKLocal;
  const knnKOptions = useMemo(
    () => (isChemeleon ? [2, 3, 4] : Array.from({ length: 9 }, (_, i) => i + 2)),
    [isChemeleon]
  );
  const minKnnK = knnKOptions[0];
  const maxKnnK = knnKOptions[knnKOptions.length - 1];
  const normalizedKnnK = Number.isFinite(Number(knnK)) ? Number(knnK) : minKnnK;
  const clampedKnnK = Math.max(minKnnK, Math.min(maxKnnK, normalizedKnnK));

  useEffect(() => {
    if (!isKnn) return;
    if (clampedKnnK !== knnK) {
      setKnnK(clampedKnnK);
    }
  }, [clampedKnnK, isKnn, knnK, setKnnK]);

  const minOptions = useMemo(() => {
    const list = Array.from({ length: 9 }, (_, i) => i + 2).filter((v) => v <= maxValue);
    return list.length ? list : [2];
  }, [maxValue]);

  const maxOptions = useMemo(() => {
    const list = Array.from({ length: 9 }, (_, i) => i + 2).filter((v) => v >= minValue);
    return list.length ? list : [10];
  }, [minValue]);

  return (
    <div className="w-full bg-white flex flex-col overflow-visible">
      <div className="shrink-0 px-2 py-1 text-sm font-semibold text-zinc-700 border-b border-zinc-200">
        Control Panel
      </div>
      <div className="px-2 py-2 flex flex-col gap-2">
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
      {onSankeySplitsChange && (
        <div className="flex flex-col gap-1 min-w-0 mb-2">
          <label className="text-[10px] text-zinc-700 shrink-0">SANKEY SPLITS</label>
          <div className="flex flex-wrap gap-2">
            {availableSankeySplits.map((sp) => {
              const checked = sankeySplits.includes(sp);
              return (
                <label
                  key={`sankey-split-${sp}`}
                  className="inline-flex items-center gap-1 text-[11px] text-zinc-700"
                >
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 accent-emerald-600"
                    checked={checked}
                    onChange={(e) => {
                      const nextSet = new Set(sankeySplits);
                      if (e.target.checked) nextSet.add(sp);
                      else nextSet.delete(sp);
                      const next = availableSankeySplits.filter((x) => nextSet.has(x));
                      if (next.length === 0) return;
                      onSankeySplitsChange(next);
                    }}
                  />
                  <span>{sp}</span>
                </label>
              );
            })}
          </div>
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
        <>
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
        </>
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
                const nextMax = maxValue < next ? next : maxValue;
                setMinValue(next);
                if (maxValue < next) setMaxValue(nextMax);
                onPgmeansRangeChange?.(next, nextMax);
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
                const nextMin = minValue > next ? next : minValue;
                setMaxValue(next);
                if (minValue > next) setMinValue(nextMin);
                onPgmeansRangeChange?.(nextMin, next);
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
              onChange={(e) => {
                const nextLinkage = e.target.value as "ward" | "complete" | "average";
                setAgglomerativeLinkage(nextLinkage);
                onAgglomerativeParamsChange?.({
                  k: agglomerativeKMode === "manual" ? agglomerativeK : null,
                  linkage: nextLinkage,
                });
              }}
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
              onChange={(e) => {
                const nextMode = e.target.value as "auto" | "manual";
                setAgglomerativeKMode(nextMode);
                onAgglomerativeParamsChange?.({
                  k: nextMode === "manual" ? agglomerativeK : null,
                  linkage: agglomerativeLinkage,
                });
              }}
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
                onChange={(e) => {
                  const nextK = Number(e.target.value);
                  setAgglomerativeK(nextK);
                  onAgglomerativeParamsChange?.({
                    k: nextK,
                    linkage: agglomerativeLinkage,
                  });
                }}
              >
                {Array.from({ length: 9 }, (_, i) => i + 2).map((v) => (
                  <option key={`agg-k-${v}`} value={v}>
                    {v}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
        </>
      ) : isKnn ? (
        <>
          <div className="text-[10px] text-zinc-500 text-center mb-0.5">KNN parameter</div>
          <div className="flex items-center justify-between gap-2 min-w-0">
            <label className="text-xs text-zinc-600 shrink-0">k</label>
            <select
              className="h-6 min-w-0 flex-1 px-1 text-xs border border-zinc-300 rounded bg-white"
              value={clampedKnnK}
              onChange={(e) => setKnnK(Number(e.target.value))}
            >
              {knnKOptions.map((v) => (
                <option key={`knn-k-${v}`} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>
        </>
      ) : null}
      {!isPgmeans && !isAgglomerative && !isKnn ? (
        <div className="text-[10px] text-zinc-500 text-center">No extra parameters for this method.</div>
      ) : null}

      <div className="flex items-center justify-center gap-2 min-w-0">
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
      </div>
    </div>
  );
}
