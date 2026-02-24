"use client";

import { useMemo, useState } from "react";

type SyncControlPanelProps = {
  dataset: string;
  onDatasetChange: (dataset: string) => void;
  datasetOptions: string[];
  clustering?: string;
  clusteringOptions?: Array<{ value: string; label: string }>;
  onClusteringChange?: (clustering: string) => void;
  syncOn: boolean;
  onToggleSync: () => void;
  largePreviewOn: boolean;
  onToggleLargePreview: () => void;
  largePreviewSize: number;
  onLargePreviewSizeChange: (size: number) => void;
  onExportBookmarks?: () => void;
  bookmarkCount?: number;
};

export default function SyncControlPanel({
  dataset,
  onDatasetChange,
  datasetOptions,
  clustering,
  clusteringOptions = [],
  onClusteringChange,
  syncOn,
  onToggleSync,
  largePreviewOn,
  onToggleLargePreview,
  largePreviewSize,
  onLargePreviewSizeChange,
  onExportBookmarks,
  bookmarkCount = 0,
}: SyncControlPanelProps) {
  const [minValue, setMinValue] = useState(1);
  const [maxValue, setMaxValue] = useState(20);

  const minOptions = useMemo(() => {
    return Array.from({ length: 20 }, (_, i) => i + 1).filter((v) => v < maxValue);
  }, [maxValue]);

  const maxOptions = useMemo(() => {
    return Array.from({ length: 20 }, (_, i) => i + 1).filter((v) => v > minValue);
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
            if (minValue >= next) setMinValue(Math.max(1, next - 1));
          }}
        >
          {maxOptions.map((v) => (
            <option key={`max-${v}`} value={v}>
              {v}
            </option>
          ))}
        </select>
      </div>

      <div className="flex items-center justify-center gap-2 min-w-0">
        <button
          type="button"
          className="h-6 px-3 rounded border border-zinc-300 bg-zinc-50 text-xs text-zinc-700 hover:bg-zinc-100 shrink-0"
        >
          Generate
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
      </div>
    </div>
  );
}
