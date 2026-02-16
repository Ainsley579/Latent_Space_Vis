"use client";

import { useMemo, useState } from "react";

type SyncControlPanelProps = {
  syncOn: boolean;
  onToggleSync: () => void;
};

export default function SyncControlPanel({ syncOn, onToggleSync }: SyncControlPanelProps) {
  const [minValue, setMinValue] = useState(1);
  const [maxValue, setMaxValue] = useState(20);

  const minOptions = useMemo(() => {
    return Array.from({ length: 20 }, (_, i) => i + 1).filter((v) => v < maxValue);
  }, [maxValue]);

  const maxOptions = useMemo(() => {
    return Array.from({ length: 20 }, (_, i) => i + 1).filter((v) => v > minValue);
  }, [minValue]);

  return (
    <div className="w-full h-full bg-white px-3 py-2 flex flex-col justify-center gap-2">
      <div className="flex items-center gap-3">
        <label className="text-xs text-zinc-600">Min</label>
        <select
          className="h-7 px-2 text-xs border border-zinc-300 rounded bg-white"
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

        <label className="text-xs text-zinc-600">Max</label>
        <select
          className="h-7 px-2 text-xs border border-zinc-300 rounded bg-white"
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

        <button
          type="button"
          className="h-7 px-3 rounded border border-zinc-300 bg-zinc-50 text-xs font-semibold text-zinc-700 hover:bg-zinc-100"
        >
          Generate
        </button>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={onToggleSync}
          className={
            "relative inline-flex h-7 w-14 items-center rounded-full transition-colors " +
            (syncOn ? "bg-emerald-500" : "bg-zinc-300")
          }
          aria-pressed={syncOn}
          title={syncOn ? "Sync ON" : "Sync OFF"}
        >
          <span
            className={
              "inline-block h-5 w-5 transform rounded-full bg-white shadow transition-transform " +
              (syncOn ? "translate-x-8" : "translate-x-1")
            }
          />
        </button>
        <span className="text-xs font-semibold text-zinc-700">{syncOn ? "SYNC ON" : "SYNC OFF"}</span>
      </div>
    </div>
  );
}
