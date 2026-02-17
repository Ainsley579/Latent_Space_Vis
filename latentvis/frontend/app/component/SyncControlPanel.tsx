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
    <div className="w-full h-full bg-white px-1.5 py-1 flex flex-col gap-1 overflow-hidden">
      <div className="flex items-center flex-wrap gap-1 min-w-0">
        <label className="text-[10px] text-zinc-600">Min</label>
        <select
          className="h-5 w-12 px-1 text-[10px] border border-zinc-300 rounded bg-white shrink-0"
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

        <label className="text-[10px] text-zinc-600">Max</label>
        <select
          className="h-5 w-12 px-1 text-[10px] border border-zinc-300 rounded bg-white shrink-0"
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

      <div className="flex items-center justify-between gap-1 min-w-0">
        <button
          type="button"
          className="h-5 px-2 rounded border border-zinc-300 bg-zinc-50 text-[10px] text-zinc-700 hover:bg-zinc-100 shrink-0"
        >
          Generate
        </button>
        <div className="flex items-center gap-1 shrink-0">
          <span className="text-[9px] text-zinc-700">SYNC</span>
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
      </div>
    </div>
  );
}
