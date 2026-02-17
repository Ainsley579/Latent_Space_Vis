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
    <div className="w-full h-full bg-white px-2 py-1.5 flex flex-col justify-center gap-1.5">
      <div className="flex items-center gap-2">
        <label className="text-[11px] text-zinc-600">Min</label>
        <select
          className="h-6 px-1.5 text-[11px] border border-zinc-300 rounded bg-white"
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

        <label className="text-[11px] text-zinc-600">Max</label>
        <select
          className="h-6 px-1.5 text-[11px] border border-zinc-300 rounded bg-white"
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

      <div className="flex items-center justify-between">
        <button
          type="button"
          className="h-6 px-2.5 rounded border border-zinc-300 bg-zinc-50 text-[11px] font-semibold text-zinc-700 hover:bg-zinc-100"
        >
          Generate
        </button>
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold text-zinc-700">{syncOn ? "SYNC ON" : "SYNC OFF"}</span>
          <button
            type="button"
            onClick={onToggleSync}
            className={
              "relative inline-flex h-6 w-12 items-center rounded-full transition-colors " +
              (syncOn ? "bg-emerald-500" : "bg-zinc-300")
            }
            aria-pressed={syncOn}
            title={syncOn ? "Sync ON" : "Sync OFF"}
          >
            <span
              className={
                "inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform " +
                (syncOn ? "translate-x-7" : "translate-x-1")
              }
            />
          </button>
        </div>
      </div>
    </div>
  );
}
