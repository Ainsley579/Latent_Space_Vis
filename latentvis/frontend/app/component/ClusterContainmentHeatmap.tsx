"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type ContainmentEntry = {
  user_cluster_id: number;
  our_cluster_id: number;
  count: number;
  percentage_of_user_cluster: number;
};

type ClusterContainmentHeatmapProps = {
  userClusterIds: number[];
  ourClusterIds: number[];
  matrix: number[][];
  entries?: ContainmentEntry[];
  userClusterLabelNames?: string[];
  loading?: boolean;
  error?: string | null;
  title?: string;
};

function colorForPct(p: number): string {
  const t = Math.max(0, Math.min(1, p / 100));
  const r = Math.round(245 - 130 * t);
  const g = Math.round(248 - 40 * t);
  const b = Math.round(255 - 180 * t);
  return `rgb(${r}, ${g}, ${b})`;
}

export default function ClusterContainmentHeatmap({
  userClusterIds,
  ourClusterIds,
  matrix,
  entries = [],
  userClusterLabelNames = [],
  loading = false,
  error = null,
  title = "Containment Heatmap",
}: ClusterContainmentHeatmapProps) {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const plotRef = useRef<HTMLDivElement | null>(null);
  const [plotSize, setPlotSize] = useState({ width: 1, height: 1 });
  const leftPad = 120;
  const topPad = 36;
  const rightPad = 8;
  const bottomPad = 8;
  const width = Math.max(1, plotSize.width);
  const height = Math.max(1, plotSize.height);
  const availW = Math.max(1, width - leftPad - rightPad);
  const availH = Math.max(1, height - topPad - bottomPad);
  const cellW = ourClusterIds.length > 0 ? availW / ourClusterIds.length : availW;
  const cellH = userClusterIds.length > 0 ? availH / userClusterIds.length : availH;

  const entryByPair = useMemo(() => {
    const map = new Map<string, ContainmentEntry>();
    entries.forEach((e) => map.set(`${e.user_cluster_id}|${e.our_cluster_id}`, e));
    return map;
  }, [entries]);

  const userLabelFor = (userClusterId: number): string => {
    const label = userClusterLabelNames[userClusterId];
    if (typeof label === "string" && label.trim().length > 0) return label.trim();
    return `User ${userClusterId}`;
  };

  useEffect(() => {
    const el = plotRef.current;
    if (!el) return;
    const updateSize = () => {
      setPlotSize({
        width: Math.max(1, el.clientWidth),
        height: Math.max(1, el.clientHeight),
      });
    };
    updateSize();
    const observer = new ResizeObserver(updateSize);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  if (loading) {
    return <div className="w-full h-full flex items-center justify-center text-xs text-zinc-500">Computing containment...</div>;
  }
  if (error) {
    return <div className="w-full h-full p-2 text-xs text-red-600 overflow-auto">{error}</div>;
  }
  if (!userClusterIds.length || !ourClusterIds.length || !matrix.length) {
    return <div className="w-full h-full p-2 text-xs text-zinc-500">Upload CSV labels to compute containment.</div>;
  }

  return (
    <div className="w-full h-full relative bg-white flex flex-col min-h-0">
      <div className="px-2 py-1 border-b border-zinc-200 text-xs font-semibold text-zinc-700">{title}</div>
      <div ref={plotRef} className="flex-1 min-h-0 w-full h-full overflow-hidden">
        <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`}>
          <text x={leftPad + (ourClusterIds.length * cellW) / 2} y={14} textAnchor="middle" fontSize={11} fill="#52525b">
            Our Clusters (Backend)
          </text>
          <text
            x={14}
            y={topPad + (userClusterIds.length * cellH) / 2}
            textAnchor="middle"
            fontSize={11}
            fill="#52525b"
            transform={`rotate(-90, 14, ${topPad + (userClusterIds.length * cellH) / 2})`}
          >
            User Clusters (Uploaded)
          </text>

          {ourClusterIds.map((cid, c) => (
            <text key={`x-${cid}`} x={leftPad + c * cellW + cellW / 2} y={28} textAnchor="middle" fontSize={10} fill="#374151">
              {cid}
            </text>
          ))}

          {userClusterIds.map((cid, r) => {
            const label = userLabelFor(cid);
            return (
              <g key={`y-${cid}`}>
                <text x={leftPad - 8} y={topPad + r * cellH + cellH * 0.68} textAnchor="end" fontSize={10} fill="#374151">
                  {label}
                </text>
                <title>{`User cluster ${cid}: ${label}`}</title>
              </g>
            );
          })}

          {userClusterIds.map((uCid, r) =>
            ourClusterIds.map((oCid, c) => {
              const pct = matrix[r]?.[c] ?? 0;
              const entry = entryByPair.get(`${uCid}|${oCid}`);
              const count = entry?.count ?? 0;
              const userName = userLabelFor(uCid);
              const msg = `Our cluster ${oCid} contains ${pct.toFixed(2)}% of user cluster \"${userName}\" (count: ${count}).`;
              return (
                <rect
                  key={`${uCid}-${oCid}`}
                  x={leftPad + c * cellW}
                  y={topPad + r * cellH}
                  width={Math.max(1, cellW - 1)}
                  height={Math.max(1, cellH - 1)}
                  fill={colorForPct(pct)}
                  stroke="#e5e7eb"
                  strokeWidth={0.6}
                  onMouseEnter={(e) => setTip({ x: e.clientX + 12, y: e.clientY + 12, text: msg })}
                  onMouseMove={(e) => setTip({ x: e.clientX + 12, y: e.clientY + 12, text: msg })}
                  onMouseLeave={() => setTip(null)}
                />
              );
            })
          )}
        </svg>
      </div>

      {tip && (
        <div
          className="fixed z-[9999] pointer-events-none max-w-[280px] rounded border border-zinc-300 bg-white/95 px-2 py-1 text-[11px] text-zinc-700 shadow"
          style={{ left: tip.x, top: tip.y }}
        >
          {tip.text}
        </div>
      )}
    </div>
  );
}
