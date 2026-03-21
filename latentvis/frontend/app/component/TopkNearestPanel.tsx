"use client";
import { useState } from "react";
import SmilesStructure from "./SmilesStructure";

type TopkNearestRow = {
  cluster_id: number;
  top_k_closest_real_ids: Array<string | number>;
};

type ClusterSubstructureData = {
  knn: boolean;
  kfn: boolean;
  knnSmiles: string | null;
  knnSmarts: string | null;
};

type TopkNearestPanelProps = {
  rows: TopkNearestRow[];
  selectedModel: string;
  selectedLayer: string;
  selectedCluster: number | null;
  hoveredId: string | null;
  onToggleHoverId: (id: string) => void;
  onSelectHoverId: (id: string) => void;
  onHoverIdChange: (id: string | null) => void;
  getSmilesById: (realId: string | number) => string | null;
  getYTrueById: (realId: string | number) => number | null;
  getYPredById: (realId: string | number) => number | null;
  substructureByCluster: Record<number, ClusterSubstructureData>;
  clusterBadgeColor: (cluster: number) => string;
  onClusterSelect?: (cluster: number) => void;
  hideTopk?: boolean;
  largePreviewOn?: boolean;
  onSmilesPreviewStart?: (
    smiles: string | null,
    anchor?: HTMLElement | SVGElement | DOMRect | null
  ) => void;
  onSmilesPreviewEnd?: () => void;
  bookmarkedIds: Set<string>;
  onToggleBookmark: (id: string) => void;
  knnTopm: "5" | "10" | "20" | "all";
  onKnnTopmChange: (value: "5" | "10" | "20" | "all") => void;
  substructureSplit: "all" | "train";
  onSubstructureSplitChange: (value: "all" | "train") => void;
  isKnnTopmLoading?: boolean;
  tanimotoByCluster?: Record<number, number | null>;
  isTanimotoLoading?: boolean;
};

function parseSplitTaggedId(value: string | number): { split: "train" | "test" | null; realId: string } {
  const raw = String(value ?? "").trim();
  if (raw.startsWith("train::")) return { split: "train", realId: raw.slice("train::".length) };
  if (raw.startsWith("test::")) return { split: "test", realId: raw.slice("test::".length) };
  return { split: null, realId: raw };
}

export default function TopkNearestPanel({
  rows,
  selectedModel,
  selectedLayer,
  selectedCluster,
  hoveredId,
  onToggleHoverId,
  onSelectHoverId,
  onHoverIdChange,
  getSmilesById,
  getYTrueById,
  getYPredById,
  substructureByCluster,
  clusterBadgeColor,
  onClusterSelect,
  hideTopk = false,
  largePreviewOn = true,
  onSmilesPreviewStart,
  onSmilesPreviewEnd,
  bookmarkedIds,
  onToggleBookmark,
  knnTopm,
  onKnnTopmChange,
  substructureSplit,
  onSubstructureSplitChange,
  isKnnTopmLoading = false,
  tanimotoByCluster = {},
  isTanimotoLoading = false,
}: TopkNearestPanelProps) {
  const PAGE_SIZE = 5;
  const [pageByCluster, setPageByCluster] = useState<Record<number, number>>({});

  function ridgeStats(ids: Array<string | number>, getter: (id: string | number) => number | null) {
    const values = ids
      .map((id) => getter(id))
      .filter((v): v is number => Number.isFinite(Number(v)))
      .map((v) => Number(v));
    if (values.length === 0) return { values, avg: null as number | null, std: null as number | null };
    const n = values.length;
    const avg = values.reduce((s, v) => s + v, 0) / n;
    const variance = values.reduce((s, v) => s + (v - avg) * (v - avg), 0) / n;
    return { values, avg, std: Math.sqrt(Math.max(0, variance)) };
  }

  function RidgeMini({
    values,
    color,
    sharedDomain,
  }: {
    values: number[];
    color: string;
    sharedDomain?: { min: number; max: number } | null;
  }) {
    const W = 118;
    const H = 30;
    if (!values.length) {
      return (
        <div className="w-[118px] h-[30px] border border-zinc-200 bg-white flex items-center justify-center text-[10px] text-zinc-400">
          null
        </div>
      );
    }
    const localMinRaw = Math.min(...values);
    const localMaxRaw = Math.max(...values);
    const minRaw = sharedDomain ? sharedDomain.min : localMinRaw;
    const maxRaw = sharedDomain ? sharedDomain.max : localMaxRaw;
    const spanRaw = Math.max(1e-9, maxRaw - minRaw);
    const min = minRaw - spanRaw * 0.08;
    const max = maxRaw + spanRaw * 0.08;
    const n = values.length;
    const avg = values.reduce((s, v) => s + v, 0) / n;
    const variance = values.reduce((s, v) => s + (v - avg) * (v - avg), 0) / Math.max(1, n);
    const std = Math.sqrt(Math.max(variance, 1e-12));
    const silverman = 1.06 * std * Math.pow(Math.max(1, n), -0.2);
    const bandwidth = Math.max(spanRaw / 14, silverman, 1e-3);
    const sampleN = 72;
    const gaussian = (u: number) => Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
    const densities: number[] = [];
    for (let i = 0; i < sampleN; i += 1) {
      const xVal = min + (i / (sampleN - 1)) * (max - min);
      let sum = 0;
      for (let j = 0; j < values.length; j += 1) {
        sum += gaussian((xVal - values[j]) / bandwidth);
      }
      densities.push(sum / (n * bandwidth));
    }
    const peak = Math.max(1e-12, ...densities);
    const points = densities.map((d, i) => {
      const x = (i / (sampleN - 1)) * W;
      const y = H - (d / peak) * (H - 3);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });
    const area = `M 0,${H} L ${points.join(" L ")} L ${W},${H} Z`;
    const line = `M ${points.join(" L ")}`;
    return (
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="border border-zinc-200 bg-white">
        <path d={area} fill={color} fillOpacity={0.18} />
        <path d={line} fill="none" stroke={color} strokeWidth={1.4} />
      </svg>
    );
  }

  return (
    <div
      className="w-full h-full box-border overflow-auto"
      onMouseLeave={() => {
        onHoverIdChange(null);
        onSmilesPreviewEnd?.();
      }}
    >
      {rows.length === 0 && (
        <div className="text-sm text-zinc-500">
          No top-k nearest data for {selectedModel} / {selectedLayer}.
        </div>
      )}
      {rows.length > 0 && (
        <div className="flex flex-col gap-1 px-2 pb-2">
          <div className="sticky top-0 z-10 bg-white py-1 border-b border-zinc-200 flex items-center gap-2 w-full min-w-0">
            <div className="shrink-0 w-6" />
            <div className={"flex-1 min-w-0 grid items-center " + (hideTopk ? "grid-cols-[1fr_1fr]" : "grid-cols-[1fr_1fr_14px_5fr]")}>
              <div className="text-xs font-semibold text-zinc-700 text-center">Y Ridge</div>
              <div className="text-xs font-semibold text-zinc-700 text-center">
                <span>Top-k Substructure</span>
                <label className="ml-1 inline-flex items-center gap-1 text-[11px] font-normal text-zinc-600">
                  <select
                    className="rounded border border-zinc-300 bg-white px-1 py-0.5 text-[11px]"
                    value={knnTopm}
                    onChange={(e) => onKnnTopmChange(e.target.value as "5" | "10" | "20" | "all")}
                  >
                    <option value="5">5</option>
                    <option value="10">10</option>
                    <option value="20">20</option>
                    <option value="all">all</option>
                  </select>
                  <select
                    className="rounded border border-zinc-300 bg-white px-1 py-0.5 text-[11px]"
                    value={substructureSplit}
                    onChange={(e) => onSubstructureSplitChange(e.target.value as "all" | "train")}
                    title="Substructure source split"
                  >
                    <option value="all">all</option>
                    <option value="train">train</option>
                  </select>
                  {isKnnTopmLoading && <span className="text-zinc-500">...</span>}
                </label>
              </div>
              {!hideTopk && <div />}
              {!hideTopk && (
                <div className="text-sm font-semibold text-zinc-700 text-left">Top-k Nearest by Cluster</div>
              )}
            </div>
          </div>
          {rows.map((row) => {
            const subData = substructureByCluster[row.cluster_id] ?? { knn: false, kfn: false, knnSmiles: null, knnSmarts: null };
            const totalPages = Math.max(1, Math.ceil(row.top_k_closest_real_ids.length / PAGE_SIZE));
            const page = Math.min(pageByCluster[row.cluster_id] ?? 0, totalPages - 1);
            const start = page * PAGE_SIZE;
            const pageIds = row.top_k_closest_real_ids.slice(start, start + PAGE_SIZE);
            const paddedPageIds: Array<string | number | null> = [...pageIds];
            while (paddedPageIds.length < PAGE_SIZE) paddedPageIds.push(null);
            const yTrueStats = ridgeStats(row.top_k_closest_real_ids, getYTrueById);
            const yPredStats = ridgeStats(row.top_k_closest_real_ids, getYPredById);
            const sharedValues = [...yTrueStats.values, ...yPredStats.values];
            const sharedDomain =
              sharedValues.length > 0
                ? { min: Math.min(...sharedValues), max: Math.max(...sharedValues) }
                : null;

            return (
              <div key={`nearest-row-${row.cluster_id}`} className="flex items-start gap-2 w-full min-w-0">
                <div
                  className={
                    "shrink-0 text-xs font-semibold text-white px-1.5 py-0.5 rounded-[2px] mt-0.5 cursor-pointer " +
                    (selectedCluster === row.cluster_id ? "ring-2 ring-red-500" : "")
                  }
                  style={{ backgroundColor: clusterBadgeColor(row.cluster_id) }}
                  onClick={() => onClusterSelect?.(row.cluster_id)}
                  title={`Select cluster ${row.cluster_id}`}
                >
                  {row.cluster_id}
                </div>

                <div className={"flex-1 min-w-0 grid items-start " + (hideTopk ? "grid-cols-[1fr_1fr]" : "grid-cols-[1fr_1fr_14px_5fr]")}>
                  <div className="flex flex-col items-center gap-0.5">
                    <RidgeMini values={yTrueStats.values} color="#2563eb" sharedDomain={sharedDomain} />
                    <div className="text-[10px] text-zinc-600">
                      ytrue avg {yTrueStats.avg == null ? "-" : yTrueStats.avg.toFixed(3)} std{" "}
                      {yTrueStats.std == null ? "-" : yTrueStats.std.toFixed(3)}
                    </div>
                    <RidgeMini values={yPredStats.values} color="#d97706" sharedDomain={sharedDomain} />
                    <div className="text-[10px] text-zinc-600">
                      ypred avg {yPredStats.avg == null ? "-" : yPredStats.avg.toFixed(3)} std{" "}
                      {yPredStats.std == null ? "-" : yPredStats.std.toFixed(3)}
                    </div>
                  </div>
                  <div className="flex flex-col items-center">
                    {/*
                      Substructure preview represents the cluster rather than a single molecule.
                      Use the first top-k member as the hover proxy so scatter can highlight on image hover.
                    */}
                    {(() => {
                      const hoverProxyId =
                        row.top_k_closest_real_ids.length > 0 ? String(row.top_k_closest_real_ids[0]) : null;
                      return (
                    <div
                      className="relative w-full max-w-[100px] aspect-square mx-auto"
                      onMouseEnter={(e) => {
                        if (hoverProxyId) onHoverIdChange(hoverProxyId);
                        onSmilesPreviewStart?.(subData.knnSmiles, e.currentTarget);
                      }}
                      onMouseLeave={() => {
                        onHoverIdChange(null);
                        onSmilesPreviewEnd?.();
                      }}
                    >
                      {subData.knn ? (
                        <div
                          className={
                            `w-full h-full bg-white border p-1.5 flex flex-col ` +
                            (selectedCluster === row.cluster_id ? "border-black" : "border-zinc-200")
                          }
                        >
                          <SmilesStructure
                            smiles={subData.knnSmiles}
                            className="w-full h-full scale-[1.2]"
                          />
                        </div>
                      ) : (
                        <div
                          className={
                            `w-full h-full bg-white border flex items-center justify-center text-sm text-zinc-400 ` +
                            (selectedCluster === row.cluster_id ? "border-black" : "border-zinc-200")
                          }
                        >
                          null
                        </div>
                      )}
                    </div>
                      );
                    })()}
                    <div className="mt-0.5 text-[11px] text-zinc-600 text-center">
                      Tanimoto:{" "}
                      <span className="font-medium text-zinc-700">
                        {isTanimotoLoading && tanimotoByCluster[row.cluster_id] == null
                          ? "..."
                          : Number.isFinite(Number(tanimotoByCluster[row.cluster_id]))
                            ? Number(tanimotoByCluster[row.cluster_id]).toFixed(3)
                            : "-"}
                      </span>
                    </div>
                  </div>
                  {!hideTopk && <div />}
                  {!hideTopk && (
                    <div className="min-w-0 flex flex-col">
                      <div className="grid grid-cols-5 gap-1">
                        {paddedPageIds.map((id, idx) => (
                          <div
                            key={`nearest-${row.cluster_id}-${page}-${idx}-${id ?? "blank"}`}
                            className="relative w-full max-w-[180px] aspect-square mx-auto"
                            onMouseEnter={
                              id == null
                                ? undefined
                                : (e) => {
                                    onHoverIdChange(String(id));
                                    onSmilesPreviewStart?.(
                                      getSmilesById(id),
                                      e.currentTarget
                                    );
                                  }
                            }
                            onMouseLeave={
                              id == null
                                ? undefined
                                : () => {
                                    onHoverIdChange(null);
                                    onSmilesPreviewEnd?.();
                                  }
                            }
                          >
                            {id == null ? (
                              <div
                                className={
                                  `w-full h-full bg-white border flex items-center justify-center text-sm text-zinc-300 ` +
                                  (selectedCluster === row.cluster_id ? "border-black" : "border-zinc-200")
                                }
                              >
                                null
                              </div>
                            ) : (
                              <>
                                <div
                                  className={
                                    `w-full h-full bg-white border p-1 ` +
                                    (!largePreviewOn && hoveredId === String(id)
                                      ? "border-red-500 border-2"
                                      : selectedCluster === row.cluster_id
                                        ? "border-black"
                                        : "border-zinc-200")
                                  }
                                >
                                  <SmilesStructure smiles={getSmilesById(id)} className="w-full h-full" />
                                </div>
                                <div
                                  className="absolute top-0.5 left-0.5 text-[10px] leading-4 px-1.5 py-0.5 rounded-[2px] border border-white/80 text-white font-semibold shadow-sm"
                                  style={{ backgroundColor: clusterBadgeColor(row.cluster_id) }}
                                >
                                  {row.cluster_id}
                                </div>
                                <button
                                  type="button"
                                  className={
                                    "absolute top-0.5 right-0.5 z-[2] h-5 w-5 rounded bg-white/85 border shadow-sm flex items-center justify-center " +
                                    (bookmarkedIds.has(String(id)) ? "border-amber-500 text-amber-500" : "border-zinc-300 text-zinc-500")
                                  }
                                  title={bookmarkedIds.has(String(id)) ? "Remove bookmark" : "Add bookmark"}
                                  aria-label={bookmarkedIds.has(String(id)) ? "Remove bookmark" : "Add bookmark"}
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onToggleBookmark(String(id));
                                  }}
                                >
                                  <svg width="12" height="12" viewBox="0 0 24 24" aria-hidden="true">
                                    <path
                                      d="M7 4h10a1 1 0 0 1 1 1v15l-6-3-6 3V5a1 1 0 0 1 1-1z"
                                      fill={bookmarkedIds.has(String(id)) ? "currentColor" : "none"}
                                      stroke="currentColor"
                                      strokeWidth="2"
                                      strokeLinejoin="round"
                                    />
                                  </svg>
                                </button>
                                <div
                                  className={
                                    `absolute bottom-0.5 left-0.5 text-[10px] px-1 rounded bg-white/85 ` +
                                    (hoveredId === String(id) ? "text-amber-700 font-semibold" : "text-zinc-700")
                                  }
                                  onMouseEnter={() => onHoverIdChange(String(id))}
                                  onMouseLeave={() => onHoverIdChange(null)}
                                >
                                  {(() => {
                                    const parsed = parseSplitTaggedId(id);
                                    return parsed.split ? `${parsed.realId} (${parsed.split})` : parsed.realId;
                                  })()}
                                </div>
                              </>
                            )}
                          </div>
                        ))}
                      </div>
                      {totalPages > 1 && (
                        <div className="mt-1 flex items-center justify-end gap-1 text-[11px] text-zinc-600">
                          <button
                            type="button"
                            className="px-1.5 py-0.5 border border-zinc-300 rounded disabled:opacity-40"
                            disabled={page <= 0}
                            onClick={() => {
                              setPageByCluster((prev) => ({ ...prev, [row.cluster_id]: Math.max(0, page - 1) }));
                              onHoverIdChange(null);
                              onSmilesPreviewEnd?.();
                            }}
                          >
                            Prev
                          </button>
                          <span>
                            {page + 1}/{totalPages}
                          </span>
                          <button
                            type="button"
                            className="px-1.5 py-0.5 border border-zinc-300 rounded disabled:opacity-40"
                            disabled={page >= totalPages - 1}
                            onClick={() => {
                              setPageByCluster((prev) => ({
                                ...prev,
                                [row.cluster_id]: Math.min(totalPages - 1, page + 1),
                              }));
                              onHoverIdChange(null);
                              onSmilesPreviewEnd?.();
                            }}
                          >
                            Next
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
