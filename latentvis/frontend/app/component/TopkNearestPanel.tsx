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
  substructureByCluster: Record<number, ClusterSubstructureData>;
  clusterBadgeColor: (cluster: number) => string;
  onClusterSelect?: (cluster: number) => void;
  hideTopk?: boolean;
  largePreviewOn?: boolean;
  onSmilesPreviewStart?: (smiles: string | null, anchor?: DOMRect) => void;
  onSmilesPreviewEnd?: () => void;
  bookmarkedIds: Set<string>;
  onToggleBookmark: (id: string) => void;
  knnTopm: "5" | "10" | "20" | "all";
  onKnnTopmChange: (value: "5" | "10" | "20" | "all") => void;
  isKnnTopmLoading?: boolean;
};

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
  isKnnTopmLoading = false,
}: TopkNearestPanelProps) {
  const PAGE_SIZE = 5;
  const [pageByCluster, setPageByCluster] = useState<Record<number, number>>({});

  return (
    <div className="w-full h-full p-2 box-border overflow-auto">
      {rows.length === 0 && (
        <div className="text-sm text-zinc-500">
          No top-k nearest data for {selectedModel} / {selectedLayer}.
        </div>
      )}
      {rows.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 w-full min-w-0">
            <div className="shrink-0 w-6" />
            <div className={"flex-1 min-w-0 grid items-center " + (hideTopk ? "grid-cols-1" : "grid-cols-[1fr_14px_5fr]")}>
              <div className="text-xs font-semibold text-zinc-700 text-center">
                <span>KNN</span>
                <label className="ml-1 inline-flex items-center gap-1 text-[11px] font-normal text-zinc-600">
                  <span>m</span>
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

                <div className={"flex-1 min-w-0 grid items-start " + (hideTopk ? "grid-cols-1" : "grid-cols-[1fr_14px_5fr]")}>
                  <div
                    className="relative w-full max-w-[180px] aspect-square mx-auto"
                    onMouseEnter={(e) => onSmilesPreviewStart?.(subData.knnSmiles, e.currentTarget.getBoundingClientRect())}
                    onMouseLeave={() => onSmilesPreviewEnd?.()}
                  >
                    {subData.knn ? (
                      <div
                        className={
                          `w-full h-full bg-white border p-2 flex flex-col ` +
                          (selectedCluster === row.cluster_id ? "border-black" : "border-zinc-200")
                        }
                      >
                        <SmilesStructure
                          smiles={subData.knnSmiles}
                          className="w-full h-full"
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
                  {!hideTopk && <div />}
                  {!hideTopk && <div className="grid grid-cols-5 gap-1">
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
                                  e.currentTarget.getBoundingClientRect()
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
                        onClick={id == null ? undefined : () => onSelectHoverId(String(id))}
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
                                `absolute bottom-0.5 left-0.5 text-[10px] px-1 rounded bg-white/85 cursor-pointer ` +
                                (hoveredId === String(id) ? "text-amber-700 font-semibold" : "text-zinc-700")
                              }
                              onMouseEnter={() => onHoverIdChange(String(id))}
                              onMouseLeave={() => onHoverIdChange(null)}
                              onClick={() => onToggleHoverId(String(id))}
                            >
                              {id}
                            </div>
                          </>
                        )}
                      </div>
                    ))}
                  </div>}
                  {!hideTopk && totalPages > 1 && (
                    <div className="mt-1 flex items-center justify-end gap-1 text-[11px] text-zinc-600">
                      <button
                        type="button"
                        className="px-1.5 py-0.5 border border-zinc-300 rounded disabled:opacity-40"
                        disabled={page <= 0}
                        onClick={() =>
                          setPageByCluster((prev) => ({ ...prev, [row.cluster_id]: Math.max(0, page - 1) }))
                        }
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
                        onClick={() =>
                          setPageByCluster((prev) => ({
                            ...prev,
                            [row.cluster_id]: Math.min(totalPages - 1, page + 1),
                          }))
                        }
                      >
                        Next
                      </button>
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
