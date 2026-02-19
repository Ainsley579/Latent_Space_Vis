"use client";
import { useEffect, useState, type MouseEvent as ReactMouseEvent } from "react";

type TopkNearestRow = {
  cluster_id: number;
  top_k_closest_real_ids: number[];
};

type ClusterSubstructureFlags = {
  knn: boolean;
  kfn: boolean;
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
  resolveImageSrc: (realId: string | number, imageType: string) => string;
  resolveSubstructureImage: (model: string, layer: string, cluster: number, group: "knn" | "kfn") => string;
  substructureByCluster: Record<number, ClusterSubstructureFlags>;
  clusterBadgeColor: (cluster: number) => string;
  onClusterSelect?: (cluster: number) => void;
  onImagePreviewStart?: (src: string, anchor?: DOMRect) => void;
  onImagePreviewEnd?: () => void;
  hideTopk?: boolean;
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
  resolveImageSrc,
  resolveSubstructureImage,
  substructureByCluster,
  clusterBadgeColor,
  onClusterSelect,
  onImagePreviewStart,
  onImagePreviewEnd,
  hideTopk = false,
}: TopkNearestPanelProps) {
  const PAGE_SIZE = 5;
  const [pageByCluster, setPageByCluster] = useState<Record<number, number>>({});

  useEffect(() => {
    setPageByCluster((prev) => {
      const next: Record<number, number> = {};
      rows.forEach((row) => {
        const totalPages = Math.max(1, Math.ceil(row.top_k_closest_real_ids.length / PAGE_SIZE));
        const prevPage = prev[row.cluster_id] ?? 0;
        next[row.cluster_id] = Math.min(prevPage, totalPages - 1);
      });
      return next;
    });
  }, [rows]);

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
              <div className="text-xs font-semibold text-zinc-700 text-center">KNN</div>
              {!hideTopk && <div />}
              {!hideTopk && (
                <div className="text-sm font-semibold text-zinc-700 text-left">Top-k Nearest by Cluster</div>
              )}
            </div>
          </div>
          {rows.map((row) => {
            const subFlags = substructureByCluster[row.cluster_id] ?? { knn: false, kfn: false };
            const totalPages = Math.max(1, Math.ceil(row.top_k_closest_real_ids.length / PAGE_SIZE));
            const page = Math.min(pageByCluster[row.cluster_id] ?? 0, totalPages - 1);
            const start = page * PAGE_SIZE;
            const pageIds = row.top_k_closest_real_ids.slice(start, start + PAGE_SIZE);
            const paddedPageIds: Array<number | null> = [...pageIds];
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
                    onMouseEnter={
                      subFlags.knn
                        ? (e: ReactMouseEvent<HTMLDivElement>) =>
                            onImagePreviewStart?.(
                              resolveSubstructureImage(selectedModel, selectedLayer, row.cluster_id, "knn"),
                              (e.currentTarget as HTMLDivElement).getBoundingClientRect(),
                            )
                        : undefined
                    }
                    onMouseLeave={subFlags.knn ? () => onImagePreviewEnd?.() : undefined}
                  >
                    {subFlags.knn ? (
                      <img
                        src={resolveSubstructureImage(selectedModel, selectedLayer, row.cluster_id, "knn")}
                        alt={`KNN substructure ${row.cluster_id}`}
                        className={
                          `w-full h-full object-contain bg-white border ` +
                          (selectedCluster === row.cluster_id ? "border-black" : "border-zinc-200")
                        }
                        onLoad={(e) => {
                          const target = e.currentTarget as HTMLImageElement;
                          target.style.display = "block";
                        }}
                        onError={(e) => {
                          const target = e.currentTarget as HTMLImageElement;
                          target.style.display = "none";
                        }}
                      />
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
                        onMouseEnter={id == null ? undefined : () => onHoverIdChange(String(id))}
                        onMouseLeave={id == null ? undefined : () => onHoverIdChange(null)}
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
                              className="absolute inset-0 z-[1]"
                              onMouseEnter={(e: ReactMouseEvent<HTMLDivElement>) =>
                                onImagePreviewStart?.(
                                  resolveImageSrc(id, "ligand"),
                                  (e.currentTarget as HTMLDivElement).getBoundingClientRect(),
                                )
                              }
                              onMouseLeave={() => onImagePreviewEnd?.()}
                            />
                            <img
                              src={resolveImageSrc(id, "ligand")}
                              alt={`ligand ${id}`}
                              className={
                                `w-full h-full object-contain bg-white border ` +
                                (selectedCluster === row.cluster_id ? "border-black" : "border-zinc-200")
                              }
                              onLoad={(e) => {
                                const target = e.currentTarget as HTMLImageElement;
                                target.style.display = "block";
                              }}
                              onError={(e) => {
                                const target = e.currentTarget as HTMLImageElement;
                                target.style.display = "none";
                              }}
                            />
                            <div
                              className="absolute top-0.5 right-0.5 text-[10px] leading-4 px-1.5 py-0.5 rounded-[2px] border border-white/80 text-white font-semibold shadow-sm"
                              style={{ backgroundColor: clusterBadgeColor(row.cluster_id) }}
                            >
                              {row.cluster_id}
                            </div>
                            <div
                              className={
                                `absolute top-0.5 left-0.5 text-[10px] px-1 rounded bg-white/85 cursor-pointer ` +
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
