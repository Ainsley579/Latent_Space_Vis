"use client";

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
  resolveImageSrc: (realId: string | number, imageType: string) => string;
  resolveSubstructureImage: (model: string, layer: string, cluster: number, group: "knn" | "kfn") => string;
  substructureByCluster: Record<number, ClusterSubstructureFlags>;
  clusterBadgeColor: (cluster: number) => string;
};

export default function TopkNearestPanel({
  rows,
  selectedModel,
  selectedLayer,
  selectedCluster,
  hoveredId,
  onToggleHoverId,
  resolveImageSrc,
  resolveSubstructureImage,
  substructureByCluster,
  clusterBadgeColor,
}: TopkNearestPanelProps) {
  return (
    <div className="w-full h-full p-2 box-border overflow-auto">
      {rows.length === 0 && (
        <div className="text-sm text-zinc-500">
          No top-k nearest data for {selectedModel} / {selectedLayer}.
        </div>
      )}
      {rows.length > 0 && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-1 min-w-max">
            <div className="shrink-0 w-6" />
            <div className="flex flex-nowrap items-center gap-1 min-w-max">
              <div className="w-18 text-xs font-semibold text-zinc-700 text-center">KNN</div>
              <div className="w-25 text-xs font-semibold text-zinc-700 text-center">KFN</div>
              <div className="text-sm font-semibold text-zinc-700 pl-1">Top-k Nearest by Cluster</div>
            </div>
          </div>
          {rows.map((row) => {
            const subFlags = substructureByCluster[row.cluster_id] ?? { knn: false, kfn: false };
            const subCards: Array<{ group: "knn" | "kfn"; label: string; hasData: boolean }> = [
              { group: "knn", label: "KNN", hasData: subFlags.knn },
              { group: "kfn", label: "KFN", hasData: subFlags.kfn },
            ];

            return (
              <div key={`nearest-row-${row.cluster_id}`} className="flex items-start gap-2 min-w-max">
                <div
                  className="shrink-0 text-xs font-semibold text-white px-1.5 py-0.5 rounded-[2px] mt-0.5"
                  style={{ backgroundColor: clusterBadgeColor(row.cluster_id) }}
                >
                  {row.cluster_id}
                </div>

                <div className="flex flex-nowrap items-start gap-1 min-w-max">
                  {subCards.map((sub) => (
                    <div key={`${row.cluster_id}-${sub.group}`} className="relative w-20 h-20 shrink-0">
                      {sub.hasData ? (
                        <img
                          src={resolveSubstructureImage(selectedModel, selectedLayer, row.cluster_id, sub.group)}
                          alt={`${sub.label} substructure ${row.cluster_id}`}
                          className={
                            `w-20 h-20 object-contain bg-white border ` +
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
                            `w-20 h-20 bg-white border flex items-center justify-center text-sm text-zinc-400 ` +
                            (selectedCluster === row.cluster_id ? "border-black" : "border-zinc-200")
                          }
                        >
                          null
                        </div>
                      )}
                    </div>
                  ))}

                  <div className="w-px h-32 bg-zinc-300 shrink-0 mx-1" />

                  {row.top_k_closest_real_ids.map((id) => (
                    <div key={`nearest-${row.cluster_id}-${id}`} className="relative w-32 h-32 shrink-0">
                      <img
                        src={resolveImageSrc(id, "ligand")}
                        alt={`ligand ${id}`}
                        className={
                          `w-32 h-32 object-contain bg-white border ` +
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
                        className={
                          `absolute top-0.5 left-0.5 text-[10px] px-1 rounded bg-white/85 cursor-pointer ` +
                          (hoveredId === String(id) ? "text-amber-700 font-semibold" : "text-zinc-700")
                        }
                        onClick={() => onToggleHoverId(String(id))}
                      >
                        {id}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
