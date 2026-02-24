import { useMemo } from "react";

type SelectedNode = { model: string; layer: string; cluster: number } | null;

type TopkEntry = {
  layer_name: string;
  model_type: string;
  cluster_id: number;
  top_k_closest_real_ids: (string | number)[];
  top_k_farthest_real_ids: (string | number)[];
  k: number;
};

type BottomPanel2ContentProps = {
  selectedNode: SelectedNode;
  topkData: TopkEntry[];
  hoveredId: string | null;
  onToggleHoverId: (id: string) => void;
  resolveImageSrc: (realId: string | number, imageType: string) => string;
  clusterBadgeColor: (cluster: number) => string;
};

export default function BottomPanel2Content({
  selectedNode,
  topkData,
  hoveredId,
  onToggleHoverId,
  resolveImageSrc,
  clusterBadgeColor,
}: BottomPanel2ContentProps) {
  const topkEntry = useMemo(() => {
    if (!selectedNode) return null;
    return (
      topkData.find(
        (d) =>
          d.model_type === selectedNode.model &&
          d.layer_name === selectedNode.layer &&
          d.cluster_id === selectedNode.cluster,
      ) ?? null
    );
  }, [topkData, selectedNode]);

  return (
    <div className="w-full h-full p-2 box-border overflow-auto">
      {!selectedNode && (
        <div className="text-sm text-zinc-500">Select a cluster to view top-k closest/farthest.</div>
      )}
      {selectedNode && !topkEntry && (
        <div className="text-sm text-zinc-500">
          No top-k data for {selectedNode.model} / {selectedNode.layer} / cluster {selectedNode.cluster}.
        </div>
      )}
      {selectedNode && topkEntry && (
        <div className="flex flex-col gap-2">
          <div className="text-sm font-semibold text-zinc-700">Top-k Closest</div>
          <div className="flex flex-wrap gap-1">
            {topkEntry.top_k_closest_real_ids.map((id) => (
              <div key={`close-${id}`} className="relative w-32 h-32">
                <img
                  src={resolveImageSrc(id, "ligand")}
                  alt={`ligand ${id}`}
                  className="w-32 h-32 object-contain bg-white border border-zinc-200"
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
                  className="absolute top-1 right-1 text-[10px] leading-4 px-1.5 py-0.5 rounded-[2px] border border-white/80 text-white font-semibold shadow-sm"
                  style={{ backgroundColor: clusterBadgeColor(selectedNode.cluster) }}
                >
                  {selectedNode.cluster}
                </div>
                <div
                  className={
                    `absolute top-1 left-1 text-xs px-1 rounded bg-white/85 cursor-pointer ` +
                    (hoveredId === String(id) ? "text-amber-700 font-semibold" : "text-zinc-700")
                  }
                  onClick={() => onToggleHoverId(String(id))}
                >
                  {id}
                </div>
              </div>
            ))}
          </div>
          <div className="text-sm font-semibold text-zinc-700">Top-k Farthest</div>
          <div className="flex flex-wrap gap-1">
            {topkEntry.top_k_farthest_real_ids.map((id) => (
              <div key={`far-${id}`} className="relative w-32 h-32">
                <img
                  src={resolveImageSrc(id, "ligand")}
                  alt={`ligand ${id}`}
                  className="w-32 h-32 object-contain bg-white border border-zinc-200"
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
                  className="absolute top-1 right-1 text-[10px] leading-4 px-1.5 py-0.5 rounded-[2px] border border-white/80 text-white font-semibold shadow-sm"
                  style={{ backgroundColor: clusterBadgeColor(selectedNode.cluster) }}
                >
                  {selectedNode.cluster}
                </div>
                <div
                  className={
                    `absolute top-1 left-1 text-xs px-1 rounded bg-white/85 cursor-pointer ` +
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
      )}
    </div>
  );
}
