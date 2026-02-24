"use client";

import { useMemo } from "react";

type SelectedNode = { model: string; layer: string; cluster: number };

type SubstructureRow = {
  layer_name: string;
  model_type: string;
  cluster_id: number;
  group: "knn" | "kfn";
  k: number;
  real_ids: (string | number)[];
  valid_molecules: number;
  missing_real_ids: number[];
  invalid_smiles_real_ids: number[];
  mcs_smarts: string | null;
  mcs_smiles: string | null;
  mcs_note?: string | null;
};

type PatternViewPanelProps = {
  selectedNode: SelectedNode | null;
  substructures: SubstructureRow[];
  resolveSubstructureImage: (model: string, layer: string, cluster: number, group: "knn" | "kfn") => string;
  clusterBadgeColor: (cluster: number) => string;
};

export default function PatternViewPanel({
  selectedNode,
  substructures,
  resolveSubstructureImage,
  clusterBadgeColor,
}: PatternViewPanelProps) {
  const substructureEntry = useMemo(() => {
    if (!selectedNode) return null;
    const matches = substructures.filter(
      (d) =>
        d.model_type === selectedNode.model &&
        d.layer_name === selectedNode.layer &&
        d.cluster_id === selectedNode.cluster,
    );
    if (!matches.length) return null;
    const knn = matches.find((d) => d.group === "knn") ?? null;
    const kfn = matches.find((d) => d.group === "kfn") ?? null;
    return { knn, kfn };
  }, [substructures, selectedNode]);

  return (
    <div className="w-full h-full p-2 box-border overflow-auto">
      {!selectedNode && (
        <div className="text-sm text-zinc-500">Select a cluster to view substructures.</div>
      )}
      {selectedNode && !substructureEntry && (
        <div className="text-sm text-zinc-500">
          No substructure data for {selectedNode.model} / {selectedNode.layer} / cluster {selectedNode.cluster}.
        </div>
      )}
      {selectedNode && substructureEntry && (
        <div className="flex flex-col gap-3">
          {(["knn", "kfn"] as const).map((group) => {
            const entry = group === "knn" ? substructureEntry.knn : substructureEntry.kfn;
            return (
              <div key={group} className="flex flex-col gap-1">
                <div className="text-sm font-semibold text-zinc-700">
                  {group === "knn" ? "KNN Substructure" : "KFN Substructure"}
                </div>
                {!entry && (
                  <div className="text-xs text-zinc-500">No entry for this group.</div>
                )}
                {entry && (
                  <div className="flex flex-col gap-1">
                    <div className="relative w-full">
                      <img
                        key={`${selectedNode.model}-${selectedNode.layer}-${selectedNode.cluster}-${group}`}
                        src={resolveSubstructureImage(selectedNode.model, selectedNode.layer, selectedNode.cluster, group)}
                        alt={`${group} substructure`}
                        className="w-full max-h-36 object-contain border border-zinc-200 bg-white"
                        onLoad={(e) => {
                          const target = e.currentTarget as HTMLImageElement;
                          target.style.display = "block";
                        }}
                        onError={(e) => {
                          const target = e.currentTarget as HTMLImageElement;
                          target.style.display = "none";
                        }}
                      />
                      {group !== "knn" && (
                        <div
                          className="absolute top-1 right-1 text-[10px] leading-4 px-1.5 py-0.5 rounded-[2px] border border-white/80 text-white font-semibold shadow-sm"
                          style={{ backgroundColor: clusterBadgeColor(selectedNode.cluster) }}
                        >
                          {selectedNode.cluster}
                        </div>
                      )}
                    </div>
                    <div className="text-xs text-zinc-600">
                      <span className="font-semibold">SMILES:</span>{" "}
                      {entry.mcs_smiles ?? entry.mcs_note ?? "No common substructure"}
                    </div>
                    <div className="text-xs text-zinc-600 break-all">
                      <span className="font-semibold">SMARTS:</span>{" "}
                      {entry.mcs_smarts ?? entry.mcs_note ?? "No common substructure"}
                    </div>
                    <div className="text-xs text-zinc-500">
                      Valid molecules: {entry.valid_molecules} / K={entry.k}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
