"use client";
import { useEffect, useMemo, useState } from "react";
import SankeyView from "./component/SankeyView";
import SankeyHorizontal from "./component/SankeyHorizontal";
import SankeyVertical from "./component/SankeyVertical";
import ScatterPlot from "./component/ScatterPlot";
import AicBicChart from "./component/AicBicChart";

export default function Home() {
  const [sankeyPath, setSankeyPath] = useState("/data/sankey/noaug_sankey_samples.json");
  const [sankeyPathVertical, setSankeyPathVertical] = useState("/data/sankey/vertical_model_samples_emb.json");
  const [sankeyTransform, setSankeyTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [selectedLayer, setSelectedLayer] = useState("emb");
  const [selectedModel, setSelectedModel] = useState("dimenet_noaug");
  const [scatterPath, setScatterPath] = useState("/data/scatterplot/dimenet_noaug_emb.json");
  const [selectedNode, setSelectedNode] = useState<{ model: string; layer: string; cluster: number } | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[] | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [imageMap, setImageMap] = useState<Record<string, Record<string, string>> | null>(null);
  const [substructures, setSubstructures] = useState<
    {
      layer_name: string;
      model_type: string;
      cluster_id: number;
      group: "knn" | "kfn";
      k: number;
      real_ids: number[];
      valid_molecules: number;
      missing_real_ids: number[];
      invalid_smiles_real_ids: number[];
      mcs_smarts: string | null;
      mcs_smiles: string | null;
    }[]
  >([]);
  const [cellSelected, setCellSelected] = useState(false);
  const [swapAxes, setSwapAxes] = useState(false);
  const [suppressScatterCluster, setSuppressScatterCluster] = useState(false);
  const [topkData, setTopkData] = useState<
    {
      layer_name: string;
      model_type: string;
      cluster_id: number;
      top_k_closest_real_ids: number[];
      top_k_farthest_real_ids: number[];
      k: number;
    }[]
  >([]);
  const allAicBicPaths = useMemo(() => {
    const models = ["dimenet_noaug", "dimenet10conf"];
    const layers = [
      "emb",
      "output_block_0",
      "output_block_1",
      "output_block_2",
      "output_block_3",
      "output_block_4",
      "u_mlp_input",
      "u_mlp_l1",
      "u_mlp_l2",
    ];
    return models.flatMap((m) => layers.map((l) => `/data/bic_aic/${m}_${l}.json`));
  }, []);

  function toggleNode(sel: { model: string; layer: string; cluster: number; ids?: string[] }) {
    if (
      selectedNode &&
      selectedNode.model === sel.model &&
      selectedNode.layer === sel.layer &&
      selectedNode.cluster === sel.cluster
    ) {
      setSelectedNode(null);
      setSelectedIds(null);
      setCellSelected(false);
      setSuppressScatterCluster(false);
      return;
    }
    setSelectedNode(sel);
    setSelectedIds(sel.ids ?? null);
    setSuppressScatterCluster(false);
  }

  useEffect(() => {
    setScatterPath(`/data/scatterplot/${selectedModel}_${selectedLayer}.json`);
  }, [selectedModel, selectedLayer]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setSwapAxes(params.get("swap") === "1");
  }, []);

  useEffect(() => {
    if (!selectedNode) {
      setCellSelected(false);
    }
  }, [selectedNode]);

  useEffect(() => {
    fetch("/data/topk/cluster_representatives.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load topk data");
        return res.json();
      })
      .then((json) => {
        setTopkData(Array.isArray(json) ? json : []);
      })
      .catch((err) => {
        console.error(err);
        setTopkData([]);
      });
  }, []);


  useEffect(() => {
    fetch("/data/substructures/cluster_substructures.json")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load substructure data");
        return res.json();
      })
      .then((json) => {
        setSubstructures(Array.isArray(json) ? json : []);
      })
      .catch((err) => {
        console.error(err);
        setSubstructures([]);
      });
  }, []);

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

  function resolveImageSrc(realId: string | number, imageType: string) {
    const key = String(realId);
    return `/molecule_images/${key}_${imageType}.png`;
  }

  function resolveSubstructureImage(
    model: string,
    layer: string,
    cluster: number,
    group: "knn" | "kfn",
  ) {
    const safeModel = model.replace(/[^a-zA-Z0-9_-]/g, "_");
    const safeLayer = layer.replace(/[^a-zA-Z0-9_-]/g, "_");
    return `/data/substructures/images/${safeModel}_${safeLayer}_cluster${cluster}_${group}.png`;
  }
  return (
    <div className="flex min-h-screen">
      <main className="flex min-h-screen">
        <div className="h-[850px] w-[1480px] flex  flex-wrap">
          <div className="w-[60%] h-[65%] p-1 box-border">
            <div
              id="sankey2d"
              className="w-full h-full border border-zinc-400"
            >
              <SankeyView
                onModelSelect={(model) => {
                  if (model === "dimenet_noaug") {
                    setSankeyPath("/data/sankey/noaug_sankey_samples.json");
                  } else if (model === "dimenet10conf") {
                    setSankeyPath("/data/sankey/10conf_sankey_samples.json");
                  }
                  setSelectedModel(model);
                  setSelectedNode(null);
                  setSelectedIds(null);
                }}
                onLayerSelect={(layer) => {
                  setSelectedLayer(layer);
                  setSankeyPathVertical(`/data/sankey/vertical_model_samples_${layer}.json`);
                  setSelectedNode(null);
                  setSelectedIds(null);
                }}
                onNodeSelect={(sel) => {
                  toggleNode(sel);
                  setSelectedModel(sel.model);
                  setSelectedLayer(sel.layer);
                  if (sel.model === "dimenet_noaug") {
                    setSankeyPath("/data/sankey/noaug_sankey_samples.json");
                  } else if (sel.model === "dimenet10conf") {
                    setSankeyPath("/data/sankey/10conf_sankey_samples.json");
                  }
                  setSankeyPathVertical(`/data/sankey/vertical_model_samples_${sel.layer}.json`);
                  setCellSelected(true);
                }}
                selectedNode={selectedNode}
                onCellSelect={(active) => setCellSelected(active)}
                onSwapAxes={(swap) => setSwapAxes(swap)}
                onTransform={(t) => setSankeyTransform(t)}
              />
            </div>
          </div>
          <div className="w-[20%] h-[65%] p-1 flex flex-col gap-1">
            <div
              id="aicbic"
              className="w-full flex-[1] border border-zinc-400"
            >
              <AicBicChart
                dataPath={`/data/bic_aic/${selectedModel}_${selectedLayer}.json`}
                title={`${selectedModel} / ${selectedLayer}`}
                isSelected={cellSelected}
                allDataPaths={allAicBicPaths}
              />
            </div>
            <div
              id="sankeyVertical"
              className="w-full flex-[3] border border-zinc-400 min-h-0 relative"
            >
              <div className="absolute top-1 right-2 text-[10px] text-zinc-600 bg-white/80 px-1 rounded">
                {swapAxes ? selectedModel : selectedLayer}
              </div>
              <SankeyVertical
                dataPath={swapAxes ? sankeyPath : sankeyPathVertical}
                mode={swapAxes ? "layer" : "model"}
                layerKey={selectedLayer}
                modelKey={selectedModel}
                syncTransform={{ x: sankeyTransform.x, y: sankeyTransform.y, scale: sankeyTransform.scale }}
                activeNodeKey={
                  selectedNode
                    ? swapAxes
                      ? `${selectedNode.layer}_${selectedNode.cluster}`
                      : `${selectedNode.model}_${selectedNode.cluster}`
                    : null
                }
                highlightId={hoveredId}
                onNodeSelect={(sel) => {
                  if (!sel.model || sel.layer == null) return;
                  toggleNode({ model: sel.model, layer: sel.layer ?? selectedLayer, cluster: sel.cluster, ids: sel.ids });
                  setSelectedModel(sel.model);
                  setSelectedLayer(sel.layer ?? selectedLayer);
                  if (sel.model === "dimenet_noaug") {
                    setSankeyPath("/data/sankey/noaug_sankey_samples.json");
                  } else if (sel.model === "dimenet10conf") {
                    setSankeyPath("/data/sankey/10conf_sankey_samples.json");
                  }
                  setSankeyPathVertical(`/data/sankey/vertical_model_samples_${sel.layer ?? selectedLayer}.json`);
                  setCellSelected(true);
                }}
              />
            </div>
          </div>
          <div className="w-[20%] h-[65%] p-1 box-border">
          
            <div
              id="instanceview"
              className="w-full h-full border border-zinc-400 bg-white"
            >
              <div className="w-full h-full p-2 box-border overflow-auto">
                {!selectedNode && (
                  <div className="text-xs text-zinc-500">Select a cluster to view top-k closest/farthest.</div>
                )}
                {selectedNode && !topkEntry && (
                  <div className="text-xs text-zinc-500">
                    No top-k data for {selectedNode.model} / {selectedNode.layer} / cluster {selectedNode.cluster}.
                  </div>
                )}
                {selectedNode && topkEntry && (
                  <div className="flex flex-col gap-2">
                    <div className="text-xs font-semibold text-zinc-700">Top-k Closest</div>
                    <div className="flex flex-wrap gap-2">
                      {topkEntry.top_k_closest_real_ids.map((id) => (
                        <div key={`close-${id}`} className="flex flex-col items-center gap-1">
                          <img
                            src={resolveImageSrc(id, "ligand")}
                            alt={`ligand ${id}`}
                            className="w-20 h-20 object-contain bg-white border border-zinc-200"
                            onError={(e) => {
                              const target = e.currentTarget as HTMLImageElement;
                              target.style.display = "none";
                            }}
                          />
                          <div
                            className={
                              `text-[11px] cursor-pointer ` +
                              (hoveredId === String(id) ? "text-amber-700 font-semibold" : "text-zinc-600")
                            }
                            onClick={() => setHoveredId((prev) => (prev === String(id) ? null : String(id)))}
                          >
                            {id}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="text-xs font-semibold text-zinc-700">Top-k Farthest</div>
                    <div className="flex flex-wrap gap-2">
                      {topkEntry.top_k_farthest_real_ids.map((id) => (
                        <div key={`far-${id}`} className="flex flex-col items-center gap-1">
                          <img
                            src={resolveImageSrc(id, "ligand")}
                            alt={`ligand ${id}`}
                            className="w-20 h-20 object-contain bg-white border border-zinc-200"
                            onError={(e) => {
                              const target = e.currentTarget as HTMLImageElement;
                              target.style.display = "none";
                            }}
                          />
                          <div
                            className={
                              `text-[11px] cursor-pointer ` +
                              (hoveredId === String(id) ? "text-amber-700 font-semibold" : "text-zinc-600")
                            }
                            onClick={() => setHoveredId((prev) => (prev === String(id) ? null : String(id)))}
                          >
                            {id}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="w-[60%] h-[35%] p-1 box-border">
            <div
              id="sankeyHorizontal"
              className="w-full h-full border border-zinc-400 relative"
            >
              <div className="absolute top-1 left-2 text-[10px] text-zinc-600 bg-white/80 px-1 rounded">
                {swapAxes ? selectedLayer : selectedModel}
              </div>
              <SankeyHorizontal
                dataPath={swapAxes ? sankeyPathVertical : sankeyPath}
                mode={swapAxes ? "model" : "layer"}
                modelKey={selectedModel}
                layerKey={selectedLayer}
                syncTransform={{ x: sankeyTransform.x, scale: sankeyTransform.scale }}
                activeNodeKey={
                  selectedNode
                    ? swapAxes
                      ? `${selectedNode.model}_${selectedNode.cluster}`
                      : `${selectedNode.layer}_${selectedNode.cluster}`
                    : null
                }
                highlightId={hoveredId}
                onNodeSelect={(sel) => {
                  if (!sel.layer || !sel.model) return;
                  toggleNode({ model: sel.model, layer: sel.layer, cluster: sel.cluster, ids: sel.ids });
                  setSelectedModel(sel.model);
                  setSelectedLayer(sel.layer);
                  if (sel.model === "dimenet_noaug") {
                    setSankeyPath("/data/sankey/noaug_sankey_samples.json");
                  } else if (sel.model === "dimenet10conf") {
                    setSankeyPath("/data/sankey/10conf_sankey_samples.json");
                  }
                  setSankeyPathVertical(`/data/sankey/vertical_model_samples_${sel.layer}.json`);
                  setCellSelected(true);
                }}
              />
            </div>
          </div>
          <div className="w-[20%] h-[35%] p-1 box-border">
            <div
              id="scatterplot"
              className="w-full h-full border border-zinc-400 relative"
            >
              <div className="absolute top-1 right-2 text-[10px] text-zinc-600 bg-white/80 px-1 rounded">
                {selectedModel} / {selectedLayer}
              </div>
              <ScatterPlot
                dataPath={scatterPath}
                activeCluster={suppressScatterCluster ? null : selectedNode?.cluster ?? null}
                activeIds={selectedIds}
                highlightId={hoveredId}
                modelKey={selectedModel}
                layerKey={selectedLayer}
                onPointSelect={(sel) => {
                  if (!sel.model || !sel.layer) return;
                  setSelectedNode({ model: sel.model, layer: sel.layer, cluster: sel.cluster });
                  setSelectedIds([sel.real_id]);
                  setHoveredId((prev) => (prev === String(sel.real_id) ? null : String(sel.real_id)));
                  setSuppressScatterCluster(true);
                  setSelectedModel(sel.model);
                  setSelectedLayer(sel.layer);
                  if (sel.model === "dimenet_noaug") {
                    setSankeyPath("/data/sankey/noaug_sankey_samples.json");
                  } else if (sel.model === "dimenet10conf") {
                    setSankeyPath("/data/sankey/10conf_sankey_samples.json");
                  }
                  setSankeyPathVertical(`/data/sankey/vertical_model_samples_${sel.layer}.json`);
                }}
                onClearSelection={() => {
                  setSelectedNode(null);
                  setSelectedIds(null);
                  setHoveredId(null);
                  setSuppressScatterCluster(false);
                }}
              />
            </div>
          </div>
          <div className="w-[20%] h-[35%] p-1 box-border">
            <div
              id="patternview"
              className="w-full h-full border border-zinc-400 bg-white"
            >
              <div className="w-full h-full p-2 box-border overflow-auto">
                {!selectedNode && (
                  <div className="text-xs text-zinc-500">Select a cluster to view substructures.</div>
                )}
                {selectedNode && !substructureEntry && (
                  <div className="text-xs text-zinc-500">
                    No substructure data for {selectedNode.model} / {selectedNode.layer} / cluster {selectedNode.cluster}.
                  </div>
                )}
                {selectedNode && substructureEntry && (
                  <div className="flex flex-col gap-3">
                    {(["knn", "kfn"] as const).map((group) => {
                      const entry = group === "knn" ? substructureEntry.knn : substructureEntry.kfn;
                      return (
                        <div key={group} className="flex flex-col gap-1">
                          <div className="text-xs font-semibold text-zinc-700">
                            {group === "knn" ? "KNN Substructure" : "KFN Substructure"}
                          </div>
                          {!entry && (
                            <div className="text-[11px] text-zinc-500">No entry for this group.</div>
                          )}
                          {entry && (
                            <div className="flex flex-col gap-1">
                              <img
                                src={resolveSubstructureImage(selectedNode.model, selectedNode.layer, selectedNode.cluster, group)}
                                alt={`${group} substructure`}
                                className="w-full max-h-36 object-contain border border-zinc-200 bg-white"
                                onError={(e) => {
                                  const target = e.currentTarget as HTMLImageElement;
                                  target.style.display = "none";
                                }}
                              />
                              <div className="text-[11px] text-zinc-600">
                                <span className="font-semibold">SMILES:</span>{" "}
                                {entry.mcs_smiles ?? "null"}
                              </div>
                              <div className="text-[11px] text-zinc-600 break-all">
                                <span className="font-semibold">SMARTS:</span>{" "}
                                {entry.mcs_smarts ?? "null"}
                              </div>
                              <div className="text-[11px] text-zinc-500">
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
            </div>
          </div>
        </div>
        
      </main>
    </div>
  );
}
