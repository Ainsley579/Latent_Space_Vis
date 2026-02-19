"use client";
import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import SankeyView from "./component/SankeyView";
import SankeyHorizontal from "./component/SankeyHorizontal";
import SankeyVertical from "./component/SankeyVertical";
import ScatterPlot from "./component/ScatterPlot";
import TopkNearestPanel from "./component/TopkNearestPanel";
import StructureEditorPanel from "./component/StructureEditorPanel";
import SyncControlPanel from "./component/SyncControlPanel";
import { CLUSTER_COLORS } from "./theme/colors";

type SankeySampleRow = {
  real_id: string | number;
  from_layer: string;
  to_layer: string;
  from_model?: string;
  to_model?: string;
  from_cluster: number;
  to_cluster: number;
};

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
  const [hoverLock, setHoverLock] = useState(false);
  const [hoverPreview, setHoverPreview] = useState<{ src: string; left: number; top: number } | null>(null);
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
  const [activeSankeyRows, setActiveSankeyRows] = useState<SankeySampleRow[]>([]);
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
  const [syncLinkedSankeyViews, setSyncLinkedSankeyViews] = useState(false);
  const [showLargeHoverPreview, setShowLargeHoverPreview] = useState(true);
  const [largeHoverPreviewSize, setLargeHoverPreviewSize] = useState(320);
  const [isControlPanelOpen, setIsControlPanelOpen] = useState(false);
  const [clusterPage, setClusterPage] = useState(0);
  const [clusterPageSize, setClusterPageSize] = useState(10);
  const [bottomSplitPct, setBottomSplitPct] = useState(35);
  const [workspaceWidthPct, setWorkspaceWidthPct] = useState(80);
  const [workspaceHeightPct, setWorkspaceHeightPct] = useState(70);
  const layoutGridRef = useRef<HTMLDivElement | null>(null);
  const bottomSplitHostRef = useRef<HTMLDivElement | null>(null);
  const syncRafRef = useRef<number | null>(null);
  const pendingTransformRef = useRef<{ x: number; y: number; scale: number } | null>(null);
  const lastSyncedTransformRef = useRef<{ x: number; y: number; scale: number }>({ x: 0, y: 0, scale: 1 });
  function toggleNode(sel: { model: string; layer: string; cluster: number; ids?: string[] }) {
    setHoverLock(false);
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

  function setHoverFromMouse(id: string | null) {
    if (hoverLock) return;
    setHoveredId(id);
  }

  function handleSankeyTransformSync(next: { x: number; y: number; scale: number }) {
    pendingTransformRef.current = next;
    if (syncRafRef.current != null) return;

    syncRafRef.current = window.requestAnimationFrame(() => {
      syncRafRef.current = null;
      const t = pendingTransformRef.current;
      if (!t) return;

      const prev = lastSyncedTransformRef.current;
      const dx = Math.abs(t.x - prev.x);
      const dy = Math.abs(t.y - prev.y);
      const ds = Math.abs(t.scale - prev.scale);

      // Skip sub-pixel / tiny zoom jitter to reduce redraw flicker in linked Sankey views.
      if (dx < 0.8 && dy < 0.8 && ds < 0.002) return;

      lastSyncedTransformRef.current = t;
      setSankeyTransform(t);
    });
  }

  function toggleSyncLinkedViews() {
    setSyncLinkedSankeyViews((prev) => {
      const nextEnabled = !prev;
      if (nextEnabled) {
        const latest = pendingTransformRef.current ?? lastSyncedTransformRef.current;
        lastSyncedTransformRef.current = latest;
        setSankeyTransform(latest);
      }
      return nextEnabled;
    });
  }

  useEffect(() => {
    return () => {
      if (syncRafRef.current != null) {
        window.cancelAnimationFrame(syncRafRef.current);
        syncRafRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (!showLargeHoverPreview) {
      setHoverPreview(null);
    }
  }, [showLargeHoverPreview]);

  const syncTransformVertical = useMemo(
    () => ({ x: sankeyTransform.x, y: sankeyTransform.y, scale: sankeyTransform.scale }),
    [sankeyTransform.x, sankeyTransform.y, sankeyTransform.scale]
  );

  const syncTransformHorizontal = useMemo(
    () => ({ x: sankeyTransform.x, scale: sankeyTransform.scale }),
    [sankeyTransform.x, sankeyTransform.scale]
  );

  function toggleLockedHighlight(id: string) {
    if (hoverLock && hoveredId === id) {
      setHoveredId(null);
      setHoverLock(false);
      return;
    }
    setHoveredId(id);
    setHoverLock(true);
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

  const activeSankeyPath = useMemo(() => {
    return swapAxes ? sankeyPathVertical : sankeyPath;
  }, [swapAxes, sankeyPath, sankeyPathVertical]);

  useEffect(() => {
    fetch(activeSankeyPath)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load active sankey rows");
        return res.json();
      })
      .then((json: SankeySampleRow[]) => {
        setActiveSankeyRows(Array.isArray(json) ? json : []);
      })
      .catch((err) => {
        console.error(err);
        setActiveSankeyRows([]);
      });
  }, [activeSankeyPath]);

  const tabCycleData = useMemo(() => {
    if (!selectedNode) return null;

    const clusterSet = new Set<number>();
    const idsByCluster = new Map<number, Set<string>>();

    const add = (cluster: number, id: string | number) => {
      if (!Number.isFinite(cluster)) return;
      clusterSet.add(cluster);
      if (!idsByCluster.has(cluster)) idsByCluster.set(cluster, new Set());
      idsByCluster.get(cluster)?.add(String(id));
    };

    activeSankeyRows.forEach((r) => {
      if (swapAxes) {
        const fromModel = r.from_model ?? r.from_layer;
        const toModel = r.to_model ?? r.to_layer;
        if (fromModel === selectedNode.model) add(r.from_cluster, r.real_id);
        if (toModel === selectedNode.model) add(r.to_cluster, r.real_id);
      } else {
        if (r.from_layer === selectedNode.layer) add(r.from_cluster, r.real_id);
        if (r.to_layer === selectedNode.layer) add(r.to_cluster, r.real_id);
      }
    });

    const clusters = Array.from(clusterSet).sort((a, b) => a - b);
    return { clusters, idsByCluster };
  }, [activeSankeyRows, selectedNode, swapAxes]);

  useEffect(() => {
    function isInteractiveTarget(target: EventTarget | null) {
      const el = target as HTMLElement | null;
      if (!el) return false;
      const tag = el.tagName?.toLowerCase();
      return (
        tag === "input" ||
        tag === "textarea" ||
        tag === "select" ||
        el.isContentEditable
      );
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.key !== "Tab") return;
      if (!selectedNode || !tabCycleData || tabCycleData.clusters.length === 0) return;
      if (isInteractiveTarget(event.target)) return;

      event.preventDefault();

      const { clusters, idsByCluster } = tabCycleData;
      const currIdx = clusters.indexOf(selectedNode.cluster);
      const nextIdx = event.shiftKey
        ? (currIdx <= 0 ? clusters.length - 1 : currIdx - 1)
        : (currIdx === -1 ? 0 : (currIdx + 1) % clusters.length);
      const nextCluster = clusters[nextIdx];

      setSelectedNode({
        model: selectedNode.model,
        layer: selectedNode.layer,
        cluster: nextCluster,
      });
      setSelectedIds(Array.from(idsByCluster.get(nextCluster) ?? []));
      setSelectedModel(selectedNode.model);
      setSelectedLayer(selectedNode.layer);
      setCellSelected(true);
      setSuppressScatterCluster(false);
      setHoveredId(null);
      setHoverLock(false);
    }

    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [selectedNode, tabCycleData]);

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

  const topkNearestRows = useMemo(() => {
    return topkData
      .filter((d) => d.model_type === selectedModel && d.layer_name === selectedLayer)
      .sort((a, b) => a.cluster_id - b.cluster_id);
  }, [topkData, selectedModel, selectedLayer]);


  const substructureByCluster = useMemo(() => {
    const rows = substructures.filter(
      (d) => d.model_type === selectedModel && d.layer_name === selectedLayer,
    );
    const map: Record<number, { knn: boolean; kfn: boolean }> = {};

    rows.forEach((d) => {
      if (!map[d.cluster_id]) map[d.cluster_id] = { knn: false, kfn: false };
      const hasData = d.mcs_smiles != null || d.mcs_smarts != null;
      if (d.group === "knn") map[d.cluster_id].knn = hasData;
      if (d.group === "kfn") map[d.cluster_id].kfn = hasData;
    });

    return map;
  }, [substructures, selectedModel, selectedLayer]);

  const clusterInstanceIds = useMemo(() => {
    if (!selectedNode) return [];

    const ids = new Set<string>();
    activeSankeyRows.forEach((r) => {
      const fromModel = r.from_model ?? null;
      const toModel = r.to_model ?? null;

      const fromMatch =
        r.from_cluster === selectedNode.cluster &&
        r.from_layer === selectedNode.layer &&
        (fromModel == null || fromModel === selectedNode.model);

      const toMatch =
        r.to_cluster === selectedNode.cluster &&
        r.to_layer === selectedNode.layer &&
        (toModel == null || toModel === selectedNode.model);

      if (fromMatch || toMatch) {
        ids.add(String(r.real_id));
      }
    });

    return Array.from(ids).sort((a, b) => {
      const na = Number(a);
      const nb = Number(b);
      if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
      return a.localeCompare(b);
    });
  }, [activeSankeyRows, selectedNode]);

  const clusterTotalPages = useMemo(() => {
    if (clusterInstanceIds.length === 0) return 1;
    return Math.max(1, Math.ceil(clusterInstanceIds.length / Math.max(1, clusterPageSize)));
  }, [clusterInstanceIds.length, clusterPageSize]);

  const clusterPagedIds = useMemo(() => {
    const size = Math.max(1, clusterPageSize);
    const start = clusterPage * size;
    return clusterInstanceIds.slice(start, start + size);
  }, [clusterInstanceIds, clusterPage, clusterPageSize]);

  useEffect(() => {
    setClusterPage(0);
  }, [selectedNode?.model, selectedNode?.layer, selectedNode?.cluster]);

  useEffect(() => {
    setClusterPage((prev) => Math.min(prev, Math.max(0, clusterTotalPages - 1)));
  }, [clusterTotalPages]);

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

  function clusterBadgeColor(cluster: number) {
    const len = CLUSTER_COLORS.length || 1;
    const idx = ((cluster % len) + len) % len;
    return CLUSTER_COLORS[idx];
  }

  function showHoverPreviewAt(src: string, anchor?: DOMRect) {
    if (!showLargeHoverPreview) return;
    const W = largeHoverPreviewSize;
    const H = largeHoverPreviewSize;
    const PAD = 8;
    if (!anchor) {
      setHoverPreview({ src, left: window.innerWidth - W - 16, top: window.innerHeight - H - 16 });
      return;
    }
    let left = anchor.left + (anchor.width - W) / 2;
    let top = anchor.top + (anchor.height - H) / 2;
    left = Math.max(PAD, Math.min(left, window.innerWidth - W - PAD));
    top = Math.max(PAD, Math.min(top, window.innerHeight - H - PAD));
    setHoverPreview({ src, left, top });
  }

  const LEFT_COL_RATIO_IN_WORKSPACE = 45 / 55;
  const MID_COL_RATIO_IN_WORKSPACE = 10 / 55;
  const TOP_ROW_RATIO_IN_WORKSPACE = 40 / 60;
  const MID_ROW_RATIO_IN_WORKSPACE = 20 / 60;
  const effectiveWorkspaceWidthPct = workspaceWidthPct;
  const leftColPct = effectiveWorkspaceWidthPct * LEFT_COL_RATIO_IN_WORKSPACE;
  const midColPct = effectiveWorkspaceWidthPct * MID_COL_RATIO_IN_WORKSPACE;
  const rightColPct = 100 - effectiveWorkspaceWidthPct;
  const topRowPct = workspaceHeightPct * TOP_ROW_RATIO_IN_WORKSPACE;
  const midRowPct = workspaceHeightPct * MID_ROW_RATIO_IN_WORKSPACE;
  const bottomRowPct = 100 - workspaceHeightPct;

  function onWorkspaceResizeStart(event: ReactMouseEvent<HTMLDivElement>) {
    event.preventDefault();
    const onMove = (ev: MouseEvent) => {
      const host = layoutGridRef.current;
      if (!host) return;
      const rect = host.getBoundingClientRect();
      const relX = ((ev.clientX - rect.left) / rect.width) * 100;
      const relY = ((ev.clientY - rect.top) / rect.height) * 100;
      const nextW = Math.min(80, Math.max(40, relX));
      const nextH = Math.min(85, Math.max(45, relY));
      setWorkspaceWidthPct((prev) => (Math.abs(prev - nextW) < 0.1 ? prev : nextW));
      setWorkspaceHeightPct((prev) => (Math.abs(prev - nextH) < 0.1 ? prev : nextH));
    };

    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  function onBottomSplitResizeStart(event: ReactMouseEvent<HTMLDivElement>) {
    event.preventDefault();
    const onMove = (ev: MouseEvent) => {
      const host = bottomSplitHostRef.current;
      if (!host) return;
      const rect = host.getBoundingClientRect();
      const relX = ((ev.clientX - rect.left) / rect.width) * 100;
      const next = Math.min(80, Math.max(20, relX));
      setBottomSplitPct(next);
    };

    const onUp = () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }

  return (
    <div className="flex min-h-screen w-full">
      <main className="flex min-h-screen w-full">
        <div
          ref={layoutGridRef}
          className="h-screen w-full grid relative transition-all duration-300 ease-out"
          style={{
            gridTemplateColumns: `${leftColPct}% ${midColPct}% ${rightColPct}%`,
            gridTemplateRows: `${topRowPct}% ${midRowPct}% ${bottomRowPct}%`,
          }}
        >
          <div className="p-1 box-border" style={{ gridColumn: "1 / 2", gridRow: "1 / 2" }}>
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
                activeCell={cellSelected ? { model: selectedModel, layer: selectedLayer } : null}
                onCellSelect={(active) => {
                  setCellSelected(active);
                  if (active) {
                    // Entering 2D-cell focus should clear cluster-level selection and recolor by new cell.
                    setSelectedNode(null);
                    setSelectedIds(null);
                    setHoverLock(false);
                    setSuppressScatterCluster(false);
                  }
                }}
                onSwapAxes={(swap) => setSwapAxes(swap)}
                externalTransform={syncLinkedSankeyViews ? sankeyTransform : undefined}
                onTransform={(t) => handleSankeyTransformSync(t)}
              />
            </div>
          </div>
          <div className="p-1 box-border" style={{ gridColumn: "2 / 3", gridRow: "1 / 2" }}>
            <div
              id="sankeyVertical"
              className="w-full h-full border border-zinc-400 min-h-0 relative"
            >
              <div className="absolute top-1 right-2 text-xs text-zinc-600 bg-white/80 px-1 rounded">
                {swapAxes ? selectedModel : selectedLayer}
              </div>
              <SankeyVertical
                dataPath={swapAxes ? sankeyPath : sankeyPathVertical}
                mode={swapAxes ? "layer" : "model"}
                layerKey={selectedLayer}
                modelKey={selectedModel}
                syncTransform={syncLinkedSankeyViews ? syncTransformVertical : undefined}
                onSyncTransform={(t) => {
                  if (!syncLinkedSankeyViews) return;
                  handleSankeyTransformSync({ x: sankeyTransform.x, y: t.y, scale: t.scale });
                }}
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
          <div className="p-1 box-border" style={{ gridColumn: "3 / 4", gridRow: "1 / 4" }}>
          
            <div
              id="instanceview"
              className="w-full h-full border border-zinc-400 bg-white"
            >
              <TopkNearestPanel
                rows={topkNearestRows}
                selectedModel={selectedModel}
                selectedLayer={selectedLayer}
                selectedCluster={selectedNode?.cluster ?? null}
                hoveredId={hoveredId}
                onToggleHoverId={(id) => setHoveredId((prev) => (prev === id ? null : id))}
                onSelectHoverId={(id) => toggleLockedHighlight(id)}
                onHoverIdChange={(id) => setHoverFromMouse(id)}
                resolveImageSrc={resolveImageSrc}
                resolveSubstructureImage={resolveSubstructureImage}
                substructureByCluster={substructureByCluster}
                clusterBadgeColor={clusterBadgeColor}
                onClusterSelect={(cluster) => {
                  setSelectedNode({ model: selectedModel, layer: selectedLayer, cluster });
                  setSelectedIds(null);
                  setHoveredId(null);
                  setHoverLock(false);
                  setSuppressScatterCluster(false);
                  setCellSelected(true);
                }}
                onImagePreviewStart={(src, anchor) => showHoverPreviewAt(src, anchor)}
                onImagePreviewEnd={() => setHoverPreview(null)}
                hideTopk={workspaceWidthPct >= 79.5}
              />
            </div>
          </div>
          <div className="p-1 box-border" style={{ gridColumn: "1 / 2", gridRow: "2 / 3" }}>
            <div
              id="sankeyHorizontal"
              className="w-full h-full border border-zinc-400 relative"
            >
              <div className="absolute top-1 left-2 text-xs text-zinc-600 bg-white/80 px-1 rounded">
                {swapAxes ? selectedLayer : selectedModel}
              </div>
              <SankeyHorizontal
                dataPath={swapAxes ? sankeyPathVertical : sankeyPath}
                mode={swapAxes ? "model" : "layer"}
                modelKey={selectedModel}
                layerKey={selectedLayer}
                syncTransform={syncLinkedSankeyViews ? syncTransformHorizontal : undefined}
                onSyncTransform={(t) => {
                  if (!syncLinkedSankeyViews) return;
                  handleSankeyTransformSync({ x: t.x, y: sankeyTransform.y, scale: t.scale });
                }}
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
          <div className="p-1 box-border" style={{ gridColumn: "2 / 3", gridRow: "2 / 3" }}>
            <div
              id="editor"
              className="w-full h-full border border-zinc-400 relative"
            >
              <StructureEditorPanel />
            </div>
          </div>

          <div className="p-1 box-border" style={{ gridColumn: "1 / 3", gridRow: "3 / 4" }}>
            <div ref={bottomSplitHostRef} className="w-full h-full flex min-w-0">
              <div className="h-full min-w-0" style={{ width: `${bottomSplitPct}%` }}>
                <div
                  id="scatterplot"
                  className="w-full h-full border border-zinc-400 bg-white"
                >
                  <ScatterPlot
                    dataPath={scatterPath}
                    activeCluster={suppressScatterCluster ? null : selectedNode?.cluster ?? null}
                    activeIds={selectedIds}
                    highlightId={hoveredId}
                    modelKey={selectedModel}
                    layerKey={selectedLayer}
                    onPointHoverIdChange={(id) => setHoverFromMouse(id)}
                    onPointSelect={(sel) => {
                      if (!sel.model || !sel.layer) return;
                      setSelectedNode({ model: sel.model, layer: sel.layer, cluster: sel.cluster });
                      setSelectedIds([sel.real_id]);
                      setHoveredId(String(sel.real_id));
                      setHoverLock(true);
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
                    onClusterSelect={(sel) => {
                      if (!sel.model || !sel.layer) return;
                      setSelectedNode({ model: sel.model, layer: sel.layer, cluster: sel.cluster });
                      setSelectedIds(sel.real_ids);
                      setHoveredId(null);
                      setHoverLock(false);
                      setSuppressScatterCluster(false);
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
                      setHoverLock(false);
                      setSuppressScatterCluster(false);
                    }}
                  />
                </div>
              </div>
              <div
                className="w-4 shrink-0 cursor-col-resize group flex items-center justify-center bg-zinc-200/70 hover:bg-zinc-400/70 transition-colors"
                onMouseDown={onBottomSplitResizeStart}
                title="Resize Scatterplot / Cluster Instances"
              >
                <div className="h-16 w-3 rounded bg-white/85 border border-zinc-300 shadow-sm flex flex-col items-center justify-center gap-0.5">
                  <svg width="10" height="10" viewBox="0 0 16 16" aria-hidden="true" className="text-zinc-500 group-hover:text-zinc-700">
                    <path d="M6 3L3 8l3 5M10 3l3 5-3 5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <div className="grid grid-cols-2 gap-[2px]">
                    <span className="w-[2px] h-[2px] bg-zinc-500 rounded-full" />
                    <span className="w-[2px] h-[2px] bg-zinc-500 rounded-full" />
                    <span className="w-[2px] h-[2px] bg-zinc-500 rounded-full" />
                    <span className="w-[2px] h-[2px] bg-zinc-500 rounded-full" />
                  </div>
                </div>
              </div>
              <div className="h-full min-w-0 flex-1">
                <div
                  id="bottomPanel1"
                  className="w-full h-full border border-zinc-400 bg-white"
                >
                  <div className="w-full h-full p-2 box-border overflow-hidden">
                    {!selectedNode && (
                      <div className="text-sm text-zinc-500">Select a cluster to view all instances.</div>
                    )}
                    {selectedNode && (
                      <div className="h-full min-h-0 flex flex-col gap-2">
                        <div className="text-sm font-semibold text-zinc-700">Cluster Members ({clusterInstanceIds.length})</div>
                        {clusterInstanceIds.length === 0 && (
                          <div className="text-sm text-zinc-500">
                            No instances found for {selectedNode.model} / {selectedNode.layer} / cluster {selectedNode.cluster}.
                          </div>
                        )}
                        {clusterInstanceIds.length > 0 && (
                          <>
                            <div className="flex items-center justify-between text-xs text-zinc-600">
                              <div>
                                Page {clusterPage + 1} / {clusterTotalPages}
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="flex items-center gap-1">
                                  <label htmlFor="cluster-page-size">Per page</label>
                                  <select
                                    id="cluster-page-size"
                                    className="px-1.5 py-0.5 border border-zinc-300 rounded bg-white"
                                    value={clusterPageSize}
                                    onChange={(e) => {
                                      setClusterPage(0);
                                      setClusterPageSize(Number(e.target.value));
                                    }}
                                  >
                                    {[1, 5, 10, 20, 50].map((size) => (
                                      <option key={`cluster-size-${size}`} value={size}>
                                        {size}
                                      </option>
                                    ))}
                                  </select>
                                </div>
                                <button
                                  type="button"
                                  className="px-2 py-0.5 border border-zinc-300 rounded disabled:opacity-40"
                                  disabled={clusterPage <= 0}
                                  onClick={() => setClusterPage((p) => Math.max(0, p - 1))}
                                >
                                  Prev
                                </button>
                                <button
                                  type="button"
                                  className="px-2 py-0.5 border border-zinc-300 rounded disabled:opacity-40"
                                  disabled={clusterPage >= clusterTotalPages - 1}
                                  onClick={() => setClusterPage((p) => Math.min(clusterTotalPages - 1, p + 1))}
                                >
                                  Next
                                </button>
                              </div>
                            </div>
                            <div className="flex-1 min-h-0 overflow-auto pr-1">
                              <div className="grid grid-cols-5 gap-1">
                                {clusterPagedIds.map((id) => (
                                  <div
                                    key={`instance-${id}`}
                                    className="relative w-full max-w-[180px] aspect-square mx-auto"
                                    onMouseEnter={(e) => {
                                      setHoverFromMouse(String(id));
                                      showHoverPreviewAt(
                                        resolveImageSrc(id, "ligand"),
                                        (e.currentTarget as HTMLDivElement).getBoundingClientRect(),
                                      );
                                    }}
                                    onMouseLeave={() => {
                                      setHoverFromMouse(null);
                                      setHoverPreview(null);
                                    }}
                                    onClick={() => toggleLockedHighlight(String(id))}
                                  >
                                    <img
                                      src={resolveImageSrc(id, "ligand")}
                                      alt={`ligand ${id}`}
                                      className="w-full h-full object-contain bg-white border border-zinc-200"
                                      onLoad={(e) => {
                                        const target = e.currentTarget as HTMLImageElement;
                                        target.style.display = "block";
                                      }}
                                      onError={(e) => {
                                        const target = e.currentTarget as HTMLImageElement;
                                        target.style.display = "none";
                                      }}
                                    />
                                    {selectedNode && (
                                      <div
                                        className="absolute top-0.5 right-0.5 text-[10px] leading-4 px-1.5 py-0.5 rounded-[2px] border border-white/80 text-white font-semibold shadow-sm"
                                        style={{ backgroundColor: clusterBadgeColor(selectedNode.cluster) }}
                                      >
                                        {selectedNode.cluster}
                                      </div>
                                    )}
                                    <div
                                      className={
                                        `absolute top-0.5 left-0.5 text-xs px-1 rounded bg-white/85 cursor-pointer ` +
                                        (hoveredId === String(id) ? "text-amber-700 font-semibold" : "text-zinc-700")
                                      }
                                      onMouseEnter={() => setHoverFromMouse(String(id))}
                                      onMouseLeave={() => setHoverFromMouse(null)}
                                      onClick={() => setHoveredId((prev) => (prev === String(id) ? null : String(id)))}
                                    >
                                      {id}
                                    </div>
                                  </div>
                                ))}
                              </div>
                            </div>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
          <button
            type="button"
            className="absolute z-30 w-6 h-6 rounded-md border border-zinc-400 bg-white/95 hover:bg-zinc-100 shadow cursor-se-resize flex items-center justify-center"
            style={{ left: `calc(${effectiveWorkspaceWidthPct}% - 12px)`, top: `calc(${workspaceHeightPct}% - 12px)` }}
            onMouseDown={onWorkspaceResizeStart}
            title="Resize Sankey Workspace"
            aria-label="Resize Sankey Workspace"
          >
            <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
              <path d="M6 2H2v4M10 14h4v-4M2 6l4-4M14 10l-4 4" stroke="currentColor" strokeWidth="1.8" fill="none" strokeLinecap="round" />
            </svg>
          </button>
          <div
            className="absolute left-0 top-0 bottom-0 z-30 flex items-start"
            onMouseEnter={() => setIsControlPanelOpen(true)}
            onMouseLeave={() => setIsControlPanelOpen(false)}
          >
            <button
              type="button"
              className="mt-3 h-9 w-6 rounded-r border border-l-0 border-zinc-400 bg-white text-xs text-zinc-600 shadow pointer-events-auto"
              onClick={() => setIsControlPanelOpen((prev) => !prev)}
              title={isControlPanelOpen ? "Hide control panel" : "Show control panel"}
              aria-label={isControlPanelOpen ? "Hide control panel" : "Show control panel"}
            >
              {isControlPanelOpen ? "<" : ">"}
            </button>
            <div
              className={
                "mt-3 h-[280px] bg-white pointer-events-auto overflow-hidden transition-all duration-300 ease-out " +
                (isControlPanelOpen
                  ? "w-56 opacity-100 translate-x-0 border border-zinc-400 shadow-lg"
                  : "w-0 opacity-0 -translate-x-2 border border-transparent shadow-none")
              }
            >
              <div className={"w-56 h-full transition-opacity duration-200 " + (isControlPanelOpen ? "opacity-100" : "opacity-0")}>
                <SyncControlPanel
                  syncOn={syncLinkedSankeyViews}
                  onToggleSync={toggleSyncLinkedViews}
                  largePreviewOn={showLargeHoverPreview}
                  onToggleLargePreview={() => setShowLargeHoverPreview((prev) => !prev)}
                  largePreviewSize={largeHoverPreviewSize}
                  onLargePreviewSizeChange={setLargeHoverPreviewSize}
                />
              </div>
            </div>
          </div>
        </div>
        {hoverPreview && (
          <div
            className="fixed z-[10000] pointer-events-none bg-white/95 border border-zinc-300 rounded shadow-xl p-2"
            style={{ left: hoverPreview.left, top: hoverPreview.top }}
          >
            <img
              src={hoverPreview.src}
              alt="large preview"
              style={{ width: `${largeHoverPreviewSize}px`, height: `${largeHoverPreviewSize}px` }}
              className="object-contain bg-white"
            />
          </div>
        )}
        
      </main>
    </div>
  );
}
