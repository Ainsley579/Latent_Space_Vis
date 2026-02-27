"use client";
import { useEffect, useMemo, useRef, useState, type MouseEvent as ReactMouseEvent } from "react";
import SankeyView from "./component/SankeyView";
import SankeyHorizontal from "./component/SankeyHorizontal";
import SankeyVertical from "./component/SankeyVertical";
import ScatterPlot from "./component/ScatterPlot";
import TopkNearestPanel from "./component/TopkNearestPanel";
import SmilesStructure from "./component/SmilesStructure";
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

type ScatterPoint = {
  real_id: string | number;
  x: number;
  y: number;
  cluster_id: number;
};

type GeneratedSankeyPayload = {
  dataset: DatasetKey;
  clustering: ClusteringKey;
  allSankey: any;
  modelSamples: Record<string, SankeySampleRow[]>;
  verticalSamplesByLayer: Record<string, SankeySampleRow[]>;
};

type GeneratedRuntimePayload = {
  dataset: DatasetKey;
  clustering: ClusteringKey;
  runId: string;
  topkRepresentatives: any[];
  clusterSortedMembers: any[];
  substructures: any[];
  scatterClusterByCell: Record<string, Array<{ real_id: string | number; cluster_id: number }>>;
};
type HoverPreviewState =
  | { kind: "image"; src: string; left: number; top: number }
  | { kind: "smiles"; smiles: string; left: number; top: number };

type DatasetKey = "dimenet" | "chemprop" | "chemeleon";
type ClusteringKey = "pgmeans" | "agglomerative" | "hdbscan";
const DATASET_OPTIONS: DatasetKey[] = ["dimenet", "chemprop", "chemeleon"];
const DATASET_OPTIONS_UI: DatasetKey[] = ["chemprop", "chemeleon"];
const CLUSTERING_OPTIONS: Array<{ value: ClusteringKey; label: string }> = [
  { value: "pgmeans", label: "PG Means" },
  { value: "agglomerative", label: "Agglomerative" },
  { value: "hdbscan", label: "HDBSCAN" },
];
const DATASET_DEFAULTS: Record<DatasetKey, { model: string; layer: string }> = {
  dimenet: { model: "dimenet_noaug", layer: "emb" },
  chemprop: { model: "0", layer: "fingerprint" },
  chemeleon: { model: "0", layer: "fingerprint" },
};

function sanitizePathKey(value: string) {
  return value.replace(/[^a-zA-Z0-9_-]/g, "_");
}

export default function Home() {
  const [selectedDataset, setSelectedDataset] = useState<DatasetKey>("chemprop");
  const [selectedClustering, setSelectedClustering] = useState<ClusteringKey>("pgmeans");
  const [generatedSankeyPayloads, setGeneratedSankeyPayloads] = useState<Record<string, GeneratedSankeyPayload>>({});
  const [generatedRuntimePayloads, setGeneratedRuntimePayloads] = useState<Record<string, GeneratedRuntimePayload>>({});
  const generatedSankeyBlobUrlsRef = useRef<Map<string, string>>(new Map());
  const datasetBaseFor = (
    dataset: DatasetKey,
    clustering: ClusteringKey = selectedClustering
  ) => {
    if (clustering === "pgmeans") return `/data/${dataset}`;
    return `/data/${dataset}/clustering/${clustering}`;
  };
  const datasetBase = datasetBaseFor(selectedDataset, selectedClustering);
  const assetBase = `/data/${selectedDataset}`;
  const revokeGeneratedSankeyBlobUrls = () => {
    generatedSankeyBlobUrlsRef.current.forEach((u) => URL.revokeObjectURL(u));
    generatedSankeyBlobUrlsRef.current.clear();
  };
  const buildGeneratedSankeyBlobUrl = (cacheKey: string, payload: unknown): string => {
    const cached = generatedSankeyBlobUrlsRef.current.get(cacheKey);
    if (cached) return cached;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(payload ?? [])], { type: "application/json" })
    );
    generatedSankeyBlobUrlsRef.current.set(cacheKey, url);
    return url;
  };
  const runtimeKeyFor = (dataset: DatasetKey, clustering: ClusteringKey) => `${dataset}::${clustering}`;
  const currentRuntimeKey = runtimeKeyFor(selectedDataset, selectedClustering);
  const generatedSankeyPayload = generatedSankeyPayloads[currentRuntimeKey] ?? null;
  const generatedRuntimePayload = generatedRuntimePayloads[currentRuntimeKey] ?? null;
  const generatedMatchesCurrent = !!generatedSankeyPayload;
  const generatedRuntimeMatchesCurrent = !!generatedRuntimePayload;
  const apiBase = (process.env.NEXT_PUBLIC_FLASK_API_BASE || "http://127.0.0.1:5001").replace(/\/+$/, "");
  const withVersion = (url: string, version: number = sankeyRefreshKey) => {
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}v=${version}`;
  };
  const buildAllSankeyPath = (
    dataset: DatasetKey = selectedDataset,
    clustering: ClusteringKey = selectedClustering,
    version: number = sankeyRefreshKey
  ) => {
    const payload = generatedSankeyPayloads[runtimeKeyFor(dataset, clustering)] ?? null;
    if (
      payload &&
      payload.dataset === dataset &&
      payload.clustering === clustering
    ) {
      return buildGeneratedSankeyBlobUrl(`${dataset}:${clustering}:all_sankey`, payload.allSankey);
    }
    return withVersion(`${datasetBaseFor(dataset, clustering)}/sankey/all_sankey.json`, version);
  };
  const buildSankeyModelPath = (
    model: string,
    dataset: DatasetKey = selectedDataset,
    clustering: ClusteringKey = selectedClustering,
    version: number = sankeyRefreshKey
  ) => {
    const payload = generatedSankeyPayloads[runtimeKeyFor(dataset, clustering)] ?? null;
    if (
      payload &&
      payload.dataset === dataset &&
      payload.clustering === clustering
    ) {
      const rows = payload.modelSamples?.[model] ?? [];
      return buildGeneratedSankeyBlobUrl(`${dataset}:${clustering}:model:${model}`, rows);
    }
    return withVersion(
      `${datasetBaseFor(dataset, clustering)}/sankey/${sanitizePathKey(model)}_sankey_samples.json`,
      version
    );
  };
  const buildVerticalPath = (
    layer: string,
    dataset: DatasetKey = selectedDataset,
    clustering: ClusteringKey = selectedClustering,
    version: number = sankeyRefreshKey
  ) => {
    const payload = generatedSankeyPayloads[runtimeKeyFor(dataset, clustering)] ?? null;
    if (
      payload &&
      payload.dataset === dataset &&
      payload.clustering === clustering
    ) {
      const rows = payload.verticalSamplesByLayer?.[layer] ?? [];
      return buildGeneratedSankeyBlobUrl(`${dataset}:${clustering}:vertical:${layer}`, rows);
    }
    return withVersion(
      `${datasetBaseFor(dataset, clustering)}/sankey/vertical_model_samples_${sanitizePathKey(layer)}.json`,
      version
    );
  };
  const buildScatterPaths = (
    model: string,
    layer: string,
    projection: "pca" | "mds" | "umap" | "tsne",
    dataset: DatasetKey = selectedDataset,
    clustering: ClusteringKey = selectedClustering,
    version: number = dataRefreshKey
  ) => {
    const base = `${model}_${layer}`;
    const xyFile = `${base}_${projection}_xy.json`;
    const clusterFile = `${base}_${projection}_cluster.json`;
    const xyUrl = `/data/${dataset}/scatterplot/${xyFile}`;
    const clusterBase = datasetBaseFor(dataset, clustering);
    const runtimePayload = generatedRuntimePayloads[runtimeKeyFor(dataset, clustering)] ?? null;
    const cellKey = `${model}__${layer}`;
    const overrideRows =
      runtimePayload?.scatterClusterByCell?.[cellKey] &&
      Array.isArray(runtimePayload.scatterClusterByCell[cellKey])
        ? runtimePayload.scatterClusterByCell[cellKey]
        : null;
    const clusterUrl = overrideRows
      ? buildGeneratedSankeyBlobUrl(`${dataset}:${clustering}:scatter_cluster:${cellKey}`, overrideRows)
      : `${clusterBase}/scatterplot/${clusterFile}`;
    const sepXY = xyUrl.includes("?") ? "&" : "?";
    const sepCluster = clusterUrl.startsWith("blob:") ? "" : (clusterUrl.includes("?") ? "&" : "?");
    return {
      xy: `${xyUrl}${sepXY}v=${version}`,
      cluster: clusterUrl.startsWith("blob:") ? clusterUrl : `${clusterUrl}${sepCluster}v=${version}`,
    };
  };
  const syncSankeyPaths = (
    model: string,
    layer: string,
    dataset: DatasetKey = selectedDataset,
    clustering: ClusteringKey = selectedClustering,
    version: number = sankeyRefreshKey
  ) => {
    setSankeyPath(buildSankeyModelPath(model, dataset, clustering, version));
    setSankeyPathVertical(buildVerticalPath(layer, dataset, clustering, version));
  };

  const [sankeyPath, setSankeyPath] = useState("/data/chemprop/sankey/0_sankey_samples.json");
  const [sankeyPathVertical, setSankeyPathVertical] = useState("/data/chemprop/sankey/vertical_model_samples_fingerprint.json");
  const [sankeyTransform, setSankeyTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [selectedLayer, setSelectedLayer] = useState("fingerprint");
  const [selectedModel, setSelectedModel] = useState("0");
  const [scatterProjection, setScatterProjection] = useState<"pca" | "mds" | "umap" | "tsne">("pca");
  const [scatterPaths, setScatterPaths] = useState({
    xy: "/data/chemprop/scatterplot/0_fingerprint_pca_xy.json",
    cluster: "/data/chemprop/scatterplot/0_fingerprint_pca_cluster.json",
  });
  const scatterOverrideUrlRef = useRef<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<{ model: string; layer: string; cluster: number } | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[] | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [hoverLock, setHoverLock] = useState(false);
  const [hoverPreview, setHoverPreview] = useState<HoverPreviewState | null>(null);
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
  const [sankeyRefreshKey, setSankeyRefreshKey] = useState(0);
  const [dataRefreshKey, setDataRefreshKey] = useState(0);
  const [isGeneratingRange, setIsGeneratingRange] = useState(false);
  const [generateStatusText, setGenerateStatusText] = useState("");
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
  const [clusterSortedMembers, setClusterSortedMembers] = useState<
    {
      layer_name: string;
      model_type: string;
      cluster_id: number;
      sorted_real_ids: Array<string | number>;
      count?: number;
    }[]
  >([]);
  const [syncLinkedSankeyViews, setSyncLinkedSankeyViews] = useState(false);
  const [showLargeHoverPreview, setShowLargeHoverPreview] = useState(true);
  const [largeHoverPreviewSize, setLargeHoverPreviewSize] = useState(320);
  const [isControlPanelOpen, setIsControlPanelOpen] = useState(false);
  const [clusterPage, setClusterPage] = useState(0);
  const [clusterPageSize, setClusterPageSize] = useState(10);
  const [bookmarkedIds, setBookmarkedIds] = useState<Set<string>>(new Set());
  const [smilesByRealId, setSmilesByRealId] = useState<Record<string, string>>({});
  const [searchResultIds, setSearchResultIds] = useState<string[]>([]);
  const [searchSmiles, setSearchSmiles] = useState<string>("");
  const [showSearchResultsInMembers, setShowSearchResultsInMembers] = useState(false);
  const [lastSearchCount, setLastSearchCount] = useState<number | null>(null);
  const [bottomSplitPct, setBottomSplitPct] = useState(35);
  const [workspaceWidthPct, setWorkspaceWidthPct] = useState(80);
  const [workspaceHeightPct, setWorkspaceHeightPct] = useState(70);
  const layoutGridRef = useRef<HTMLDivElement | null>(null);
  const bottomSplitHostRef = useRef<HTMLDivElement | null>(null);
  const syncRafRef = useRef<number | null>(null);
  const pendingTransformRef = useRef<{ x: number; y: number; scale: number } | null>(null);
  const lastSyncedTransformRef = useRef<{ x: number; y: number; scale: number }>({ x: 0, y: 0, scale: 1 });
  const syncSourceRef = useRef<{ source: "view" | "vertical" | "horizontal"; ts: number } | null>(null);
  useEffect(() => {
    return () => {
      revokeGeneratedSankeyBlobUrls();
      if (scatterOverrideUrlRef.current) {
        URL.revokeObjectURL(scatterOverrideUrlRef.current);
        scatterOverrideUrlRef.current = null;
      }
    };
  }, []);
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

  function handleSankeyTransformSync(
    next: { x: number; y: number; scale: number },
    source: "view" | "vertical" | "horizontal"
  ) {
    const now = typeof performance !== "undefined" ? performance.now() : Date.now();
    const lock = syncSourceRef.current;
    const SOURCE_LOCK_MS = 140;
    if (lock && lock.source !== source && now - lock.ts < SOURCE_LOCK_MS) return;
    syncSourceRef.current = { source, ts: now };

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

  function mergeAndSyncSankeyTransform(
    partial: Partial<{ x: number; y: number; scale: number }>,
    source: "view" | "vertical" | "horizontal"
  ) {
    const base = pendingTransformRef.current ?? lastSyncedTransformRef.current;
    handleSankeyTransformSync(
      {
        x: partial.x ?? base.x,
        y: partial.y ?? base.y,
        scale: partial.scale ?? base.scale,
      },
      source
    );
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

  function toggleBookmark(id: string) {
    setBookmarkedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function exportBookmarkedIdsCsv() {
    if (bookmarkedIds.size === 0) return;
    const ids = Array.from(bookmarkedIds).sort((a, b) => {
      const na = Number(a);
      const nb = Number(b);
      if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
      return a.localeCompare(b);
    });
    const quoteCsv = (value: string) => `"${value.replace(/"/g, '""')}"`;
    const rows = [
      "real_id,smiles",
      ...ids.map((id) => {
        const raw = String(id);
        const smiles =
          smilesByRealId[raw] ??
          smilesByRealId[raw.trim()] ??
          smilesByRealId[raw.toLowerCase()] ??
          smilesByRealId[raw.trim().toLowerCase()] ??
          "";
        return `${quoteCsv(raw)},${quoteCsv(String(smiles))}`;
      }),
    ];
    const csv = rows.join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    const dateTag = new Date().toISOString().slice(0, 10);
    a.href = url;
    a.download = `${selectedDataset}_${selectedClustering}_bookmarked_ids_${dateTag}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  useEffect(() => {
    fetch(`/data/${selectedDataset}/real_id_to_smiles.json`)
      .then((res) => {
        if (res.status === 404) return {};
        if (!res.ok) throw new Error("Failed to load real_id_to_smiles.json");
        return res.json();
      })
      .then((json) => {
        const out: Record<string, string> = {};
        if (Array.isArray(json)) {
          json.forEach((row) => {
            if (!row || typeof row !== "object") return;
            const rid = String((row as any).real_id ?? "").trim();
            if (!rid) return;
            const smiles = typeof (row as any).smiles === "string" ? (row as any).smiles : "";
            out[rid] = smiles;
            out[rid.trim()] = smiles;
            out[rid.toLowerCase()] = smiles;
            out[rid.trim().toLowerCase()] = smiles;
          });
        } else if (json && typeof json === "object") {
          Object.entries(json as Record<string, unknown>).forEach(([k, v]) => {
            const key = String(k);
            const smiles = typeof v === "string" ? v : "";
            out[key] = smiles;
            out[key.trim()] = smiles;
            out[key.toLowerCase()] = smiles;
            out[key.trim().toLowerCase()] = smiles;
          });
        }
        setSmilesByRealId(out);
      })
      .catch((err) => {
        console.error(err);
        setSmilesByRealId({});
      });
  }, [selectedDataset]);

  useEffect(() => {
    setSearchResultIds([]);
    setSearchSmiles("");
    setShowSearchResultsInMembers(false);
    setLastSearchCount(null);
  }, [selectedDataset, selectedClustering]);

  async function generateFromRange(minValue: number, maxValue: number) {
    if (isGeneratingRange) return;
    setIsGeneratingRange(true);
    setGenerateStatusText(`Generating PGMeans [${minValue}, ${maxValue}] ...`);
    try {
      const res = await fetch(`${apiBase}/generate-range`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset: selectedDataset,
          clustering: selectedClustering,
          include_payload: true,
          persist: false,
          min: minValue,
          max: maxValue,
        }),
      });
      if (!res.ok) {
        throw new Error(`Generate API failed: ${res.status}`);
      }
      const json = await res.json();
      const casePayload = json?.payload?.[selectedDataset];
      if (!casePayload) {
        throw new Error("Generate API missing payload for selected dataset");
      }
      revokeGeneratedSankeyBlobUrls();
      const key = runtimeKeyFor(selectedDataset, selectedClustering);
      setGeneratedSankeyPayloads((prev) => ({
        ...prev,
        [key]: {
        dataset: selectedDataset,
        clustering: selectedClustering,
        allSankey: casePayload.all_sankey ?? { epoch: {}, layer: {} },
        modelSamples: (casePayload.model_samples ?? {}) as Record<string, SankeySampleRow[]>,
        verticalSamplesByLayer: (casePayload.vertical_samples_by_layer ?? {}) as Record<string, SankeySampleRow[]>,
        },
      }));
      const runId = String(json?.run_id ?? "");
      if (runId) {
        setGeneratedRuntimePayloads((prev) => ({
          ...prev,
          [key]: {
            dataset: selectedDataset,
            clustering: selectedClustering,
            runId,
            topkRepresentatives: Array.isArray(casePayload.topk_representatives)
              ? casePayload.topk_representatives
              : [],
            clusterSortedMembers: Array.isArray(casePayload.cluster_sorted_members)
              ? casePayload.cluster_sorted_members
              : [],
            substructures: Array.isArray(casePayload.substructures)
              ? casePayload.substructures
              : [],
            scatterClusterByCell:
              casePayload?.scatter_cluster_by_cell &&
              typeof casePayload.scatter_cluster_by_cell === "object"
                ? casePayload.scatter_cluster_by_cell
                : {},
          },
        }));
      }
      setTopkData(
        Array.isArray(casePayload.topk_representatives) ? casePayload.topk_representatives : []
      );
      setClusterSortedMembers(
        Array.isArray(casePayload.cluster_sorted_members) ? casePayload.cluster_sorted_members : []
      );
      setSubstructures(Array.isArray(casePayload.substructures) ? casePayload.substructures : []);
      setSankeyRefreshKey((k: number) => k + 1);
      setGenerateStatusText("Generate finished (backend temp payload, no static file writes).");
    } catch (err) {
      console.error("Failed to generate payload:", err);
      setGenerateStatusText(`Generate failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsGeneratingRange(false);
    }
  }

  async function generateAgglomerative(config: {
    k: number | null;
    linkage: "ward" | "complete" | "average";
  }) {
    if (isGeneratingRange) return;
    setIsGeneratingRange(true);
    setGenerateStatusText(
      `Re-running agglomerative for ALL cells (k=${config.k == null ? "auto" : config.k}, linkage=${config.linkage}) ...`
    );
    try {
      const kMin = config.k == null ? 2 : config.k;
      const kMax = config.k == null ? 15 : config.k;
      const res = await fetch(`${apiBase}/generate-range`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset: selectedDataset,
          min: kMin,
          max: kMax,
          clustering: "agglomerative",
          k: config.k,
          linkage: config.linkage,
          include_payload: true,
          persist: false,
        }),
      });
      if (!res.ok) {
        throw new Error(`Generate API failed: ${res.status}`);
      }
      const json = await res.json();
      const casePayload = json?.payload?.[selectedDataset];
      if (!casePayload) {
        throw new Error("Generate API missing payload for selected dataset");
      }
      revokeGeneratedSankeyBlobUrls();
      const key = runtimeKeyFor(selectedDataset, "agglomerative");
      setGeneratedSankeyPayloads((prev) => ({
        ...prev,
        [key]: {
        dataset: selectedDataset,
        clustering: "agglomerative",
        allSankey: casePayload.all_sankey ?? { epoch: {}, layer: {} },
        modelSamples: (casePayload.model_samples ?? {}) as Record<string, SankeySampleRow[]>,
        verticalSamplesByLayer: (casePayload.vertical_samples_by_layer ?? {}) as Record<string, SankeySampleRow[]>,
        },
      }));
      const runId = String(json?.run_id ?? "");
      if (runId) {
        setGeneratedRuntimePayloads((prev) => ({
          ...prev,
          [key]: {
            dataset: selectedDataset,
            clustering: "agglomerative",
            runId,
            topkRepresentatives: Array.isArray(casePayload.topk_representatives)
              ? casePayload.topk_representatives
              : [],
            clusterSortedMembers: Array.isArray(casePayload.cluster_sorted_members)
              ? casePayload.cluster_sorted_members
              : [],
            substructures: Array.isArray(casePayload.substructures)
              ? casePayload.substructures
              : [],
            scatterClusterByCell:
              casePayload?.scatter_cluster_by_cell &&
              typeof casePayload.scatter_cluster_by_cell === "object"
                ? casePayload.scatter_cluster_by_cell
                : {},
          },
        }));
      }
      setTopkData(
        Array.isArray(casePayload.topk_representatives) ? casePayload.topk_representatives : []
      );
      setClusterSortedMembers(
        Array.isArray(casePayload.cluster_sorted_members) ? casePayload.cluster_sorted_members : []
      );
      setSubstructures(Array.isArray(casePayload.substructures) ? casePayload.substructures : []);
      if (scatterOverrideUrlRef.current) {
        URL.revokeObjectURL(scatterOverrideUrlRef.current);
        scatterOverrideUrlRef.current = null;
      }
      setSankeyRefreshKey((k) => k + 1);
      setGenerateStatusText(
        `Agglomerative all-cell regenerate finished: k=${config.k == null ? "auto(2-15)" : config.k}, linkage=${config.linkage}.`
      );
    } catch (err) {
      console.error("Failed to regenerate all cells for agglomerative:", err);
      setGenerateStatusText(`Agg regenerate failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsGeneratingRange(false);
    }
  }

  useEffect(() => {
    if (scatterOverrideUrlRef.current) {
      URL.revokeObjectURL(scatterOverrideUrlRef.current);
      scatterOverrideUrlRef.current = null;
    }
    setScatterPaths(buildScatterPaths(selectedModel, selectedLayer, scatterProjection));
  }, [datasetBase, selectedModel, selectedLayer, scatterProjection, generatedRuntimePayloads]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setSwapAxes(params.get("swap") === "1");
  }, []);

  useEffect(() => {
    fetch(buildAllSankeyPath(selectedDataset, selectedClustering))
      .then((res) => {
        if (!res.ok) throw new Error("Failed to load all_sankey.json");
        return res.json();
      })
      .then((data) => {
        const models = Object.keys(data?.epoch || {});
        const layers = Object.keys(data?.layer || {});
        if (!models.length || !layers.length) return;

        const preferredModel =
          selectedDataset === "dimenet"
            ? (models.find((m) => m === "dimenet_noaug") ?? models[0])
            : (models.find((m) => m === "0") ?? models[0]);
        const preferredLayer =
          selectedDataset === "dimenet"
            ? (layers.find((l) => l === "emb") ?? layers[0])
            : (layers.find((l) => l === "fingerprint") ?? layers[0]);

        setSelectedModel(preferredModel);
        setSelectedLayer(preferredLayer);
        setSankeyPath(buildSankeyModelPath(preferredModel));
        setSankeyPathVertical(buildVerticalPath(preferredLayer));
        setSelectedNode(null);
        setSelectedIds(null);
        setActiveSankeyRows([]);
      })
      .catch((err) => {
        console.error(err);
      });
  }, [datasetBase, selectedDataset, selectedClustering, sankeyRefreshKey, generatedSankeyPayload]);

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
  }, [activeSankeyPath, sankeyRefreshKey]);

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
    if (generatedRuntimeMatchesCurrent) {
      setTopkData(generatedRuntimePayload?.topkRepresentatives ?? []);
      return;
    }
    fetch(`${datasetBase}/topk/cluster_representatives.json`)
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
  }, [datasetBase, dataRefreshKey, generatedRuntimeMatchesCurrent, generatedRuntimePayload]);

  useEffect(() => {
    if (generatedRuntimeMatchesCurrent) {
      setClusterSortedMembers(generatedRuntimePayload?.clusterSortedMembers ?? []);
      return;
    }
    fetch(`${datasetBase}/topk/cluster_sorted_members.json`)
      .then((res) => {
        if (res.status === 404) return [];
        if (!res.ok) throw new Error("Failed to load cluster sorted members");
        return res.json();
      })
      .then((json) => {
        setClusterSortedMembers(Array.isArray(json) ? json : []);
      })
      .catch((err) => {
        console.error(err);
        setClusterSortedMembers([]);
      });
  }, [datasetBase, dataRefreshKey, generatedRuntimeMatchesCurrent, generatedRuntimePayload]);

  useEffect(() => {
    if (generatedRuntimeMatchesCurrent) {
      setSubstructures(generatedRuntimePayload?.substructures ?? []);
      return;
    }
    fetch(`${datasetBase}/substructures/cluster_substructures.json`)
      .then((res) => {
        // Some datasets may not provide substructure files yet; treat missing file as empty.
        if (res.status === 404) return [];
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
  }, [datasetBase, dataRefreshKey, generatedRuntimeMatchesCurrent, generatedRuntimePayload]);

  const topkNearestRows = useMemo(() => {
    return topkData
      .filter((d) => d.model_type === selectedModel && d.layer_name === selectedLayer)
      .sort((a, b) => a.cluster_id - b.cluster_id);
  }, [topkData, selectedModel, selectedLayer]);


  const substructureByCluster = useMemo(() => {
    const rows = substructures.filter(
      (d) => d.model_type === selectedModel && d.layer_name === selectedLayer,
    );
    const map: Record<number, { knn: boolean; kfn: boolean; knnSmiles: string | null; knnSmarts: string | null }> = {};

    rows.forEach((d) => {
      if (!map[d.cluster_id]) map[d.cluster_id] = { knn: false, kfn: false, knnSmiles: null, knnSmarts: null };
      const hasData = d.mcs_smiles != null || d.mcs_smarts != null;
      if (d.group === "knn") {
        map[d.cluster_id].knn = hasData;
        map[d.cluster_id].knnSmiles = d.mcs_smiles;
        map[d.cluster_id].knnSmarts = d.mcs_smarts;
      }
      if (d.group === "kfn") map[d.cluster_id].kfn = hasData;
    });

    return map;
  }, [substructures, selectedModel, selectedLayer]);

  const sortedMembersByClusterKey = useMemo(() => {
    const map = new Map<string, string[]>();
    clusterSortedMembers.forEach((d) => {
      const key = `${d.model_type}__${d.layer_name}__${d.cluster_id}`;
      if (Array.isArray(d.sorted_real_ids) && d.sorted_real_ids.length > 0) {
        map.set(
          key,
          d.sorted_real_ids.map((id) => String(id)),
        );
      }
    });
    return map;
  }, [clusterSortedMembers]);

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

    const key = `${selectedNode.model}__${selectedNode.layer}__${selectedNode.cluster}`;
    const sorted = sortedMembersByClusterKey.get(key);
    if (sorted && sorted.length > 0) {
      const ordered = sorted.filter((id) => ids.size === 0 || ids.has(id));
      if (ordered.length > 0) {
        const orderedSet = new Set(ordered);
        const extras = Array.from(ids).filter((id) => !orderedSet.has(id));
        extras.sort((a, b) => {
          const na = Number(a);
          const nb = Number(b);
          if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
          return a.localeCompare(b);
        });
        return [...ordered, ...extras];
      }
    }

    return Array.from(ids).sort((a, b) => {
      const na = Number(a);
      const nb = Number(b);
      if (!Number.isNaN(na) && !Number.isNaN(nb)) return na - nb;
      return a.localeCompare(b);
    });
  }, [activeSankeyRows, selectedNode, sortedMembersByClusterKey]);

  const displayedMemberIds = useMemo(() => {
    if (showSearchResultsInMembers) return searchResultIds;
    return clusterInstanceIds;
  }, [showSearchResultsInMembers, searchResultIds, clusterInstanceIds]);

  const clusterTotalPages = useMemo(() => {
    if (displayedMemberIds.length === 0) return 1;
    return Math.max(1, Math.ceil(displayedMemberIds.length / Math.max(1, clusterPageSize)));
  }, [displayedMemberIds.length, clusterPageSize]);

  const clusterPagedIds = useMemo(() => {
    const size = Math.max(1, clusterPageSize);
    const start = clusterPage * size;
    return displayedMemberIds.slice(start, start + size);
  }, [displayedMemberIds, clusterPage, clusterPageSize]);

  useEffect(() => {
    setClusterPage(0);
  }, [selectedNode?.model, selectedNode?.layer, selectedNode?.cluster]);

  useEffect(() => {
    setClusterPage(0);
  }, [showSearchResultsInMembers, searchResultIds]);

  useEffect(() => {
    setClusterPage((prev) => Math.min(prev, Math.max(0, clusterTotalPages - 1)));
  }, [clusterTotalPages]);

  function resolveImageSrc(realId: string | number, imageType: string) {
    const key = sanitizePathKey(String(realId).trim());
    return `${assetBase}/molecule_images/${key}_${imageType}.png`;
  }

  function getSmilesById(realId: string | number): string | null {
    const key = String(realId).trim().toLowerCase();
    return smilesByRealId[key] ?? null;
  }

  function clusterBadgeColor(cluster: number) {
    const len = CLUSTER_COLORS.length || 1;
    const idx = ((cluster % len) + len) % len;
    return CLUSTER_COLORS[idx];
  }

  function getPreviewAnchorPosition(anchor?: DOMRect) {
    const W = largeHoverPreviewSize;
    const H = largeHoverPreviewSize;
    const PAD = 8;
    if (!anchor) {
      return { left: window.innerWidth - W - 16, top: window.innerHeight - H - 16 };
    }
    let left = anchor.left + (anchor.width - W) / 2;
    let top = anchor.top + (anchor.height - H) / 2;
    left = Math.max(PAD, Math.min(left, window.innerWidth - W - PAD));
    top = Math.max(PAD, Math.min(top, window.innerHeight - H - PAD));
    return { left, top };
  }

  function showHoverPreviewAt(src: string, anchor?: DOMRect) {
    if (!showLargeHoverPreview) return;
    const { left, top } = getPreviewAnchorPosition(anchor);
    setHoverPreview({ kind: "image", src, left, top });
  }

  function showSmilesPreviewAt(smiles: string | null, anchor?: DOMRect) {
    if (!showLargeHoverPreview) return;
    if (!smiles || !smiles.trim()) {
      setHoverPreview(null);
      return;
    }
    const { left, top } = getPreviewAnchorPosition(anchor);
    setHoverPreview({ kind: "smiles", smiles, left, top });
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

  function onWorkspaceResizeStart(event: ReactMouseEvent<HTMLElement>) {
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

  function onBottomSplitResizeStart(event: ReactMouseEvent<HTMLElement>) {
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

  const nowTs = typeof performance !== "undefined" ? performance.now() : Date.now();
  const viewIsSyncSource =
    syncSourceRef.current?.source === "view" &&
    nowTs - (syncSourceRef.current?.ts ?? 0) < 180;
  const sankeyViewExternalTransform =
    syncLinkedSankeyViews && !viewIsSyncSource ? sankeyTransform : undefined;

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
                key={`sankeyview-${selectedDataset}-${selectedClustering}-${sankeyRefreshKey}`}
                dataPath={buildAllSankeyPath(selectedDataset, selectedClustering)}
                metricsPath={`${datasetBase}/bic_aic/avg_fmi_cspa_by_layer.json`}
                transitionStatsPath={`${datasetBase}/clustering_metrics/layer_transition_stats.json`}
                onModelSelect={(model) => {
                  setSankeyPath(buildSankeyModelPath(model));
                  setSelectedModel(model);
                  setSelectedNode(null);
                  setSelectedIds(null);
                }}
                onLayerSelect={(layer) => {
                  setSelectedLayer(layer);
                  setSankeyPathVertical(buildVerticalPath(layer));
                  setSelectedNode(null);
                  setSelectedIds(null);
                }}
                onNodeSelect={(sel) => {
                  toggleNode(sel);
                  setSelectedModel(sel.model);
                  setSelectedLayer(sel.layer);
                  syncSankeyPaths(sel.model, sel.layer);
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
                externalTransform={sankeyViewExternalTransform}
                onTransform={(t) => handleSankeyTransformSync(t, "view")}
              />
            </div>
          </div>
          <div className="p-1 box-border" style={{ gridColumn: "2 / 3", gridRow: "1 / 2" }}>
            <div
              id="sankeyVertical"
              className="w-full h-full border border-zinc-400 min-h-0 relative"
            >
              <div className="absolute top-1 right-2 z-30 pointer-events-none text-xs text-zinc-600 bg-white/90 px-1 rounded shadow-sm">
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
                  mergeAndSyncSankeyTransform({ y: t.y, scale: t.scale }, "vertical");
                }}
                activeNodeKey={
                  selectedNode
                    ? swapAxes
                      ? `${selectedNode.layer}_${selectedNode.cluster}`
                      : `${selectedNode.model}_${selectedNode.cluster}`
                    : null
                }
                highlightId={hoveredId}
                highlightIds={showSearchResultsInMembers ? searchResultIds : null}
                onNodeSelect={(sel) => {
                  if (!sel.model || sel.layer == null) return;
                  toggleNode({ model: sel.model, layer: sel.layer ?? selectedLayer, cluster: sel.cluster, ids: sel.ids });
                  setSelectedModel(sel.model);
                  setSelectedLayer(sel.layer ?? selectedLayer);
                  syncSankeyPaths(sel.model, sel.layer ?? selectedLayer);
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
                getSmilesById={getSmilesById}
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
                hideTopk={workspaceWidthPct >= 79.5}
                largePreviewOn={showLargeHoverPreview}
                onSmilesPreviewStart={(smiles, anchor) => showSmilesPreviewAt(smiles, anchor)}
                onSmilesPreviewEnd={() => setHoverPreview(null)}
                bookmarkedIds={bookmarkedIds}
                onToggleBookmark={toggleBookmark}
              />
            </div>
          </div>
          <div className="p-1 box-border" style={{ gridColumn: "1 / 2", gridRow: "2 / 3" }}>
            <div
              id="sankeyHorizontal"
              className="w-full h-full border border-zinc-400 relative"
            >
              <div className="absolute top-1 left-2 z-30 pointer-events-none text-xs text-zinc-600 bg-white/90 px-1 rounded shadow-sm">
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
                  mergeAndSyncSankeyTransform({ x: t.x, scale: t.scale }, "horizontal");
                }}
                activeNodeKey={
                  selectedNode
                    ? swapAxes
                      ? `${selectedNode.model}_${selectedNode.cluster}`
                      : `${selectedNode.layer}_${selectedNode.cluster}`
                    : null
                }
                highlightId={hoveredId}
                highlightIds={showSearchResultsInMembers ? searchResultIds : null}
                onNodeSelect={(sel) => {
                  if (!sel.layer || !sel.model) return;
                  toggleNode({ model: sel.model, layer: sel.layer, cluster: sel.cluster, ids: sel.ids });
                  setSelectedModel(sel.model);
                  setSelectedLayer(sel.layer);
                  syncSankeyPaths(sel.model, sel.layer);
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
              <StructureEditorPanel
                dataset={selectedDataset}
                onSearchResultIds={(ids, smiles) => {
                  const uniq = Array.from(new Set(ids.map((x) => String(x))));
                  setSearchResultIds(uniq);
                  setSearchSmiles(smiles);
                  setLastSearchCount(uniq.length);
                  setShowSearchResultsInMembers(uniq.length > 0);
                  setSelectedIds(uniq.length > 0 ? uniq : null);
                  setHoveredId(null);
                  setHoverLock(false);
                }}
              />
            </div>
          </div>

          <div className="p-1 box-border" style={{ gridColumn: "1 / 3", gridRow: "3 / 4" }}>
            <div ref={bottomSplitHostRef} className="w-full h-full flex min-w-0">
              <div className="h-full min-w-0 flex flex-col border border-zinc-400" style={{ width: `${bottomSplitPct}%` }}>
                <div className="flex items-center gap-2 py-1 px-1 border-b border-zinc-200 bg-zinc-50 shrink-0">
                  <span className="text-xs font-medium text-zinc-600">Projection:</span>
                  <select
                    value={scatterProjection}
                    onChange={(e) => setScatterProjection(e.target.value as typeof scatterProjection)}
                    className="text-xs border border-zinc-300 rounded px-2 py-1 bg-white text-zinc-800"
                  >
                    <option value="pca">PCA</option>
                    <option value="mds">MDS</option>
                    <option value="umap">UMAP</option>
                    <option value="tsne">t-SNE</option>
                  </select>
                </div>
                <div
                  id="scatterplot"
                  className="flex-1 min-h-0 w-full bg-white"
                >
                  <ScatterPlot
                    dataPathXY={scatterPaths.xy}
                    dataPathCluster={scatterPaths.cluster}
                    activeCluster={suppressScatterCluster ? null : selectedNode?.cluster ?? null}
                    activeIds={showSearchResultsInMembers ? searchResultIds : selectedIds}
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
                      syncSankeyPaths(sel.model, sel.layer);
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
                      syncSankeyPaths(sel.model, sel.layer);
                    }}
                    onClearSelection={() => {
                      setSelectedNode(null);
                      setSelectedIds(null);
                      setHoveredId(null);
                      setHoverLock(false);
                      setSuppressScatterCluster(false);
                    }}
                    resolveImageSrc={resolveImageSrc}
                    getSmilesById={getSmilesById}
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
                    {!selectedNode && !showSearchResultsInMembers && (
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex flex-col gap-1">
                          <div className="text-sm text-zinc-500">Select a cluster to view all instances.</div>
                          {lastSearchCount != null && (
                            <div className="text-xs text-zinc-500">
                              Search result: {lastSearchCount} ids{searchSmiles ? ` for ${searchSmiles}` : ""}.
                            </div>
                          )}
                        </div>
                        {searchResultIds.length > 0 && (
                          <button
                            type="button"
                            className="px-2 py-1 rounded border text-xs border-zinc-300 bg-white text-zinc-700"
                            title={searchSmiles ? `SMILES: ${searchSmiles}` : "Show Search Results"}
                            onClick={() => {
                              setShowSearchResultsInMembers(true);
                              setSelectedIds(searchResultIds);
                              setHoveredId(null);
                              setHoverLock(false);
                            }}
                          >
                            Show Search Results ({searchResultIds.length})
                          </button>
                        )}
                      </div>
                    )}
                    {(selectedNode || showSearchResultsInMembers) && (
                      <div className="h-full min-h-0 flex flex-col gap-2">
                        <div className="flex items-center justify-between gap-2">
                          <div className="text-sm font-semibold text-zinc-700">
                            {showSearchResultsInMembers ? `Search Results (${displayedMemberIds.length})` : `Cluster Members (${displayedMemberIds.length})`}
                          </div>
                          {searchResultIds.length > 0 && (
                            <button
                              type="button"
                              className={
                                "px-2 py-1 rounded border text-xs " +
                                (showSearchResultsInMembers
                                  ? "border-emerald-500 bg-emerald-50 text-emerald-700"
                                  : "border-zinc-300 bg-white text-zinc-700")
                              }
                              title={searchSmiles ? `SMILES: ${searchSmiles}` : "Show Search Results"}
                              onClick={() => {
                                const next = !showSearchResultsInMembers;
                                setShowSearchResultsInMembers(next);
                                if (next) {
                                  setSelectedIds(searchResultIds);
                                  setHoveredId(null);
                                  setHoverLock(false);
                                }
                              }}
                            >
                              {showSearchResultsInMembers ? "Back To Cluster" : `Show Search Results (${searchResultIds.length})`}
                            </button>
                          )}
                        </div>
                        {displayedMemberIds.length === 0 && (
                          <div className="text-sm text-zinc-500">
                            {showSearchResultsInMembers
                              ? "No search results found for current SMILES."
                              : `No instances found for ${selectedNode?.model} / ${selectedNode?.layer} / cluster ${selectedNode?.cluster}.`}
                          </div>
                        )}
                        {displayedMemberIds.length > 0 && (
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
                                      showSmilesPreviewAt(
                                        getSmilesById(id),
                                        e.currentTarget.getBoundingClientRect()
                                      );
                                    }}
                                    onMouseLeave={() => {
                                      setHoverFromMouse(null);
                                      setHoverPreview(null);
                                    }}
                                    onClick={() => toggleLockedHighlight(String(id))}
                                  >
                                    <div
                                      className={
                                        "w-full h-full bg-white border p-1 " +
                                        (!showLargeHoverPreview && hoveredId === String(id)
                                          ? "border-red-500 border-2"
                                          : "border-zinc-200")
                                      }
                                    >
                                      <SmilesStructure smiles={getSmilesById(id)} className="w-full h-full" />
                                    </div>
                                    {selectedNode && !showSearchResultsInMembers && (
                                      <div
                                        className="absolute top-0.5 left-0.5 text-[10px] leading-4 px-1.5 py-0.5 rounded-[2px] border border-white/80 text-white font-semibold shadow-sm"
                                        style={{ backgroundColor: clusterBadgeColor(selectedNode.cluster) }}
                                      >
                                        {selectedNode.cluster}
                                      </div>
                                    )}
                                    <button
                                      type="button"
                                      className={
                                        "absolute top-0.5 right-0.5 z-[2] h-5 w-5 rounded bg-white/85 border shadow-sm flex items-center justify-center " +
                                        (bookmarkedIds.has(String(id))
                                          ? "border-amber-500 text-amber-500"
                                          : "border-zinc-300 text-zinc-500")
                                      }
                                      title={bookmarkedIds.has(String(id)) ? "Remove bookmark" : "Add bookmark"}
                                      aria-label={bookmarkedIds.has(String(id)) ? "Remove bookmark" : "Add bookmark"}
                                      onClick={(e) => {
                                        e.stopPropagation();
                                        toggleBookmark(String(id));
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
                                        `absolute bottom-0.5 left-0.5 text-xs px-1 rounded bg-white/85 cursor-pointer ` +
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
            onMouseLeave={() => setIsControlPanelOpen(false)}
          >
            <div
              className={
                "mt-12 h-[420px] bg-white pointer-events-auto overflow-hidden transition-all duration-300 ease-out " +
                (isControlPanelOpen
                  ? "w-56 opacity-100 translate-x-0 border border-zinc-400 shadow-lg"
                  : "w-0 opacity-0 -translate-x-2 border border-transparent shadow-none")
              }
            >
              <div className={"w-56 h-full transition-opacity duration-200 " + (isControlPanelOpen ? "opacity-100" : "opacity-0")}>
                <SyncControlPanel
                  dataset={selectedDataset}
                  onDatasetChange={(d) => {
                    const next = d as DatasetKey;
                    const defaults = DATASET_DEFAULTS[next];
                    const nextClustering = selectedClustering;
                    setSelectedDataset(next);
                    setSelectedModel(defaults.model);
                    setSelectedLayer(defaults.layer);
                    syncSankeyPaths(defaults.model, defaults.layer, next, nextClustering);
                    setScatterPaths(
                      buildScatterPaths(
                        defaults.model,
                        defaults.layer,
                        scatterProjection,
                        next,
                        nextClustering
                      )
                    );
                    setSelectedNode(null);
                    setSelectedIds(null);
                    setCellSelected(false);
                    setActiveSankeyRows([]);
                  }}
                  datasetOptions={DATASET_OPTIONS_UI}
                  clustering={selectedClustering}
                  clusteringOptions={CLUSTERING_OPTIONS}
                  onClusteringChange={(c) => {
                    const next = c as ClusteringKey;
                    const defaults = DATASET_DEFAULTS[selectedDataset];
                    setSelectedClustering(next);
                    setSelectedModel(defaults.model);
                    setSelectedLayer(defaults.layer);
                    syncSankeyPaths(defaults.model, defaults.layer, selectedDataset, next);
                    setScatterPaths(
                      buildScatterPaths(defaults.model, defaults.layer, scatterProjection, selectedDataset, next)
                    );
                    setSelectedNode(null);
                    setSelectedIds(null);
                    setCellSelected(false);
                    setActiveSankeyRows([]);
                  }}
                  syncOn={syncLinkedSankeyViews}
                  onToggleSync={toggleSyncLinkedViews}
                  largePreviewOn={showLargeHoverPreview}
                  onToggleLargePreview={() => setShowLargeHoverPreview((prev) => !prev)}
                  largePreviewSize={largeHoverPreviewSize}
                  onLargePreviewSizeChange={setLargeHoverPreviewSize}
                  onGenerateRange={generateFromRange}
                  onGenerateAgglomerative={generateAgglomerative}
                  generateEnabled={true}
                  generateBusy={isGeneratingRange}
                  generateMessage={generateStatusText}
                  onExportBookmarks={exportBookmarkedIdsCsv}
                  bookmarkCount={bookmarkedIds.size}
                />
              </div>
            </div>
            <button
              type="button"
              className="mt-12 h-9 w-6 rounded-r border border-l-0 border-zinc-400 bg-white text-xs text-zinc-600 shadow pointer-events-auto"
              onMouseEnter={() => setIsControlPanelOpen(true)}
              onClick={() => setIsControlPanelOpen((prev) => !prev)}
              title={isControlPanelOpen ? "Hide control panel" : "Show control panel"}
              aria-label={isControlPanelOpen ? "Hide control panel" : "Show control panel"}
            >
              {isControlPanelOpen ? "<" : ">"}
            </button>
          </div>
        </div>
        {hoverPreview && (
          <div
            className="fixed z-[10000] pointer-events-none bg-white/95 border border-zinc-300 rounded shadow-xl p-2"
            style={{ left: hoverPreview.left, top: hoverPreview.top }}
          >
            {hoverPreview.kind === "image" ? (
              <img
                src={hoverPreview.src}
                alt="large preview"
                style={{ width: `${largeHoverPreviewSize}px`, height: `${largeHoverPreviewSize}px` }}
                className="object-contain bg-white"
              />
            ) : (
              <div style={{ width: `${largeHoverPreviewSize}px`, height: `${largeHoverPreviewSize}px` }} className="bg-white">
                <SmilesStructure smiles={hoverPreview.smiles} className="w-full h-full" />
              </div>
            )}
          </div>
        )}
        
      </main>
    </div>
  );
}
