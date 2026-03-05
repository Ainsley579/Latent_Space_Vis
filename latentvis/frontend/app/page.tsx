"use client";
import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import SankeyView from "./component/SankeyView";
import SankeyHorizontal from "./component/SankeyHorizontal";
import SankeyVertical from "./component/SankeyVertical";
import ScatterPlot from "./component/ScatterPlot";
import TopkNearestPanel from "./component/TopkNearestPanel";
import SmilesStructure from "./component/SmilesStructure";
import StructureEditorPanel from "./component/StructureEditorPanel";
import SyncControlPanel from "./component/SyncControlPanel";
import ClusterContainmentHeatmap from "./component/ClusterContainmentHeatmap";
import { CLUSTER_COLORS } from "./theme/colors";
import { fetchJsonCached } from "./utils/fetchCache";

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
  dataSplit: DataSplitKey;
  allSankey: any;
  modelSamples: Record<string, SankeySampleRow[]>;
  verticalSamplesByLayer: Record<string, SankeySampleRow[]>;
};

type GeneratedRuntimePayload = {
  dataset: DatasetKey;
  clustering: ClusteringKey;
  dataSplit: DataSplitKey;
  runId: string;
  topkRepresentatives: any[];
  clusterSortedMembers: any[];
  substructures: any[];
  scatterClusterByCell: Record<string, Array<{ real_id: string | number; cluster_id: number }>>;
};

type UploadedUserClustering = {
  fileName: string;
  labelNames: string[];
  rows: Array<{ real_id: string; user_cluster_id: number }>;
};

type ClusterContainmentPayload = {
  user_cluster_ids: number[];
  our_cluster_ids: number[];
  matrix: number[][];
  entries: Array<{
    user_cluster_id: number;
    our_cluster_id: number;
    count: number;
    percentage_of_user_cluster: number;
  }>;
};

type CellMetricByCellPayload = {
  metrics_by_cell: Record<string, Record<string, number | null>>;
  metric_names: string[];
};

type KnnTopmOption = "5" | "10" | "20" | "all";
type HoverPreviewState =
  | { kind: "image"; src: string; left: number; top: number; source?: "topk" | "members" | "other" }
  | { kind: "smiles"; smiles: string; left: number; top: number; source?: "topk" | "members" | "other" };

type DatasetKey = "dimenet" | "chemprop" | "chemeleon";
type ClusteringKey = "pgmeans" | "agglomerative" | "hdbscan" | "knn";
type DataSplitKey = "all" | "test" | "train";
const DATASET_OPTIONS: DatasetKey[] = ["dimenet", "chemprop", "chemeleon"];
const DATASET_OPTIONS_UI: DatasetKey[] = ["chemprop", "chemeleon"];
const CLUSTERING_OPTIONS: Array<{ value: ClusteringKey; label: string }> = [
  { value: "pgmeans", label: "PG Means" },
  { value: "agglomerative", label: "Agglomerative" },
  { value: "hdbscan", label: "HDBSCAN" },
  { value: "knn", label: "KNN" },
];
const DATASET_DEFAULTS: Record<DatasetKey, { model: string; layer: string }> = {
  dimenet: { model: "dimenet_noaug", layer: "emb" },
  chemprop: { model: "0", layer: "fingerprint" },
  chemeleon: { model: "0", layer: "fingerprint" },
};
const SHOW_CONTAINMENT_VIEW = false;

function sanitizePathKey(value: string) {
  return value.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function parseSplitTaggedId(value: string | number): { split: "train" | "test" | null; realId: string } {
  const raw = String(value ?? "").trim();
  if (raw.startsWith("train::")) return { split: "train", realId: raw.slice("train::".length) };
  if (raw.startsWith("test::")) return { split: "test", realId: raw.slice("test::".length) };
  return { split: null, realId: raw };
}

export default function Home() {
  const [selectedDataset, setSelectedDataset] = useState<DatasetKey>("chemprop");
  const [selectedClustering, setSelectedClustering] = useState<ClusteringKey>("agglomerative");
  const [selectedDataSplit, setSelectedDataSplit] = useState<DataSplitKey>("all");
  const [generatedSankeyPayloads, setGeneratedSankeyPayloads] = useState<Record<string, GeneratedSankeyPayload>>({});
  const [generatedRuntimePayloads, setGeneratedRuntimePayloads] = useState<Record<string, GeneratedRuntimePayload>>({});
  const generatedSankeyPayloadsRef = useRef<Record<string, GeneratedSankeyPayload>>({});
  const preloadInFlightRef = useRef<Set<string>>(new Set());
  const generatedSankeyBlobUrlsRef = useRef<Map<string, string>>(new Map());
  const effectiveDataSplitForFiles = (dataSplit: DataSplitKey): "test" | "train" =>
    dataSplit === "all" ? "train" : dataSplit;
  const datasetRootFor = (
    dataset: DatasetKey,
    dataSplit: DataSplitKey = selectedDataSplit
  ) => `/data/${dataset}/${effectiveDataSplitForFiles(dataSplit)}`;
  const datasetBaseFor = (
    dataset: DatasetKey,
    clustering: ClusteringKey = selectedClustering,
    dataSplit: DataSplitKey = selectedDataSplit
  ) => {
    const root = datasetRootFor(dataset, dataSplit);
    if (clustering === "pgmeans") return root;
    return `${root}/clustering/${clustering}`;
  };
  const datasetBase = datasetBaseFor(selectedDataset, selectedClustering, selectedDataSplit);
  const assetBase = datasetRootFor(selectedDataset, selectedDataSplit);
  const shouldUseCurrentSuffix = (
    dataset: DatasetKey,
    clustering: ClusteringKey,
    dataSplit: DataSplitKey = selectedDataSplit
  ) => false;
  const sectionBaseFor = (
    section: "sankey" | "scatterplot" | "topk" | "substructures" | "clustering_metrics",
    dataset: DatasetKey = selectedDataset,
    clustering: ClusteringKey = selectedClustering,
    dataSplit: DataSplitKey = selectedDataSplit
  ) => {
    const suffix = shouldUseCurrentSuffix(dataset, clustering, dataSplit) ? "_current" : "";
    return `${datasetBaseFor(dataset, clustering, dataSplit)}/${section}${suffix}`;
  };
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
  const runtimeKeyFor = (dataset: DatasetKey, clustering: ClusteringKey, dataSplit: DataSplitKey) =>
    `${dataset}::${clustering}::${dataSplit}`;
  const currentRuntimeKey = runtimeKeyFor(selectedDataset, selectedClustering, selectedDataSplit);
  const generatedSankeyPayload = generatedSankeyPayloads[currentRuntimeKey] ?? null;
  const generatedRuntimePayload = generatedRuntimePayloads[currentRuntimeKey] ?? null;
  const generatedMatchesCurrent = !!generatedSankeyPayload;
  const generatedRuntimeMatchesCurrent = !!generatedRuntimePayload;
  const PRELOAD_SANKEY_COMPAT_ON_BOOT = true;
  useEffect(() => {
    generatedSankeyPayloadsRef.current = generatedSankeyPayloads;
  }, [generatedSankeyPayloads]);
  const apiBase = (process.env.NEXT_PUBLIC_FLASK_API_BASE || "http://127.0.0.1:5001").replace(/\/+$/, "");
  const buildDataFileUrl = useCallback(
    ({
      dataset = selectedDataset,
      clustering = selectedClustering,
      dataSplit = selectedDataSplit,
      section,
      file,
      datasetRoot = false,
    }: {
      dataset?: DatasetKey;
      clustering?: ClusteringKey;
      dataSplit?: DataSplitKey;
      section?: string;
      file: string;
      datasetRoot?: boolean;
    }) => {
      const p = new URLSearchParams();
      p.set("dataset", dataset);
      p.set("clustering", clustering);
      p.set("split", effectiveDataSplitForFiles(dataSplit));
      if (section) p.set("section", section);
      p.set("file", file);
      if (datasetRoot) p.set("dataset_root", "1");
      return `${apiBase}/data-file?${p.toString()}`;
    },
    [apiBase, selectedDataset, selectedClustering, selectedDataSplit]
  );
  const withVersion = (url: string, version: number = sankeyRefreshKey) => {
    const sep = url.includes("?") ? "&" : "?";
    return `${url}${sep}v=${version}`;
  };
  const buildAllSankeyPath = (
    dataset: DatasetKey = selectedDataset,
    clustering: ClusteringKey = selectedClustering,
    dataSplit: DataSplitKey = selectedDataSplit,
    version: number = sankeyRefreshKey
  ) => {
    const payload = generatedSankeyPayloads[runtimeKeyFor(dataset, clustering, dataSplit)] ?? null;
    if (
      payload &&
      payload.dataset === dataset &&
      payload.clustering === clustering &&
      payload.dataSplit === dataSplit
    ) {
      return buildGeneratedSankeyBlobUrl(`${dataset}:${clustering}:${dataSplit}:all_sankey`, payload.allSankey);
    }
    return withVersion(
      buildDataFileUrl({
        dataset,
        clustering,
        dataSplit,
        section: "sankey",
        file: "all_sankey.json",
      }),
      version
    );
  };
  const buildSankeyModelPath = (
    model: string,
    dataset: DatasetKey = selectedDataset,
    clustering: ClusteringKey = selectedClustering,
    dataSplit: DataSplitKey = selectedDataSplit,
    version: number = sankeyRefreshKey
  ) => {
    const payload = generatedSankeyPayloads[runtimeKeyFor(dataset, clustering, dataSplit)] ?? null;
    if (
      payload &&
      payload.dataset === dataset &&
      payload.clustering === clustering &&
      payload.dataSplit === dataSplit
    ) {
      const rows = payload.modelSamples?.[model] ?? [];
      return buildGeneratedSankeyBlobUrl(`${dataset}:${clustering}:${dataSplit}:model:${model}`, rows);
    }
    return withVersion(
      buildDataFileUrl({
        dataset,
        clustering,
        dataSplit,
        section: "sankey",
        file: `${sanitizePathKey(model)}_sankey_samples.json`,
      }),
      version
    );
  };
  const buildVerticalPath = (
    layer: string,
    dataset: DatasetKey = selectedDataset,
    clustering: ClusteringKey = selectedClustering,
    dataSplit: DataSplitKey = selectedDataSplit,
    version: number = sankeyRefreshKey
  ) => {
    const payload = generatedSankeyPayloads[runtimeKeyFor(dataset, clustering, dataSplit)] ?? null;
    if (
      payload &&
      payload.dataset === dataset &&
      payload.clustering === clustering &&
      payload.dataSplit === dataSplit
    ) {
      const rows = payload.verticalSamplesByLayer?.[layer] ?? [];
      return buildGeneratedSankeyBlobUrl(`${dataset}:${clustering}:${dataSplit}:vertical:${layer}`, rows);
    }
    return withVersion(
      buildDataFileUrl({
        dataset,
        clustering,
        dataSplit,
        section: "sankey",
        file: `vertical_model_samples_${sanitizePathKey(layer)}.json`,
      }),
      version
    );
  };
  const buildScatterPaths = (
    model: string,
    layer: string,
    projection: "pca" | "mds" | "umap" | "tsne",
    dataset: DatasetKey = selectedDataset,
    clustering: ClusteringKey = selectedClustering,
    dataSplit: DataSplitKey = selectedDataSplit,
    version: number = dataRefreshKey
  ) => {
    const base = `${model}_${layer}`;
    const xyFile = `${base}_${projection}_xy.json`;
    const clusterFile = `${base}_${projection}_cluster.json`;
    const xyUrl = buildDataFileUrl({
      dataset,
      clustering,
      dataSplit,
      section: "scatterplot",
      file: xyFile,
    });
    const runtimePayload = generatedRuntimePayloads[runtimeKeyFor(dataset, clustering, dataSplit)] ?? null;
    const cellKey = `${model}__${layer}`;
    const overrideRows =
      runtimePayload?.scatterClusterByCell?.[cellKey] &&
      Array.isArray(runtimePayload.scatterClusterByCell[cellKey])
        ? runtimePayload.scatterClusterByCell[cellKey]
        : null;
    const clusterUrl = overrideRows
      ? buildGeneratedSankeyBlobUrl(`${dataset}:${clustering}:${dataSplit}:scatter_cluster:${cellKey}`, overrideRows)
      : buildDataFileUrl({
          dataset,
          clustering,
          dataSplit,
          section: "scatterplot",
          file: clusterFile,
        });
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
    dataSplit: DataSplitKey = selectedDataSplit,
    version: number = sankeyRefreshKey
  ) => {
    setSankeyPath(buildSankeyModelPath(model, dataset, clustering, dataSplit, version));
    setSankeyPathVertical(buildVerticalPath(layer, dataset, clustering, dataSplit, version));
  };

  const [sankeyPath, setSankeyPath] = useState(
    withVersion(
      buildDataFileUrl({
        dataset: "chemprop",
        clustering: "agglomerative",
        dataSplit: "all",
        section: "sankey",
        file: "0_sankey_samples.json",
      }),
      0
    )
  );
  const [sankeyPathVertical, setSankeyPathVertical] = useState(
    withVersion(
      buildDataFileUrl({
        dataset: "chemprop",
        clustering: "agglomerative",
        dataSplit: "all",
        section: "sankey",
        file: "vertical_model_samples_fingerprint.json",
      }),
      0
    )
  );
  const [sankeyTransform, setSankeyTransform] = useState({ x: 0, y: 0, scale: 1 });
  const [selectedLayer, setSelectedLayer] = useState("fingerprint");
  const [selectedModel, setSelectedModel] = useState("0");
  const [scatterProjection, setScatterProjection] = useState<"pca" | "mds" | "umap" | "tsne">("pca");
  const [scatterPaths, setScatterPaths] = useState({
    xy: withVersion(
      buildDataFileUrl({
        dataset: "chemprop",
        clustering: "agglomerative",
        dataSplit: "all",
        section: "scatterplot",
        file: "0_fingerprint_pca_xy.json",
      }),
      0
    ),
    cluster: withVersion(
      buildDataFileUrl({
        dataset: "chemprop",
        clustering: "agglomerative",
        dataSplit: "all",
        section: "scatterplot",
        file: "0_fingerprint_pca_cluster.json",
      }),
      0
    ),
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
      real_ids: Array<string | number>;
      valid_molecules: number;
      missing_real_ids: Array<string | number>;
      invalid_smiles_real_ids: Array<string | number>;
      mcs_smarts: string | null;
      mcs_smiles: string | null;
    }[]
  >([]);
  const [knnTopm, setKnnTopm] = useState<KnnTopmOption>("5");
  const [isKnnTopmLoading, setIsKnnTopmLoading] = useState(false);
  const [cellSelected, setCellSelected] = useState(false);
  const [swapAxes, setSwapAxes] = useState(false);
  const [reverseEpochOrder, setReverseEpochOrder] = useState(true);
  const [visibleEpochs, setVisibleEpochs] = useState<string[] | null>(null);
  const handleVisibleEpochsChange = useCallback((epochs: string[]) => {
    setVisibleEpochs(epochs);
  }, []);
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
      top_k_closest_real_ids: Array<string | number>;
      top_k_farthest_real_ids: Array<string | number>;
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
  const [uploadedUserClustering, setUploadedUserClustering] = useState<UploadedUserClustering | null>(null);
  const [clusterContainment, setClusterContainment] = useState<ClusterContainmentPayload | null>(null);
  const [clusterContainmentLoading, setClusterContainmentLoading] = useState(false);
  const [clusterContainmentError, setClusterContainmentError] = useState<string | null>(null);
  const [cellMetricByCell, setCellMetricByCell] = useState<Record<string, Record<string, number | null>> | null>(null);
  const [cellMetricNames, setCellMetricNames] = useState<string[]>([]);
  const [selectedCellBgMetric, setSelectedCellBgMetric] = useState<string>("cluster_count");
  const memberListRef = useRef<HTMLDivElement | null>(null);

  const DEFAULT_WORKSPACE_WIDTH_PCT = 50; // right column default 50%
  const HALF_RIGHT_WORKSPACE_WIDTH_PCT = 83;
  const [workspaceWidthPct, setWorkspaceWidthPct] = useState(DEFAULT_WORKSPACE_WIDTH_PCT);
  const [workspaceHeightPct] = useState(80);
  const layoutGridRef = useRef<HTMLDivElement | null>(null);
  const rightTopRowRef = useRef<HTMLDivElement | null>(null);
  const [scatterMinWidthPx, setScatterMinWidthPx] = useState<number | null>(null);
  const [scatterWidthPx, setScatterWidthPx] = useState<number | null>(null);
  const scatterCornerResizeRef = useRef<{
    startX: number;
    startY: number;
    startWidth: number;
    startTopHeightPct: number;
  } | null>(null);
  const [isScatterCornerResizing, setIsScatterCornerResizing] = useState(false);
  const [rightTopHeightPct, setRightTopHeightPct] = useState(32);
  const cellBgMetricOptions = useMemo(
    () => {
      const metricLabelMap: Record<string, string> = {
        cluster_count: "Cluster Number",
        ari: "ARI",
        ami: "AMI",
        nmi: "NMI",
        homogeneity: "Homo",
        completeness: "Comp",
        v_measure: "V-Measure",
        fowlkes_mallows: "FMI",
      };
      const values = [
        "cluster_count",
        ...cellMetricNames.filter((m) => m && m !== "sample_count"),
      ];
      return values.map((v) => ({ value: v, label: metricLabelMap[v] ?? v }));
    },
    [cellMetricNames]
  );
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
    a.download = `${selectedDataset}_${selectedDataSplit}_${selectedClustering}_bookmarked_ids_${dateTag}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  useEffect(() => {
    const normalizeJsonInto = (json: unknown, out: Record<string, string>, splitTag: "train" | "test" | null) => {
      if (Array.isArray(json)) {
        json.forEach((row) => {
          if (!row || typeof row !== "object") return;
          const rid = String((row as any).real_id ?? "").trim();
          if (!rid) return;
          const smiles = typeof (row as any).smiles === "string" ? (row as any).smiles : "";
          if (!out[rid]) out[rid] = smiles;
          out[rid.trim()] = smiles;
          out[rid.toLowerCase()] = smiles;
          out[rid.trim().toLowerCase()] = smiles;
          if (splitTag) {
            const tagged = `${splitTag}::${rid}`;
            out[tagged] = smiles;
            out[tagged.toLowerCase()] = smiles;
          }
        });
        return;
      }
      if (json && typeof json === "object") {
        Object.entries(json as Record<string, unknown>).forEach(([k, v]) => {
          const key = String(k).trim();
          const smiles = typeof v === "string" ? v : "";
          if (!out[key]) out[key] = smiles;
          out[key.trim()] = smiles;
          out[key.toLowerCase()] = smiles;
          out[key.trim().toLowerCase()] = smiles;
          if (splitTag) {
            const tagged = `${splitTag}::${key}`;
            out[tagged] = smiles;
            out[tagged.toLowerCase()] = smiles;
          }
        });
      }
    };

    const fetchSmiles = async (split: "train" | "test") => {
      const url = buildDataFileUrl({
        dataset: selectedDataset,
        dataSplit: split,
        file: `${split}/real_id_to_smiles.json`,
        datasetRoot: true,
      });
      const res = await fetch(url);
      if (res.status === 404) return {};
      if (!res.ok) throw new Error("Failed to load real_id_to_smiles.json");
      return res.json();
    };

    const primarySplit = effectiveDataSplitForFiles(selectedDataSplit);
    fetchSmiles(primarySplit)
      .then(async (primaryJson) => {
        const out: Record<string, string> = {};
        if (selectedDataSplit === "all") {
          normalizeJsonInto(primaryJson, out, primarySplit);
          const otherSplit: "train" | "test" = primarySplit === "train" ? "test" : "train";
          try {
            const otherJson = await fetchSmiles(otherSplit);
            normalizeJsonInto(otherJson, out, otherSplit);
          } catch (err) {
            console.error("Failed to load secondary split smiles mapping:", err);
          }
        } else {
          normalizeJsonInto(primaryJson, out, null);
        }
        setSmilesByRealId(out);
      })
      .catch((err) => {
        console.error(err);
        setSmilesByRealId({});
      });
  }, [buildDataFileUrl, selectedDataset, selectedDataSplit]);

  useEffect(() => {
    setSearchResultIds([]);
    setSearchSmiles("");
    setShowSearchResultsInMembers(false);
    setLastSearchCount(null);
  }, [selectedDataset, selectedClustering, selectedDataSplit]);

  useEffect(() => {
    if (!uploadedUserClustering || uploadedUserClustering.rows.length === 0) {
      setClusterContainment(null);
      setClusterContainmentError(null);
      setClusterContainmentLoading(false);
      setCellMetricByCell(null);
      setCellMetricNames([]);
      return;
    }
    const controller = new AbortController();
    setClusterContainmentLoading(true);
    setClusterContainmentError(null);
    fetch(`${apiBase}/cluster-containment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        dataset: selectedDataset,
        epoch: selectedModel,
        layer: selectedLayer,
        clustering: selectedClustering,
        data_split: selectedDataSplit,
        use_current: generatedRuntimeMatchesCurrent,
        user_clustering: uploadedUserClustering.rows,
      }),
    })
      .then(async (res) => {
        const json = await res.json().catch(() => ({}));
        if (!res.ok || !json?.ok) {
          throw new Error(json?.error ? String(json.error) : `cluster-containment failed: ${res.status}`);
        }
        return json as ClusterContainmentPayload;
      })
      .then((json) => {
        setClusterContainment({
          user_cluster_ids: Array.isArray(json.user_cluster_ids) ? json.user_cluster_ids : [],
          our_cluster_ids: Array.isArray(json.our_cluster_ids) ? json.our_cluster_ids : [],
          matrix: Array.isArray(json.matrix) ? json.matrix : [],
          entries: Array.isArray(json.entries) ? json.entries : [],
        });
      })
      .catch((err) => {
        if ((err as Error)?.name === "AbortError") return;
        setClusterContainment(null);
        setClusterContainmentError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!controller.signal.aborted) setClusterContainmentLoading(false);
      });
    return () => controller.abort();
  }, [
    apiBase,
    uploadedUserClustering,
    selectedDataset,
    selectedClustering,
    selectedDataSplit,
    selectedModel,
    selectedLayer,
    generatedRuntimeMatchesCurrent,
  ]);

  useEffect(() => {
    const valid = new Set(cellBgMetricOptions.map((o) => o.value));
    if (!valid.has(selectedCellBgMetric)) {
      setSelectedCellBgMetric("cluster_count");
    }
  }, [cellBgMetricOptions, selectedCellBgMetric]);

  useEffect(() => {
    if (!uploadedUserClustering || uploadedUserClustering.rows.length === 0) {
      setCellMetricByCell(null);
      setCellMetricNames([]);
      return;
    }
    const controller = new AbortController();
    fetch(`${apiBase}/cell-cluster-metrics`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        dataset: selectedDataset,
        clustering: selectedClustering,
        data_split: selectedDataSplit,
        use_current: generatedRuntimeMatchesCurrent,
        user_clustering: uploadedUserClustering.rows,
      }),
    })
      .then(async (res) => {
        const json = await res.json().catch(() => ({}));
        if (!res.ok || !json?.ok) {
          throw new Error(json?.error ? String(json.error) : `cell-cluster-metrics failed: ${res.status}`);
        }
        return json as CellMetricByCellPayload;
      })
      .then((json) => {
        setCellMetricByCell(
          json && json.metrics_by_cell && typeof json.metrics_by_cell === "object"
            ? json.metrics_by_cell
            : {}
        );
        setCellMetricNames(Array.isArray(json.metric_names) ? json.metric_names : []);
      })
      .catch((err) => {
        if ((err as Error)?.name === "AbortError") return;
        console.error(err);
        setCellMetricByCell(null);
        setCellMetricNames([]);
      });
    return () => controller.abort();
  }, [
    apiBase,
    uploadedUserClustering,
    selectedDataset,
    selectedClustering,
    selectedDataSplit,
    generatedRuntimeMatchesCurrent,
  ]);

  function sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async function submitGenerateRangeJob(
    body: Record<string, unknown>,
    statusPrefix: string
  ) {
    const submitRes = await fetch(`${apiBase}/generate-range`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...body, async: true }),
    });
    if (!submitRes.ok) {
      throw new Error(`Generate submit failed: ${submitRes.status}`);
    }
    const submitJson = await submitRes.json();
    const jobId = String(submitJson?.job_id ?? "");
    if (!jobId) {
      throw new Error("Generate submit missing job_id");
    }

    const deadline = Date.now() + 70 * 60 * 1000;
    while (true) {
      const statusRes = await fetch(`${apiBase}/generate-range/jobs/${encodeURIComponent(jobId)}`);
      if (!statusRes.ok) {
        throw new Error(`Generate status failed: ${statusRes.status}`);
      }
      const statusJson = await statusRes.json();
      const status = String(statusJson?.status ?? "");
      const msg = typeof statusJson?.message === "string" ? statusJson.message : "";
      if (status === "succeeded") {
        if (statusJson?.result && typeof statusJson.result === "object") {
          return statusJson.result;
        }
        // Backward compatibility: some backends may return final payload at top-level.
        if (statusJson && typeof statusJson === "object") {
          return statusJson;
        }
        return null;
      }
      if (status === "failed") {
        const detail =
          statusJson?.error?.detail ??
          statusJson?.error?.error ??
          statusJson?.error ??
          "Generate job failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      setGenerateStatusText(msg ? `${statusPrefix} (${msg})` : `${statusPrefix} (${status})`);
      if (Date.now() > deadline) {
        throw new Error("Generate job polling timed out");
      }
      await sleep(2000);
    }
  }

  function applyGeneratedPayload(
    json: any,
    dataset: DatasetKey,
    clustering: ClusteringKey,
    dataSplit: DataSplitKey
  ) {
    const root = json?.result && typeof json.result === "object" ? json.result : json;
    const casePayload = root?.payload?.[dataset];
    if (!casePayload) {
      const payloadKeys =
        root?.payload && typeof root.payload === "object"
          ? Object.keys(root.payload)
          : [];
      throw new Error(
        `Generate API missing payload for selected dataset (${dataset}); payload keys: ${payloadKeys.join(",")}`
      );
    }

    revokeGeneratedSankeyBlobUrls();
    const key = runtimeKeyFor(dataset, clustering, dataSplit);
    setGeneratedSankeyPayloads((prev) => ({
      ...prev,
      [key]: {
        dataset,
        clustering,
        dataSplit,
        allSankey: casePayload.all_sankey ?? { epoch: {}, layer: {} },
        modelSamples: (casePayload.model_samples ?? {}) as Record<string, SankeySampleRow[]>,
        verticalSamplesByLayer: (casePayload.vertical_samples_by_layer ?? {}) as Record<
          string,
          SankeySampleRow[]
        >,
      },
    }));
    const runId = String(root?.run_id ?? `${Date.now()}`);
    setGeneratedRuntimePayloads((prev) => ({
      ...prev,
      [key]: {
        dataset,
        clustering,
        dataSplit,
        runId,
        topkRepresentatives: Array.isArray(casePayload.topk_representatives)
          ? casePayload.topk_representatives
          : [],
        clusterSortedMembers: Array.isArray(casePayload.cluster_sorted_members)
          ? casePayload.cluster_sorted_members
          : [],
        substructures: Array.isArray(casePayload.substructures) ? casePayload.substructures : [],
        scatterClusterByCell:
          casePayload?.scatter_cluster_by_cell &&
          typeof casePayload.scatter_cluster_by_cell === "object"
            ? casePayload.scatter_cluster_by_cell
            : {},
      },
    }));
    setTopkData(Array.isArray(casePayload.topk_representatives) ? casePayload.topk_representatives : []);
    setClusterSortedMembers(
      Array.isArray(casePayload.cluster_sorted_members) ? casePayload.cluster_sorted_members : []
    );
    setSubstructures(Array.isArray(casePayload.substructures) ? casePayload.substructures : []);
  }

  function applySankeyCompatPayload(
    payloadByDataset: any,
    dataset: DatasetKey,
    clustering: ClusteringKey,
    dataSplit: DataSplitKey
  ) {
    const casePayload = payloadByDataset?.[dataset];
    if (!casePayload || typeof casePayload !== "object") return;
    const key = runtimeKeyFor(dataset, clustering, dataSplit);
    setGeneratedSankeyPayloads((prev) => ({
      ...prev,
      [key]: {
        dataset,
        clustering,
        dataSplit,
        allSankey: (casePayload as any).all_sankey ?? { epoch: {}, layer: {} },
        modelSamples: ((casePayload as any).model_samples ?? {}) as Record<string, SankeySampleRow[]>,
        verticalSamplesByLayer: ((casePayload as any).vertical_samples_by_layer ?? {}) as Record<
          string,
          SankeySampleRow[]
        >,
      },
    }));
  }

  useEffect(() => {
    if (!PRELOAD_SANKEY_COMPAT_ON_BOOT) return;
    const controller = new AbortController();
    const preload = async (cacheSplit: DataSplitKey, requestSplit: "train" | "test") => {
      const key = runtimeKeyFor(selectedDataset, selectedClustering, cacheSplit);
      if (generatedSankeyPayloadsRef.current[key]) return;
      const inflightKey = `${key}::${requestSplit}`;
      if (preloadInFlightRef.current.has(inflightKey)) return;
      preloadInFlightRef.current.add(inflightKey);
      try {
        const res = await fetch(`${apiBase}/materialize-sankey-files`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            dataset: selectedDataset,
            clustering: selectedClustering,
            data_split: requestSplit,
            include_payload: true,
            write_files: false,
          }),
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`preload sankey failed: ${res.status}`);
        const json = await res.json();
        applySankeyCompatPayload(json?.payload, selectedDataset, selectedClustering, cacheSplit);
      } finally {
        preloadInFlightRef.current.delete(inflightKey);
      }
    };

    (async () => {
      try {
        if (selectedDataSplit === "all") {
          // Preload both train(primary) and test(secondary) payloads up front,
          // so horizontal/vertical sankey can switch without extra backend compute.
          await Promise.all([
            preload("all", "train"),
            preload("test", "test"),
          ]);
        } else if (selectedDataSplit === "train") {
          await preload("train", "train");
        } else {
          await preload("test", "test");
        }
      } catch (err) {
        if ((err as Error)?.name === "AbortError") return;
        console.error("Failed to preload sankey compat payload:", err);
      }
    })();

    return () => controller.abort();
  }, [apiBase, selectedDataset, selectedClustering, selectedDataSplit, PRELOAD_SANKEY_COMPAT_ON_BOOT]);

  async function generateFromRange(minValue: number, maxValue: number) {
    if (isGeneratingRange) return;
    const dataset = selectedDataset;
    const clustering = selectedClustering;
    const dataSplit = selectedDataSplit;
    setIsGeneratingRange(true);
    setGenerateStatusText(`Generating PGMeans [${minValue}, ${maxValue}] ...`);
    try {
      const json = await submitGenerateRangeJob(
        {
          dataset,
          clustering,
          data_split: dataSplit,
          include_payload: true,
          min: minValue,
          max: maxValue,
        },
        `Generating ${clustering} [${minValue}, ${maxValue}]`
      );
      applyGeneratedPayload(json, dataset, clustering, dataSplit);
      setSankeyRefreshKey((k: number) => k + 1);
      setGenerateStatusText("Generate finished.");
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
    const dataset = selectedDataset;
    const dataSplit = selectedDataSplit;
    setIsGeneratingRange(true);
    setGenerateStatusText(
      `Re-running agglomerative for ALL cells (k=${config.k == null ? "auto" : config.k}, linkage=${config.linkage}) ...`
    );
    try {
      const kMin = config.k == null ? 2 : config.k;
      const kMax = config.k == null ? 15 : config.k;
      const json = await submitGenerateRangeJob(
        {
          dataset,
          data_split: dataSplit,
          min: kMin,
          max: kMax,
          clustering: "agglomerative",
          k: config.k,
          linkage: config.linkage,
          include_payload: true,
        },
        `Re-running agglomerative (k=${config.k == null ? "auto" : config.k}, linkage=${config.linkage})`
      );
      applyGeneratedPayload(json, dataset, "agglomerative", dataSplit);
      if (scatterOverrideUrlRef.current) {
        URL.revokeObjectURL(scatterOverrideUrlRef.current);
        scatterOverrideUrlRef.current = null;
      }
      setSankeyRefreshKey((k) => k + 1);
      setGenerateStatusText(
        `Agglomerative regenerate finished (in-memory): k=${config.k == null ? "auto(2-15)" : config.k}, linkage=${config.linkage}.`
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
    fetch(buildAllSankeyPath(selectedDataset, selectedClustering, selectedDataSplit))
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
        setSankeyPath(buildSankeyModelPath(preferredModel, selectedDataset, selectedClustering, selectedDataSplit));
        setSankeyPathVertical(buildVerticalPath(preferredLayer, selectedDataset, selectedClustering, selectedDataSplit));
        setSelectedNode(null);
        setSelectedIds(null);
        setActiveSankeyRows([]);
      })
      .catch((err) => {
        console.error(err);
      });
  }, [datasetBase, selectedDataset, selectedClustering, selectedDataSplit, sankeyRefreshKey, generatedSankeyPayload]);

  useEffect(() => {
    if (!selectedNode) {
      setCellSelected(false);
    }
  }, [selectedNode]);

  const activeSankeyPath = useMemo(() => {
    return swapAxes ? sankeyPathVertical : sankeyPath;
  }, [swapAxes, sankeyPath, sankeyPathVertical]);
  const activeSankeySecondaryPath = useMemo(() => {
    if (selectedDataSplit !== "all") return null;
    return swapAxes
      ? buildVerticalPath(selectedLayer, selectedDataset, selectedClustering, "test")
      : buildSankeyModelPath(selectedModel, selectedDataset, selectedClustering, "test");
  }, [selectedDataSplit, swapAxes, selectedLayer, selectedModel, selectedDataset, selectedClustering, dataRefreshKey, sankeyRefreshKey, generatedSankeyPayloads]);

  useEffect(() => {
    const isBlobPath = typeof activeSankeyPath === "string" && activeSankeyPath.startsWith("blob:");
    const fetchRows = async (path: string): Promise<SankeySampleRow[]> => {
      const json = await fetchJsonCached(path);
      return Array.isArray(json) ? (json as SankeySampleRow[]) : [];
    };
    const mergeRows = (a: SankeySampleRow[], b: SankeySampleRow[]) => {
      const out: SankeySampleRow[] = [];
      const seen = new Set<string>();
      const add = (r: SankeySampleRow) => {
        const key = [
          String(r.real_id ?? ""),
          String(r.from_model ?? ""),
          String(r.to_model ?? ""),
          String(r.from_layer ?? ""),
          String(r.to_layer ?? ""),
          String(r.from_cluster ?? ""),
          String(r.to_cluster ?? ""),
        ].join("|");
        if (seen.has(key)) return;
        seen.add(key);
        out.push(r);
      };
      a.forEach(add);
      b.forEach(add);
      return out;
    };

    fetchRows(activeSankeyPath)
      .then(async (primaryRows) => {
        if (!activeSankeySecondaryPath) return primaryRows;
        try {
          const secondaryRows = await fetchRows(activeSankeySecondaryPath);
          return mergeRows(primaryRows, secondaryRows);
        } catch (err) {
          console.error("Failed to load secondary active sankey rows:", err);
          return primaryRows;
        }
      })
      .then((rows) => {
        setActiveSankeyRows(rows);
      })
      .catch((err) => {
        // In-memory payload mode may briefly reference stale blob URLs while state swaps.
        // Ignore these transient blob fetch failures and wait for the next valid path.
        if (isBlobPath) return;
        console.error(err);
        setActiveSankeyRows([]);
      });
  }, [activeSankeyPath, activeSankeySecondaryPath, sankeyRefreshKey]);

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
    const fetchTopk = async (url: string) => {
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to load topk data");
      const json = await res.json();
      return Array.isArray(json) ? json : [];
    };
    const tagRowIds = (rows: any[], split: "train" | "test") =>
      rows.map((row) => ({
        ...row,
        top_k_closest_real_ids: Array.isArray(row?.top_k_closest_real_ids)
          ? row.top_k_closest_real_ids.map((x: any) => `${split}::${String(x)}`)
          : [],
        top_k_farthest_real_ids: Array.isArray(row?.top_k_farthest_real_ids)
          ? row.top_k_farthest_real_ids.map((x: any) => `${split}::${String(x)}`)
          : [],
      }));
    const mergeTopk = (a: any[], b: any[]) => {
      const map = new Map<string, any>();
      const add = (row: any) => {
        const key = `${String(row?.model_type ?? "")}__${String(row?.layer_name ?? "")}__${String(row?.cluster_id ?? "")}`;
        if (!map.has(key)) {
          map.set(key, {
            ...row,
            top_k_closest_real_ids: Array.isArray(row?.top_k_closest_real_ids)
              ? row.top_k_closest_real_ids.map((x: any) => String(x))
              : [],
            top_k_farthest_real_ids: Array.isArray(row?.top_k_farthest_real_ids)
              ? row.top_k_farthest_real_ids.map((x: any) => String(x))
              : [],
          });
          return;
        }
        const cur = map.get(key);
        const mergeArr = (x: any[], y: any[]) => {
          const seen = new Set<string>(x.map((v) => String(v)));
          const out = [...x.map((v) => String(v))];
          y.forEach((v) => {
            const sv = String(v);
            if (seen.has(sv)) return;
            seen.add(sv);
            out.push(sv);
          });
          return out;
        };
        cur.top_k_closest_real_ids = mergeArr(cur.top_k_closest_real_ids ?? [], row?.top_k_closest_real_ids ?? []);
        cur.top_k_farthest_real_ids = mergeArr(cur.top_k_farthest_real_ids ?? [], row?.top_k_farthest_real_ids ?? []);
        cur.k = Math.max(Number(cur.k ?? 0), Number(row?.k ?? 0));
      };
      a.forEach(add);
      b.forEach(add);
      return Array.from(map.values());
    };

    const trainUrl = buildDataFileUrl({
      section: "topk",
      file: "cluster_representatives.json",
    });
    const testUrl = buildDataFileUrl({
      dataset: selectedDataset,
      clustering: selectedClustering,
      dataSplit: "test",
      section: "topk",
      file: "cluster_representatives.json",
    });

    fetchTopk(trainUrl)
      .then(async (trainRows) => {
        if (selectedDataSplit !== "all") return trainRows;
        try {
          const testRows = await fetchTopk(testUrl);
          return mergeTopk(tagRowIds(trainRows, "train"), tagRowIds(testRows, "test"));
        } catch (err) {
          console.error("Failed to load test topk representatives:", err);
          return tagRowIds(trainRows, "train");
        }
      })
      .then((rows) => setTopkData(rows))
      .catch((err) => {
        console.error(err);
        setTopkData([]);
      });
  }, [
    datasetBase,
    dataRefreshKey,
    generatedRuntimeMatchesCurrent,
    generatedRuntimePayload,
    selectedDataSplit,
    selectedDataset,
    selectedClustering,
    buildDataFileUrl,
  ]);

  useEffect(() => {
    if (generatedRuntimeMatchesCurrent) {
      setClusterSortedMembers(generatedRuntimePayload?.clusterSortedMembers ?? []);
      return;
    }
    const fetchSorted = async (url: string) => {
      const res = await fetch(url);
      if (res.status === 404) return [];
      if (!res.ok) throw new Error("Failed to load cluster sorted members");
      const json = await res.json();
      return Array.isArray(json) ? json : [];
    };
    const mergeSorted = (a: any[], b: any[]) => {
      const map = new Map<string, any>();
      const add = (row: any, splitTag: "train" | "test" | null = null) => {
        const key = `${String(row?.model_type ?? "")}__${String(row?.layer_name ?? "")}__${String(row?.cluster_id ?? "")}`;
        const nextIds = Array.isArray(row?.sorted_real_ids)
          ? row.sorted_real_ids.map((x: any) =>
              splitTag ? `${splitTag}::${String(x)}` : String(x)
            )
          : [];
        if (!map.has(key)) {
          map.set(key, {
            ...row,
            sorted_real_ids: nextIds,
          });
          return;
        }
        const cur = map.get(key);
        const merged = new Set<string>(Array.isArray(cur.sorted_real_ids) ? cur.sorted_real_ids.map((x: any) => String(x)) : []);
        nextIds.forEach((x: any) => merged.add(String(x)));
        cur.sorted_real_ids = Array.from(merged);
        cur.count = Math.max(Number(cur.count ?? 0), Number(row?.count ?? 0), cur.sorted_real_ids.length);
      };
      if (selectedDataSplit === "all") {
        a.forEach((r) => add(r, "train"));
        b.forEach((r) => add(r, "test"));
      } else {
        a.forEach((r) => add(r, null));
        b.forEach((r) => add(r, null));
      }
      return Array.from(map.values());
    };

    const baseTrain = buildDataFileUrl({
      section: "topk",
      file: "cluster_sorted_members.json",
    });
    const baseTest = buildDataFileUrl({
      dataset: selectedDataset,
      clustering: selectedClustering,
      dataSplit: "test",
      section: "topk",
      file: "cluster_sorted_members.json",
    });
    fetchSorted(baseTrain)
      .then(async (trainRows) => {
        if (selectedDataSplit !== "all") return trainRows;
        try {
          const testRows = await fetchSorted(baseTest);
          return mergeSorted(trainRows, testRows);
        } catch (err) {
          console.error("Failed to load test cluster sorted members:", err);
          return mergeSorted(trainRows, []);
        }
      })
      .then((rows) => {
        setClusterSortedMembers(rows);
      })
      .catch((err) => {
        console.error(err);
        setClusterSortedMembers([]);
      });
  }, [datasetBase, dataRefreshKey, generatedRuntimeMatchesCurrent, generatedRuntimePayload, selectedDataset, selectedClustering, selectedDataSplit, buildDataFileUrl]);

  useEffect(() => {
    if (generatedRuntimeMatchesCurrent) {
      setSubstructures(generatedRuntimePayload?.substructures ?? []);
      return;
    }
    fetch(
      buildDataFileUrl({
        section: "substructures",
        file: "cluster_substructures.json",
      })
    )
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
  }, [datasetBase, dataRefreshKey, generatedRuntimeMatchesCurrent, generatedRuntimePayload, buildDataFileUrl]);

  useEffect(() => {
    const controller = new AbortController();
    if (!selectedNode) {
      setIsKnnTopmLoading(false);
      return () => controller.abort();
    }
    const fetchCellTopm = async () => {
      setIsKnnTopmLoading(true);
      try {
        const useCurrent =
          generatedRuntimeMatchesCurrent;
        const body = {
          dataset: selectedDataset,
          clustering: selectedClustering,
          data_split: selectedDataSplit,
          epoch: selectedModel,
          layer: selectedLayer,
          cluster_id: "all",
          m: knnTopm,
          group: "knn",
          use_current: useCurrent,
        };
        const doRequest = async (path: string) =>
          fetch(`${apiBase}${path}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
            signal: controller.signal,
          });

        let res = await doRequest("/substructures/cluster-topm");
        if (res.status === 404) {
          res = await doRequest("/substructures/cell-topm");
        }
        if (!res.ok) throw new Error(`topm failed: ${res.status}`);
        const json = await res.json();
        const rows: unknown[] = Array.isArray(json?.rows) ? json.rows : [];
        const normalizedRows = rows.map((row: unknown) => {
          const r = (row ?? {}) as Record<string, unknown>;
          const groupRaw = String(r.group ?? "knn").toLowerCase();
          const group: "knn" | "kfn" = groupRaw === "kfn" ? "kfn" : "knn";
          return {
            model_type: String(r.model_type ?? r.model ?? r.epoch ?? selectedModel),
            layer_name: String(r.layer_name ?? r.layer ?? selectedLayer),
            cluster_id: Number(r.cluster_id ?? r.cluster ?? -1),
            group,
            k: Number(r.k ?? knnTopm),
            real_ids: Array.isArray(r.real_ids) ? (r.real_ids as Array<string | number>) : [],
            valid_molecules: Number(r.valid_molecules ?? 0),
            missing_real_ids: Array.isArray(r.missing_real_ids) ? (r.missing_real_ids as Array<string | number>) : [],
            invalid_smiles_real_ids: Array.isArray(r.invalid_smiles_real_ids)
              ? (r.invalid_smiles_real_ids as Array<string | number>)
              : [],
            mcs_smarts: (r.mcs_smarts ?? r.smarts ?? null) as string | null,
            mcs_smiles: (r.mcs_smiles ?? r.mcs_smile ?? r.smiles ?? null) as string | null,
          };
        });

        console.log("[topm] refreshed rows", {
          dataset: selectedDataset,
          split: selectedDataSplit,
          clustering: selectedClustering,
          model: selectedModel,
          layer: selectedLayer,
          m: knnTopm,
          total: normalizedRows.length,
          nonNullSmiles: normalizedRows.filter((r) => r.group === "knn" && r.mcs_smiles != null).length,
          nonNullSmarts: normalizedRows.filter((r) => r.group === "knn" && r.mcs_smarts != null).length,
          sample: normalizedRows.slice(0, 2),
        });

        setSubstructures((prev) => {
          const next = prev.filter(
            (d) =>
              !(
                String(d.model_type) === String(selectedModel) &&
                String(d.layer_name) === String(selectedLayer) &&
                d.group === "knn"
              )
          );
          return [...next, ...normalizedRows];
        });
      } catch (err) {
        if ((err as Error)?.name === "AbortError") return;
        console.error("Failed to refresh knn top-m substructures:", err);
      } finally {
        if (!controller.signal.aborted) setIsKnnTopmLoading(false);
      }
    };
    fetchCellTopm();
    return () => controller.abort();
  }, [
    apiBase,
    selectedDataset,
    selectedClustering,
    selectedDataSplit,
    selectedModel,
    selectedLayer,
    selectedNode,
    knnTopm,
    generatedRuntimeMatchesCurrent,
  ]);

  const topkNearestRows = useMemo(() => {
    const mLimit = knnTopm === "all" ? null : Number(knnTopm);
    const sortedByCluster = new Map<number, Array<string | number>>();
    clusterSortedMembers.forEach((d) => {
      if (String(d.model_type) !== String(selectedModel) || String(d.layer_name) !== String(selectedLayer)) return;
      if (!Array.isArray(d.sorted_real_ids)) return;
      sortedByCluster.set(Number(d.cluster_id), d.sorted_real_ids);
    });
    return topkData
      .filter((d) => d.model_type === selectedModel && d.layer_name === selectedLayer)
      .sort((a, b) => a.cluster_id - b.cluster_id)
      .map((d) => {
        const sortedIds = sortedByCluster.get(Number(d.cluster_id));
        if (!sortedIds || sortedIds.length === 0) {
          return {
            ...d,
            top_k_closest_real_ids:
              mLimit == null ? d.top_k_closest_real_ids : d.top_k_closest_real_ids.slice(0, mLimit),
          };
        }
        const displayIds = mLimit == null ? sortedIds : sortedIds.slice(0, mLimit);
        return {
          ...d,
          top_k_closest_real_ids: displayIds,
        };
      });
  }, [topkData, clusterSortedMembers, selectedModel, selectedLayer, knnTopm]);


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

  useEffect(() => {
    setHoverPreview(null);
  }, [clusterPage, clusterPageSize, selectedNode?.model, selectedNode?.layer, selectedNode?.cluster, showSearchResultsInMembers]);

  function resolveImageSrc(realId: string | number, imageType: string) {
    const parsed = parseSplitTaggedId(realId);
    const splitForImage = parsed.split ?? effectiveDataSplitForFiles(selectedDataSplit);
    const key = sanitizePathKey(parsed.realId.trim());
    return buildDataFileUrl({
      dataset: selectedDataset,
      dataSplit: splitForImage,
      file: `${splitForImage}/molecule_images/${key}_${imageType}.png`,
      datasetRoot: true,
    });
  }

  function getSmilesById(realId: string | number): string | null {
    const parsed = parseSplitTaggedId(realId);
    const tagged = parsed.split ? `${parsed.split}::${parsed.realId}`.toLowerCase() : "";
    const base = parsed.realId.trim().toLowerCase();
    return (tagged ? smilesByRealId[tagged] : undefined) ?? smilesByRealId[base] ?? null;
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

  function showHoverPreviewAt(src: string, anchor?: DOMRect, source: "topk" | "members" | "other" = "other") {
    if (!showLargeHoverPreview) return;
    const { left, top } = getPreviewAnchorPosition(anchor);
    setHoverPreview({ kind: "image", src, left, top, source });
  }

  function showSmilesPreviewAt(
    smiles: string | null,
    anchor?: DOMRect,
    source: "topk" | "members" | "other" = "other"
  ) {
    if (!showLargeHoverPreview) return;
    if (!smiles || !smiles.trim()) {
      setHoverPreview(null);
      return;
    }
    const { left, top } = getPreviewAnchorPosition(anchor);
    setHoverPreview({ kind: "smiles", smiles, left, top, source });
  }

  useEffect(() => {
    if (!showLargeHoverPreview) {
      setHoverPreview(null);
    }
  }, [showLargeHoverPreview]);

  useEffect(() => {
    if (!hoverPreview || hoverPreview.source !== "members") return;
    const onPointerMove = (ev: PointerEvent) => {
      const host = memberListRef.current;
      if (!host) return;
      const rect = host.getBoundingClientRect();
      const inside =
        ev.clientX >= rect.left &&
        ev.clientX <= rect.right &&
        ev.clientY >= rect.top &&
        ev.clientY <= rect.bottom;
      if (!inside) setHoverPreview(null);
    };
    window.addEventListener("pointermove", onPointerMove, { passive: true });
    return () => window.removeEventListener("pointermove", onPointerMove);
  }, [hoverPreview]);

  useEffect(() => {
    const host = layoutGridRef.current;
    if (!host) return;
    const MIN_RIGHT_COL_PCT = 100 - HALF_RIGHT_WORKSPACE_WIDTH_PCT; // 17%
    const estimateScatterMinWidth = () => {
      const hostW = host.getBoundingClientRect().width;
      if (hostW <= 0) return;
      const rightColMinW = (hostW * MIN_RIGHT_COL_PCT) / 100;
      // Account for right-column paddings/borders/gaps to keep scatter at the collapsed-width feel.
      const next = Math.max(140, Math.round(rightColMinW - 14));
      setScatterMinWidthPx(next);
      setScatterWidthPx((prev) => (prev == null ? next : Math.max(next, prev)));
    };
    estimateScatterMinWidth();
    const ro = new ResizeObserver(estimateScatterMinWidth);
    ro.observe(host);
    return () => ro.disconnect();
  }, [HALF_RIGHT_WORKSPACE_WIDTH_PCT]);

  const LEFT_COL_RATIO_IN_WORKSPACE = 45 / 55;
  const MID_COL_RATIO_IN_WORKSPACE = 10 / 55;
  const TOP_ROW_RATIO_IN_WORKSPACE = 59.8 / 60;
  const MID_ROW_RATIO_IN_WORKSPACE = 0.2 / 60;
  const effectiveWorkspaceWidthPct = workspaceWidthPct;
  const leftColPct = effectiveWorkspaceWidthPct * LEFT_COL_RATIO_IN_WORKSPACE;
  const midColPct = effectiveWorkspaceWidthPct * MID_COL_RATIO_IN_WORKSPACE;
  const rightColPct = 100 - effectiveWorkspaceWidthPct;
  const topRowPct = workspaceHeightPct * TOP_ROW_RATIO_IN_WORKSPACE;
  const midRowPct = workspaceHeightPct * MID_ROW_RATIO_IN_WORKSPACE;
  const bottomRowPct = 100 - workspaceHeightPct;
  const showMembersPanelByWidth = rightColPct >= 24;
  const isRightPanelCollapsed = Math.abs(workspaceWidthPct - HALF_RIGHT_WORKSPACE_WIDTH_PCT) < 0.5;
  const scatterResolvedWidthPx =
    scatterWidthPx == null ? null : Math.max(scatterMinWidthPx ?? 140, Math.min(900, scatterWidthPx));

  function startScatterCornerResize(e: ReactPointerEvent<HTMLElement>) {
    if (!showMembersPanelByWidth) return;
    const currentWidth = scatterResolvedWidthPx ?? scatterMinWidthPx ?? 160;
    scatterCornerResizeRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      startWidth: currentWidth,
      startTopHeightPct: rightTopHeightPct,
    };
    setIsScatterCornerResizing(true);
    try {
      (e.currentTarget as HTMLDivElement).setPointerCapture(e.pointerId);
    } catch {}
    e.preventDefault();
  }

  useEffect(() => {
    if (!isScatterCornerResizing) return;
    const onMove = (ev: PointerEvent) => {
      const state = scatterCornerResizeRef.current;
      const host = rightTopRowRef.current;
      const panel = layoutGridRef.current;
      if (!state || !host || !panel) return;
      const rect = host.getBoundingClientRect();
      const minW = scatterMinWidthPx ?? 140;
      const membersMinW = showMembersPanelByWidth ? 220 : 0;
      const maxW = Math.max(minW, Math.floor(rect.width - membersMinW - 8));
      const next = Math.max(minW, Math.min(maxW, Math.round(state.startWidth + (ev.clientX - state.startX))));
      setScatterWidthPx(next);

      const panelRect = panel.getBoundingClientRect();
      if (panelRect.height > 0) {
        const deltaYPct = ((ev.clientY - state.startY) / panelRect.height) * 100;
        const nextTopPct = Math.max(20, Math.min(58, state.startTopHeightPct + deltaYPct));
        setRightTopHeightPct(nextTopPct);
      }
    };
    const onUp = () => {
      scatterCornerResizeRef.current = null;
      setIsScatterCornerResizing(false);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp, { once: true });
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [isScatterCornerResizing, scatterMinWidthPx, showMembersPanelByWidth]);

  function toggleWorkspaceWidthPreset() {
    setWorkspaceWidthPct((prev) => {
      const isHalfRight = Math.abs(prev - HALF_RIGHT_WORKSPACE_WIDTH_PCT) < 0.5;
      return isHalfRight ? DEFAULT_WORKSPACE_WIDTH_PCT : HALF_RIGHT_WORKSPACE_WIDTH_PCT;
    });
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
                key={`sankeyview-${selectedDataset}-${selectedClustering}-${selectedDataSplit}-${sankeyRefreshKey}`}
                dataPath={buildAllSankeyPath(selectedDataset, selectedClustering, selectedDataSplit)}
                secondaryDataPath={
                  selectedDataSplit === "all"
                    ? buildAllSankeyPath(selectedDataset, selectedClustering, "test")
                    : undefined
                }
                metricsPath={withVersion(
                  buildDataFileUrl({
                    dataset: selectedDataset,
                    clustering: selectedClustering,
                    dataSplit: selectedDataSplit,
                    section: "bic_aic",
                    file: "avg_fmi_cspa_by_layer.json",
                  }),
                  dataRefreshKey
                )}
                transitionStatsPath={withVersion(
                  buildDataFileUrl({
                    dataset: selectedDataset,
                    clustering: selectedClustering,
                    dataSplit: selectedDataSplit,
                    section: "clustering_metrics",
                    file: "layer_transition_by_epoch.json",
                  }),
                  dataRefreshKey
                )}
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
                reverseEpochOrder={reverseEpochOrder}
                onToggleEpochOrder={() => setReverseEpochOrder((v) => !v)}
                onVisibleEpochsChange={handleVisibleEpochsChange}
                cellMetricByCell={cellMetricByCell}
                selectedCellMetric={selectedCellBgMetric}
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
                secondaryDataPath={
                  selectedDataSplit === "all"
                    ? swapAxes
                      ? buildSankeyModelPath(selectedModel, selectedDataset, selectedClustering, "test")
                      : buildVerticalPath(selectedLayer, selectedDataset, selectedClustering, "test")
                    : undefined
                }
                mode={swapAxes ? "layer" : "model"}
                layerKey={selectedLayer}
                modelKey={selectedModel}
                reverseEpochOrder={reverseEpochOrder}
                visibleEpochs={swapAxes ? null : visibleEpochs}
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
          <div className="p-1 box-border" style={{ gridColumn: "1 / 2", gridRow: "2 / 4" }}>
            <div
              id="sankeyHorizontal"
              className="w-full h-full border border-zinc-400 relative"
            >
              <div className="absolute top-1 left-2 z-30 pointer-events-none text-xs text-zinc-600 bg-white/90 px-1 rounded shadow-sm">
                {swapAxes ? selectedLayer : selectedModel}
              </div>
              <SankeyHorizontal
                dataPath={swapAxes ? sankeyPathVertical : sankeyPath}
                secondaryDataPath={
                  selectedDataSplit === "all"
                    ? swapAxes
                      ? buildVerticalPath(selectedLayer, selectedDataset, selectedClustering, "test")
                      : buildSankeyModelPath(selectedModel, selectedDataset, selectedClustering, "test")
                    : undefined
                }
                mode={swapAxes ? "model" : "layer"}
                modelKey={selectedModel}
                layerKey={selectedLayer}
                reverseEpochOrder={reverseEpochOrder}
                visibleEpochs={swapAxes ? visibleEpochs : null}
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
          <div className="p-1 box-border" style={{ gridColumn: "2 / 3", gridRow: "2 / 4" }}>
            <div
              id="editor"
              className="w-full h-full border border-zinc-400 relative"
            >
              <StructureEditorPanel
                dataset={selectedDataset}
                dataSplit={selectedDataSplit}
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

          <div className="p-1 box-border" style={{ gridColumn: "3 / 4", gridRow: "1 / 4" }}>
            <div className="w-full h-full min-w-0 flex flex-col gap-0">
              <div
                ref={rightTopRowRef}
                className="min-h-0 flex min-w-0 gap-0"
                style={{ height: `${rightTopHeightPct}%` }}
              >
              <div
                className="relative h-full min-w-0 flex flex-col border border-zinc-400"
                style={
                  scatterResolvedWidthPx != null
                    ? { width: `${scatterResolvedWidthPx}px`, maxWidth: "100%", flex: "0 0 auto" }
                    : { flex: "1 1 auto" }
                }
              >
                <div
                  id="scatterplot"
                  className="flex-1 min-h-0 w-full bg-white"
                >
                  <ScatterPlot
                    dataPathXY={scatterPaths.xy}
                    dataPathCluster={scatterPaths.cluster}
                    splitMapPath={withVersion(
                      `${apiBase}/real-id-split-map?dataset=${encodeURIComponent(selectedDataset)}`,
                      dataRefreshKey
                    )}
                    splitMapDataset={selectedDataset}
                    dataSplit={selectedDataSplit}
                    secondaryDataPathXY={
                      selectedDataSplit === "all"
                        ? buildScatterPaths(
                            selectedModel,
                            selectedLayer,
                            scatterProjection,
                            selectedDataset,
                            selectedClustering,
                            "test"
                          ).xy
                        : undefined
                    }
                    secondaryDataPathCluster={
                      selectedDataSplit === "all"
                        ? buildScatterPaths(
                            selectedModel,
                            selectedLayer,
                            scatterProjection,
                            selectedDataset,
                            selectedClustering,
                            "test"
                          ).cluster
                        : undefined
                    }
                    projection={scatterProjection}
                    onProjectionChange={(next) => setScatterProjection(next)}
                    onUserClusteringUpload={(payload) => setUploadedUserClustering(payload)}
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
                <button
                  type="button"
                  aria-label="Resize scatter width and height"
                  className={
                    "absolute right-1 bottom-1 z-30 w-6 h-6 rounded-md border border-zinc-400 bg-white/95 hover:bg-zinc-100 shadow flex items-center justify-center cursor-nwse-resize " +
                    (isScatterCornerResizing ? "border-zinc-600 bg-zinc-100" : "")
                  }
                  onPointerDown={startScatterCornerResize}
                  title="Drag to resize scatter width and height"
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                    <path d="M6 2H2v4M10 14h4v-4M2 6l4-4M14 10l-4 4" stroke="currentColor" strokeWidth="1.8" fill="none" strokeLinecap="round" />
                  </svg>
                </button>
              </div>
              {showMembersPanelByWidth && (
              <div className="h-full min-w-0 flex-1 flex gap-0">
                {SHOW_CONTAINMENT_VIEW && (
                  <div className="h-full min-w-[240px] w-[28%] border border-zinc-400 bg-white">
                    <ClusterContainmentHeatmap
                      loading={clusterContainmentLoading}
                      error={clusterContainmentError}
                      userClusterIds={clusterContainment?.user_cluster_ids ?? []}
                      ourClusterIds={clusterContainment?.our_cluster_ids ?? []}
                      matrix={clusterContainment?.matrix ?? []}
                      entries={clusterContainment?.entries ?? []}
                      userClusterLabelNames={uploadedUserClustering?.labelNames ?? []}
                    />
                  </div>
                )}
                <div
                  id="bottomPanel1"
                  className="w-full h-full border border-zinc-400 bg-white min-w-0 flex-1"
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
                                      setHoverPreview(null);
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
                                  onClick={() => {
                                    setClusterPage((p) => Math.max(0, p - 1));
                                    setHoverPreview(null);
                                  }}
                                  >
                                    Prev
                                  </button>
                                  <button
                                    type="button"
                                    className="px-2 py-0.5 border border-zinc-300 rounded disabled:opacity-40"
                                    disabled={clusterPage >= clusterTotalPages - 1}
                                  onClick={() => {
                                    setClusterPage((p) => Math.min(clusterTotalPages - 1, p + 1));
                                    setHoverPreview(null);
                                  }}
                                  >
                                    Next
                                  </button>
                              </div>
                            </div>
                            <div
                              ref={memberListRef}
                              className="flex-1 min-h-0 overflow-auto pr-1"
                              onMouseLeave={() => {
                                setHoverFromMouse(null);
                                setHoverPreview(null);
                              }}
                            >
                              <div className="grid grid-cols-5 gap-1">
                                {clusterPagedIds.map((id) => (
                                  <div
                                    key={`instance-${id}`}
                                    className="relative w-full max-w-[180px] aspect-square mx-auto"
                                    onMouseEnter={(e) => {
                                      setHoverFromMouse(String(id));
                                      showSmilesPreviewAt(
                                        getSmilesById(id),
                                        e.currentTarget.getBoundingClientRect(),
                                        "members"
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
              )}
              </div>
              <div
                id="instanceview"
                className="relative min-h-0 border border-zinc-400 bg-white"
                style={{ height: `${100 - rightTopHeightPct}%` }}
              >
                <button
                  type="button"
                  className="absolute top-1 left-1 z-30 w-6 h-6 rounded-md border border-zinc-400 bg-white/95 hover:bg-zinc-100 shadow cursor-pointer flex items-center justify-center"
                  onClick={toggleWorkspaceWidthPreset}
                  title="Toggle Right Panel Width (50% / default)"
                  aria-label="Toggle Right Panel Width (50% / default)"
                >
                  <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
                    {isRightPanelCollapsed ? (
                      <path d="M5 8l5-3.2v6.4L5 8z" fill="currentColor" />
                    ) : (
                      <path d="M11 8L6 4.8v6.4L11 8z" fill="currentColor" />
                    )}
                  </svg>
                </button>
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
                  onSmilesPreviewStart={(smiles, anchor) => showSmilesPreviewAt(smiles, anchor, "topk")}
                  onSmilesPreviewEnd={() => setHoverPreview(null)}
                  bookmarkedIds={bookmarkedIds}
                  onToggleBookmark={toggleBookmark}
                  knnTopm={knnTopm}
                  onKnnTopmChange={setKnnTopm}
                  isKnnTopmLoading={isKnnTopmLoading}
                />
              </div>
            </div>
          </div>
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
                    const nextSplit = selectedDataSplit;
                    setSelectedDataset(next);
                    setSelectedModel(defaults.model);
                    setSelectedLayer(defaults.layer);
                    syncSankeyPaths(defaults.model, defaults.layer, next, nextClustering, nextSplit);
                    setScatterPaths(
                      buildScatterPaths(
                        defaults.model,
                        defaults.layer,
                        scatterProjection,
                        next,
                        nextClustering,
                        nextSplit
                      )
                    );
                    setSelectedNode(null);
                    setSelectedIds(null);
                    setCellSelected(false);
                    setActiveSankeyRows([]);
                  }}
                  datasetOptions={DATASET_OPTIONS_UI}
                  dataSplit={selectedDataSplit}
                  onDataSplitChange={(split) => {
                    const defaults = DATASET_DEFAULTS[selectedDataset];
                    const nextSplit = split as DataSplitKey;
                    setSelectedDataSplit(nextSplit);
                    setSelectedModel(defaults.model);
                    setSelectedLayer(defaults.layer);
                    syncSankeyPaths(defaults.model, defaults.layer, selectedDataset, selectedClustering, nextSplit);
                    setScatterPaths(
                      buildScatterPaths(
                        defaults.model,
                        defaults.layer,
                        scatterProjection,
                        selectedDataset,
                        selectedClustering,
                        nextSplit
                      )
                    );
                    setSelectedNode(null);
                    setSelectedIds(null);
                    setCellSelected(false);
                    setActiveSankeyRows([]);
                  }}
                  clustering={selectedClustering}
                  clusteringOptions={CLUSTERING_OPTIONS}
                  onClusteringChange={(c) => {
                    const next = c as ClusteringKey;
                    const defaults = DATASET_DEFAULTS[selectedDataset];
                    const nextSplit = selectedDataSplit;
                    setSelectedClustering(next);
                    setSelectedModel(defaults.model);
                    setSelectedLayer(defaults.layer);
                    syncSankeyPaths(defaults.model, defaults.layer, selectedDataset, next, nextSplit);
                    setScatterPaths(
                      buildScatterPaths(
                        defaults.model,
                        defaults.layer,
                        scatterProjection,
                        selectedDataset,
                        next,
                        nextSplit
                      )
                    );
                    setSelectedNode(null);
                    setSelectedIds(null);
                    setCellSelected(false);
                    setActiveSankeyRows([]);
                  }}
                  cellBgMetric={selectedCellBgMetric}
                  cellBgMetricOptions={cellBgMetricOptions}
                  onCellBgMetricChange={setSelectedCellBgMetric}
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
