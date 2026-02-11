"use client";

import { useEffect, useRef, useState } from "react";
import * as d3 from "d3";
import { CLUSTER_COLORS, HIGHLIGHT_RED } from "../theme/colors";

type ScatterPoint = {
  real_id: string;
  x: number;
  y: number;
  cluster_id: number;
};

type ScatterPlotProps = {
  dataPath: string;
  activeCluster?: number | null;
  activeIds?: string[] | null;
  highlightId?: string | null;
  modelKey?: string;
  layerKey?: string;
  onPointSelect?: (sel: { model?: string; layer?: string; cluster: number; real_id: string }) => void;
  onClearSelection?: () => void;
};

export default function ScatterPlot({
  dataPath,
  activeCluster,
  activeIds,
  highlightId,
  modelKey,
  layerKey,
  onPointSelect,
  onClearSelection,
}: ScatterPlotProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [legendClusters, setLegendClusters] = useState<number[]>([]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.innerHTML = "";
    container.style.position = "relative";

    const ro = new ResizeObserver(() => draw());
    ro.observe(container);

    let lastRows: ScatterPoint[] | null = null;
    const tooltip = d3
      .select(document.body)
      .append("div")
      .style("position", "fixed")
      .style("display", "none")
      .style("pointer-events", "none")
      .style("background", "rgba(255,255,255,0.95)")
      .style("border", "1px solid #d1d5db")
      .style("box-shadow", "0 6px 20px rgba(0,0,0,0.12)")
      .style("padding", "6px")
      .style("border-radius", "6px")
      .style("z-index", "9999");

    const tooltipImg = tooltip
      .append("img")
      .attr("alt", "ligand")
      .style("display", "block")
      .style("width", "140px")
      .style("height", "140px")
      .style("object-fit", "contain")
      .on("error", function () {
        d3.select(this).style("display", "none");
      });

    function draw() {
      if (!container || !lastRows) return;
      container.innerHTML = "";
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);

      const margin = { top: 8, right: 8, bottom: 8, left: 8 };
      const innerW = Math.max(1, width - margin.left - margin.right);
      const innerH = Math.max(1, height - margin.top - margin.bottom);

      const svg = d3.select(container).append("svg").attr("width", width).attr("height", height);
      const gRoot = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
      const g = gRoot.append("g");

      const xExtent = d3.extent(lastRows, (d) => d.x) as [number, number];
      const yExtent = d3.extent(lastRows, (d) => d.y) as [number, number];
      const x = d3.scaleLinear().domain(xExtent).range([0, innerW]).nice();
      const y = d3.scaleLinear().domain(yExtent).range([innerH, 0]).nice();

      const maxCluster = d3.max(lastRows, (d) => d.cluster_id) ?? 0;
      const color = d3
        .scaleOrdinal<number, string>()
        .domain(d3.range(maxCluster + 1))
        .range(CLUSTER_COLORS);

      const activeIdSet = activeIds ? new Set(activeIds) : null;
      const circles = g.append("g")
        .selectAll("circle")
        .data(lastRows)
        .enter()
        .append("circle")
        .attr("cx", (d) => x(d.x))
        .attr("cy", (d) => y(d.y))
        .attr("r", (d) => (highlightId && String(d.real_id) === highlightId ? 5.2 : 3.2))
        .attr("fill", (d) => (highlightId && String(d.real_id) === highlightId ? "#f59e0b" : color(Math.max(0, d.cluster_id))))
        .attr("fill-opacity", (d) => {
          if (highlightId && String(d.real_id) === highlightId) return 1;
          if (activeIdSet) return activeIdSet.has(String(d.real_id)) ? 1 : 0.22;
          if (activeCluster == null) return 0.75;
          return d.cluster_id === activeCluster ? 1 : 0.22;
        })
        .attr("stroke", (d) => {
          if (highlightId && String(d.real_id) === highlightId) return "#dc2626";
          if (activeIdSet) return activeIdSet.has(String(d.real_id)) ? "#dc2626" : "none";
          if (activeCluster == null) return "none";
          return d.cluster_id === activeCluster ? "#dc2626" : "none";
        })
        .attr("stroke-width", (d) => (highlightId && String(d.real_id) === highlightId ? 1.8 : 0.9))
        .style("cursor", "pointer")
        .on("mouseover", (event, d) => {
          const imgSrc = `/molecule_images/${String(d.real_id)}_ligand.png`;
          tooltipImg.style("display", "block").attr("src", imgSrc);
          tooltip.style("display", "block");
          positionTooltip(event.clientX, event.clientY);
        })
        .on("mousemove", (event) => {
          positionTooltip(event.clientX, event.clientY);
        })
        .on("mouseout", () => {
          tooltip.style("display", "none");
        })
        .on("click", (event, d) => {
          event.stopPropagation();
          onPointSelect?.({
            model: modelKey,
            layer: layerKey,
            cluster: d.cluster_id,
            real_id: String(d.real_id),
          });
        });

      if (highlightId) {
        const hit = lastRows.find((d) => String(d.real_id) === highlightId);
        if (hit) {
          const star = d3.symbol().type(d3.symbolStar).size(180);
          const clusterColor = color(Math.max(0, hit.cluster_id));
          g.append("path")
            .attr("d", star)
            .attr("transform", `translate(${x(hit.x)},${y(hit.y)})`)
            .attr("fill", clusterColor)
            .attr("stroke", HIGHLIGHT_RED)
            .attr("stroke-width", 1.2)
            .attr("opacity", 0.95)
            .style("pointer-events", "none");
        }
      }

      const extent: [[number, number], [number, number]] = [
        [margin.left, margin.top],
        [margin.left + innerW, margin.top + innerH],
      ];
      const zoom = d3
        .zoom<SVGSVGElement, unknown>()
        .scaleExtent([1, 8])
        .translateExtent(extent)
        .extent(extent)
        .on("zoom", (event) => {
          const t = event.transform;
          g.attr("transform", `translate(${t.x - margin.left},${t.y - margin.top}) scale(${t.k})`);
        });

      svg.call(zoom);
      svg.call(zoom.transform, d3.zoomIdentity.translate(margin.left, margin.top));

      svg.on("click", () => {
        onClearSelection?.();
      });

      function positionTooltip(clientX: number, clientY: number) {
        const pad = 12;
        const tipBounds = (tooltip.node() as HTMLDivElement | null)?.getBoundingClientRect();
        const tipW = tipBounds?.width ?? 0;
        const tipH = tipBounds?.height ?? 0;

        let x = clientX + pad;
        let y = clientY + pad;

        if (x + tipW > window.innerWidth) {
          x = Math.max(pad, clientX - pad - tipW);
        }
        if (y + tipH > window.innerHeight) {
          y = Math.max(pad, clientY - pad - tipH);
        }

        tooltip.style("left", `${x}px`).style("top", `${y}px`);
      }
    }

    fetch(dataPath)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch scatter data");
        return res.json();
      })
      .then((json: ScatterPoint[]) => {
        lastRows = json;
        const clusterIds = Array.from(new Set(json.map((d) => d.cluster_id))).sort((a, b) => a - b);
        setLegendClusters(clusterIds);
        draw();
      })
      .catch((err) => {
        console.error(err);
        if (container) container.innerHTML = "<div style='color:red'>Failed to load scatter data.</div>";
      });

    return () => {
      tooltip.remove();
      ro.disconnect();
    };
  }, [dataPath, activeCluster, activeIds, highlightId, modelKey, layerKey, onPointSelect, onClearSelection]);

  return (
    <div className="w-full h-full flex flex-col">
      <div ref={containerRef} className="w-full flex-1 min-h-0" />
      <div className="shrink-0 border-t border-zinc-200 bg-white/90 px-2 py-1 overflow-x-auto overflow-y-hidden">
        <div className="flex flex-nowrap gap-x-3 gap-y-1 min-w-max">
          {legendClusters.map((clusterId) => (
            <div key={clusterId} className="flex items-center gap-1">
              <span
                className="inline-block w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length] }}
              />
              <span
                className={`text-xs ${activeCluster != null && clusterId === activeCluster ? "text-red-600 font-bold" : "text-zinc-700"}`}
              >
                {clusterId}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
