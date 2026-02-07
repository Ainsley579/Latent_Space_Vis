"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";

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
};

export default function ScatterPlot({
  dataPath,
  activeCluster,
  activeIds,
  highlightId,
  modelKey,
  layerKey,
  onPointSelect,
}: ScatterPlotProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    container.innerHTML = "";
    container.style.position = "relative";

    const ro = new ResizeObserver(() => draw());
    ro.observe(container);

    let lastRows: ScatterPoint[] | null = null;

    function draw() {
      if (!container || !lastRows) return;
      container.innerHTML = "";
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);

      const tooltip = d3
        .select(container)
        .append("div")
        .style("position", "absolute")
        .style("display", "none")
        .style("pointer-events", "none")
        .style("background", "rgba(255,255,255,0.95)")
        .style("border", "1px solid #d1d5db")
        .style("box-shadow", "0 6px 20px rgba(0,0,0,0.12)")
        .style("padding", "6px")
        .style("border-radius", "6px")
        .style("z-index", "5");

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
      const color = d3.scaleOrdinal<number, string>(d3.schemeTableau10).domain(d3.range(maxCluster + 1));

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
          if (activeIdSet) return activeIdSet.has(String(d.real_id)) ? 0.95 : 0.7;
          if (activeCluster == null) return 0.75;
          return d.cluster_id === activeCluster ? 0.95 : 0.7;
        })
        .attr("stroke", (d) => {
          if (highlightId && String(d.real_id) === highlightId) return "#b45309";
          if (activeIdSet) return activeIdSet.has(String(d.real_id)) ? "#222" : "none";
          if (activeCluster == null) return "none";
          return d.cluster_id === activeCluster ? "#222" : "none";
        })
        .attr("stroke-width", (d) => (highlightId && String(d.real_id) === highlightId ? 1.4 : 0.6))
        .style("cursor", "pointer")
        .on("mouseover", (event, d) => {
          const imgSrc = `/molecule_images/${String(d.real_id)}_ligand.png`;
          tooltipImg.style("display", "block").attr("src", imgSrc);
          tooltip.style("display", "block");
          const [mx, my] = d3.pointer(event, container);
          tooltip.style("left", `${mx + 12}px`).style("top", `${my + 12}px`);
        })
        .on("mousemove", (event) => {
          const [mx, my] = d3.pointer(event, container);
          tooltip.style("left", `${mx + 12}px`).style("top", `${my + 12}px`);
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
            .attr("stroke", d3.color(clusterColor)?.darker(0.9)?.toString() ?? "#1f2937")
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
    }

    fetch(dataPath)
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch scatter data");
        return res.json();
      })
      .then((json: ScatterPoint[]) => {
        lastRows = json;
        draw();
      })
      .catch((err) => {
        console.error(err);
        if (container) container.innerHTML = "<div style='color:red'>Failed to load scatter data.</div>";
      });

    return () => {
      ro.disconnect();
    };
  }, [dataPath, activeCluster, activeIds, highlightId, modelKey, layerKey, onPointSelect]);

  return <div ref={containerRef} className="w-full h-full" />;
}
