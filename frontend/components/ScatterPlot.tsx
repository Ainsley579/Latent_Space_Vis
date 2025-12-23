"use client";
import * as d3 from "d3";
import { brush } from "d3-brush";
import React, {
  useEffect,
  useRef,
  forwardRef,
  useImperativeHandle,
} from "react";

interface ScatterPlotProps {
  onScroll?: (scrollLeft: number) => void;

  // ⬇️ 新：直接传入 anyembedding 的 scatter 数据
  data?: {
    x: number;
    y: number;
    real_id: string | number;
    layer: string;
    color_value?: number;
  }[];

  // 旧 CSV 路径（保留但不再使用）
  clusterFilePath?: string;

  totalWindows?: number;
  zoomFactor?: number;
}

const ScatterPlot = forwardRef<HTMLDivElement, ScatterPlotProps>(
  ({ onScroll, data = [], totalWindows, zoomFactor = 1 }, ref) => {
    let tooltip = d3.select("body").select("#global-tooltip");
    if (tooltip.empty()) {
      tooltip = d3.select("body")
        .append("div")
        .attr("id", "global-tooltip")
        .style("position", "absolute")
        .style("padding", "4px")
        .style("background", "white")
        .style("border", "1px solid #ccc")
        .style("borderRadius", "4px")
        .style("pointerEvents", "none")
        .style("zIndex", "9999")
        .style("opacity", 0);
    }

    // Always reset style every mount (keeps tooltip consistent)
    tooltip
      .style("position", "absolute")
      .style("padding", "4px")
      .style("background", "white")
      .style("border", "1px solid #ccc")
      .style("borderRadius", "4px")
      .style("pointerEvents", "none")
      .style("zIndex", "9999")
      .style("opacity", 0)
      .html("");

    const containerRef = useRef<HTMLDivElement | null>(null);
    // Move selectedIDs to top-level so it's accessible in clearScatter and run
    let selectedIDs = new Set<string>();
    const [lassoEnabled, setLassoEnabled] = React.useState(false);

    useEffect(() => {
      const handleScatterHighlight = (e: any) => {
        const ids: string[] = e.detail;

        d3.selectAll(".scatter-point")
          .style("opacity", 0.1)
          .attr("r", 4);

        ids.forEach(id => {
          d3.selectAll(`.real-${id}`)
            .style("opacity", 1)
            .attr("r", 6);
        });
      };

      const clearScatter = () => {
        selectedIDs.clear?.(); // if exists
        d3.selectAll(".scatter-point")
          // .transition()
          // .duration(150)
          .style("opacity", 0.8)
          .attr("r", 4);
        // safely clear active brushes only when lasso is enabled
        if (lassoEnabled) {
          d3.selectAll(".brush").each(function () {
            const brushG = d3.select(this);
            const state = (brushG.node() as any).__brush;
            if (state) {
              brushG.call(d3.brush().move, null);
            }
          });
        }
        window.dispatchEvent(new CustomEvent("sankey-highlight-clear"));
      };

      window.addEventListener("scatter-highlight-ids", handleScatterHighlight);
      window.addEventListener("scatter-highlight-clear", clearScatter);

      return () => {
        window.removeEventListener("scatter-highlight-ids", handleScatterHighlight);
        window.removeEventListener("scatter-highlight-clear", clearScatter);
      };
    }, []);

    useImperativeHandle(ref, () => containerRef.current as HTMLDivElement);

    // NEW: zoom handler — rescale wrapper widths without redrawing circles
    useEffect(() => {
      const container = containerRef.current;
      if (!container) return;

      // select all wrappers
      d3.select(container)
        .selectAll("div.scatter-wrapper")
        .style("width", (d: any, i: number, nodes: any[]) => {
          const node = nodes[i] as HTMLDivElement;
          const baseWidth = (node as any).__baseWidth || node.clientWidth / zoomFactor;
          return baseWidth * zoomFactor + "px";
        });

      // scale SVGs too
      d3.select(container)
        .selectAll("svg.scatter-svg")
        .attr("width", function (this: SVGSVGElement) {
          const svg = d3.select(this);
          const base = (svg as any).node().__baseWidth || +svg.attr("width") / zoomFactor;
          return base * zoomFactor;
        });
    }, [zoomFactor]);

    // Prevent StrictMode double-run
    const hasDrawnRef = React.useRef(false);

    useEffect(() => {
      if (hasDrawnRef.current) return;
      hasDrawnRef.current = true;

      const run = async () => {
        const container = containerRef.current;
        if (!container) return;
        d3.select(container).selectAll("*").remove();

        // Use layerFiles from data, not hardcoded
        const layerFiles = Array.from(
          new Set(data.map(d => `${d.layer}.csv`))
        );
        window.dispatchEvent(
          new CustomEvent("scatter-layer-info", {
            detail: {
              windowId: undefined,
              names: layerFiles,
              count: layerFiles.length
            }
          })
        );

        const height = 200;
        // selectedIDs now defined at component top-level

        for (let i = 0; i < layerFiles.length; i++) {
          // subtract total margin (5px per gap)
          const totalMargin = (layerFiles.length - 1) * 5;
          const availableWidth = container.clientWidth - totalMargin;
          const svgWidth = availableWidth / layerFiles.length;

          const wrapper = d3
            .select(container)
            .append("div")
            .classed("scatter-wrapper", true)
            .each(function() { (this as any).__baseWidth = svgWidth; })
            .style("width", (svgWidth * zoomFactor) + "px")
            .style("display", "flex")
            .style("flex-direction", "column")
            .style("align-items", "center")
            .style("margin-right", "5px");

          wrapper
            .append("div")
            .style("font-size", "12px")
            .style("margin-bottom", "4px")
            .text(layerFiles[i].replace(".csv", ""));

          const svg = wrapper
            .append("svg")
            .classed("scatter-svg", true)
            .attr("height", height)
            .each(function() { (this as any).__baseWidth = svgWidth; })
            .attr("width", svgWidth * zoomFactor)
            .style("background", "#fafafa");

          // Instead of reading CSV, filter from data
          const layerName = layerFiles[i].replace(".csv", "");
          const localData = data
            .filter(d => d.layer === layerName)
            .map(d => ({
              x: d.x,
              y: d.y,
              real_id: d.real_id,
              color_value: d.color_value,
            }));

          const xExtent = d3.extent(localData, d => d.x) as [number, number];
          const yExtent = d3.extent(localData, d => d.y) as [number, number];

          const xScale = d3.scaleLinear()
            .domain(xExtent)
            .range([10, svgWidth - 10]);

          const yScale = d3.scaleLinear()
            .domain(yExtent)
            .range([height - 10, 10]); // invert y

          const color = d3.schemeCategory10[i % 10];
          // Create group for circles so we can attach events
          const circleGroup = svg.append("g");

          const circles = circleGroup
            .selectAll("circle")
            .data(localData)
            .join("circle")
            .attr("cx", (d) => xScale(d.x))
            .attr("cy", (d) => yScale(d.y))
            .attr("r", 4)
            .attr("fill", (d) => {
              return d.color_value !== undefined && !isNaN(d.color_value)
                ? d3.interpolateTurbo(d.color_value / 3)
                : color;
            })
            .attr("opacity", 0.8)
            .attr("class", (d) =>
              d.real_id !== undefined ? `scatter-point real-${d.real_id}` : "scatter-point"
            );
          // Add hover events to dim others and highlight same real_id across all plots
          circles
            .on("mouseover", function (event, d) {
              if (lassoEnabled) return;

              const idClass = `real-${d.real_id}`;

              d3.selectAll(".scatter-point")
                .style("opacity", 0.1);

              d3.selectAll(`.${idClass}`)
                .style("opacity", 1)
                .attr("r", 6);

              tooltip
                .html(`<img src="/data/molecule_images/${d.real_id}_ligand.png" style="width:200px;height:auto;" />`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY + 10) + "px")
                .style("opacity", 1);

              window.dispatchEvent(
                new CustomEvent("sankey-highlight-ids", {
                  detail: [d.real_id],
                })
              );
            })
            .on("mouseout", function () {
              if (lassoEnabled) return;

              d3.selectAll(".scatter-point")
                .style("opacity", 0.8)
                .attr("r", 4);
              tooltip
                .style("opacity", 0);
              window.dispatchEvent(new CustomEvent("sankey-highlight-clear"));
              window.dispatchEvent(new CustomEvent("scatter-highlight-clear"));
            });

          // --- LASSO / BRUSH SELECTION ---
          const brushLayer = svg.append("g").attr("class", "brush");

          const brushed = (event: any) => {
            const selection = event.selection;
            if (!selection) {
              selectedIDs.clear();
              d3.selectAll(".scatter-point")
                .style("opacity", 0.8)
                .attr("r", 4);
              window.dispatchEvent(new CustomEvent("sankey-highlight-clear"));
              return;
            }

            const [[x0, y0], [x1, y1]] = selection;
            selectedIDs.clear();

            circles.each(function (d: any) {
              const cx = xScale(d.x);
              const cy = yScale(d.y);
              if (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) {
                selectedIDs.add(d.real_id);
              }
            });

            d3.selectAll(".scatter-point")
              .style("opacity", 0.1)
              .attr("r", 4);

            selectedIDs.forEach(id => {
              d3.selectAll(`.real-${id}`)
                .style("opacity", 1)
                .attr("r", 6);
            });

            window.dispatchEvent(
              new CustomEvent("sankey-highlight-ids", {
                detail: Array.from(selectedIDs),
              })
            );
          };

          if (lassoEnabled) {
            brushLayer.call(
              d3.brush()
                .extent([[0, 0], [svgWidth, height]])
                .on("brush end", brushed)
            );
          } else {
            brushLayer.selectAll("*").remove();
          }
        }
      };

      run();
    }, [data, zoomFactor]);
 
    return (
      <div>
        <div style={{
          display: "flex",
          flexDirection: "row",
          gap: "1rem",
          alignItems: "center",
          padding: "0.5rem 0",
          borderBottom: "1px solid #ddd",
          marginBottom: "0.5rem"
        }}>
          <div>
            <label style={{ fontSize: "0.8rem", marginRight: "0.5rem" }}>Dimensionality Reduction</label>
            <select
              style={{ padding: "2px 6px", fontSize: "0.8rem" }}
              onChange={(e) => {
                window.dispatchEvent(
                  new CustomEvent("scatter-method-change", {
                    detail: e.target.value
                  })
                );
              }}
            >
              <option value="umap">UMAP</option>
              <option value="tsne">t-SNE</option>
              <option value="pca">PCA</option>
              <option value="lle">LLE</option>
              <option value="isomap">Isomap</option>
            </select>
          </div>

          <div>
            <label style={{ fontSize: "0.8rem", marginRight: "0.5rem" }}>Color Legend</label>
            <select
              style={{ padding: "2px 6px", fontSize: "0.8rem" }}
              onChange={(e) => {
                window.dispatchEvent(
                  new CustomEvent("scatter-color-legend-change", {
                    detail: e.target.value
                  })
                );
              }}
            >
              <option value="cluster">Cluster</option>
              <option value="density">Density</option>
            </select>
          </div>
          <button
            onClick={() => setLassoEnabled(!lassoEnabled)}
            style={{
              padding: "4px 8px",
              fontSize: "0.8rem",
              background: lassoEnabled ? "#c62828" : "#2e7d32",
              color: "white",
              borderRadius: "4px",
              marginLeft: "1rem",
            }}
          >
            {lassoEnabled ? "Disable Lasso" : "Enable Lasso"}
          </button>
        </div>
        <div
          ref={containerRef}
          onScroll={(e) => {
            const left = (e.target as HTMLDivElement).scrollLeft;
            onScroll?.(left);
          }}
          onWheel={(e) => {
            const container = e.currentTarget as HTMLDivElement;
            container.scrollLeft += e.deltaY;
            // Notify WindowManager for global synchronization
            window.dispatchEvent(
              new CustomEvent("sankey-scroll", { detail: container.scrollLeft })
            );
          }}
          style={{
            display: "flex",
            flexDirection: "row",
            width: "100%",
            overflowX: "scroll",
            overflowY: "hidden",
            paddingBottom: "0.5rem"
          }}
        ></div>
      </div>
    );
  }
);

export default ScatterPlot;