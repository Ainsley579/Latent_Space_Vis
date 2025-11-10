"use client";
import * as d3 from "d3";
import React, {
  useEffect,
  useRef,
  forwardRef,
  useImperativeHandle,
} from "react";

interface ScatterPlotProps {
  onScroll?: (scrollLeft: number) => void;
  clusterFilePath: string;
}

const ScatterPlot = forwardRef<HTMLDivElement, ScatterPlotProps>(
  ({ onScroll, clusterFilePath }, ref) => {
    const containerRef = useRef<HTMLDivElement | null>(null);

    useImperativeHandle(ref, () => containerRef.current as HTMLDivElement);

    useEffect(() => {
      const run = async () => {
        const container = containerRef.current;
        if (!container) return;
        container.innerHTML = "";

        const layerFiles = [
          "scatter_plot_data_emb.csv",
          "scatter_plot_data_output_block_0.csv",
          "scatter_plot_data_output_block_1.csv",
          "scatter_plot_data_output_block_2.csv",
          "scatter_plot_data_output_block_3.csv",
          "scatter_plot_data_output_block_4.csv",
          "scatter_plot_data_u_mlp_input.csv",
          "scatter_plot_data_u_mlp_l1.csv",
          "scatter_plot_data_u_mlp_l2.csv",
        ];

        const height = 200;
        const containerWidth = container.clientWidth || 800;
        const svgWidth = containerWidth / 4;

        for (let i = 0; i < layerFiles.length; i++) {
          const svg = d3
            .select(container)
            .append("svg")
            .attr("height", height)
            .attr("width", svgWidth)
            .style("flex", "0 0 auto")
            .style("marginRight", "1rem")
            .style("border", "1px solid #ccc")
            .style("borderRadius", "8px")
            .style("background", "#fafafa");

          const filePath = `${clusterFilePath}/${layerFiles[i]}`;
          const localData = await d3.csv(filePath, d => ({
            x: +d.x,
            y: +d.y,
            real_id: d.real_id,
            color_value: +d.color_value,
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
          svg
            .append("g")
            .selectAll("circle")
            .data(localData)
            .join("circle")
            .attr("cx", (d) => xScale(d.x))
            .attr("cy", (d) => yScale(d.y))
            .attr("r", 4)
            .attr("fill", (d) => {
              // use color_value if exists, else fallback
              return d.color_value !== undefined && !isNaN(d.color_value)
                ? d3.interpolateTurbo(d.color_value / 3)
                : color;
            })
            .attr("opacity", 0.8)
            .attr("class", (d) => d.real_id !== undefined
              ? `scatter-point real-${d.real_id}`
              : "scatter-point");
        }
      };
      run();
    }, [clusterFilePath]);

    return (
      <div
        ref={containerRef}
        onScroll={(e) =>
          onScroll?.((e.target as HTMLDivElement).scrollLeft)
        }
        style={{
          display: "flex",
          flexDirection: "row",
          overflowX: "auto",
          width: "100%",
          paddingBottom: "0.5rem",
        }}
      ></div>
    );
  }
);

export default ScatterPlot;