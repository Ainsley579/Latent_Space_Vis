function renderScatter(container, layersDict) {
  container.innerHTML = ""; // 清空

  const height = 200;
  const svgWidth = 350;

  const tooltip = d3.select("body")
    .append("div")
    .attr("class", "scatter-tooltip")
    .style("position", "absolute")
    .style("padding", "4px")
    .style("background", "white")
    .style("border", "1px solid #ccc")
    .style("borderRadius", "4px")
    .style("pointerEvents", "none")
    .style("opacity", 0);

  // ---- 修改布局：每 4 个换行 ----
  const main = d3.select(container)
    .style("display", "flex")
    .style("flex-wrap", "wrap")    // ★ 自动换行
    .style("flex-direction", "row")
    .style("gap", "1rem");

  Object.entries(layersDict).forEach(([layerName, pts], idx) => {
    const fallbackColor = d3.schemeCategory10[idx % 10];

    const svg = main.append("svg")
      .attr("width", svgWidth)
      .attr("height", height)
      .style("flex", "0 0 calc(25% - 1rem)")   // ★ 一行 4 个
      .style("background", "#fafafa")
      .style("border", "1px solid #ccc")
      .style("border-radius", "8px");

    svg.append("text")
      .attr("x", 6)
      .attr("y", 14)
      .attr("font-size", "10px")
      .text(layerName);

    // --- scales ---
    const xExtent = d3.extent(pts, d => d.x);
    const yExtent = d3.extent(pts, d => d.y);

    const xScale = d3.scaleLinear().domain(xExtent).range([10, svgWidth - 10]);
    const yScale = d3.scaleLinear().domain(yExtent).range([height - 10, 10]);

    const g = svg.append("g");

    const circles = g.selectAll("circle")
      .data(pts)
      .join("circle")
      .attr("cx", d => xScale(d.x))
      .attr("cy", d => yScale(d.y))
      .attr("r", 4)
      .attr("fill", d => {
        const v = d.color_value;
        if (v !== undefined && !isNaN(v)) return d3.interpolateTurbo(v / 3);
        return fallbackColor;
      })
      .attr("opacity", 0.8)
      .attr("class", d =>
        d.real_id ? `scatter-point real-${d.real_id}` : "scatter-point"
      );

    circles
      .on("mouseover", function (ev, d) {
        const idClass = `real-${d.real_id}`;

        d3.selectAll(".scatter-point").style("opacity", 0.1);

        d3.selectAll(`.${idClass}`)
          .style("opacity", 1)
          .attr("r", 6);

        tooltip
          .html(`<img src="/data/molecule_images/${d.real_id}_ligand.png" width="200"/>`)
          .style("left", (ev.pageX + 12) + "px")
          .style("top", (ev.pageY + 12) + "px")
          .style("opacity", 1);

        window.dispatchEvent(
          new CustomEvent("sankey-highlight-ids", {
            detail: [d.real_id]
          })
        );
      })
      .on("mouseout", function () {
        d3.selectAll(".scatter-point")
          .style("opacity", 0.8)
          .attr("r", 4);

        tooltip.style("opacity", 0);

        window.dispatchEvent(new CustomEvent("sankey-highlight-clear"));
        window.dispatchEvent(new CustomEvent("scatter-highlight-clear"));
      });

    window.addEventListener("scatter-highlight-clear", () => {
      d3.selectAll(".scatter-point")
        .style("opacity", 0.8)
        .attr("r", 4);
    });
  });
}