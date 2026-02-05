"use client";
import { useState } from "react";
import SankeyView from "./component/SankeyView.tsx";
import SankeyHorizontal from "./component/SankeyHorizontal.tsx";

export default function Home() {
  const [sankeyPath, setSankeyPath] = useState("/data/sankey/noaug_sankey_samples.json");
  const [sankeyTransform, setSankeyTransform] = useState({ x: 0, y: 0, scale: 1 });
  return (
    <div className="flex min-h-screen">
      <main className="flex min-h-screen">
        <div className="h-[850px] w-[1280px] flex  flex-wrap">
          <div className="w-[75%] h-[50%] p-1 box-border">
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
                }}
                onTransform={(t) => setSankeyTransform(t)}
              />
            </div>
          </div>
          <div className="w-[25%] h-[50%] p-1 box-border">
            <div
              id="sankeyVertical"
              className="w-full h-full border border-zinc-400"
            >
              
            </div>
          </div>

          <div className="w-[75%] h-[25%] p-1 box-border">
            <div
              id="sankeyHorizontal"
              className="w-full h-full border border-zinc-400"
            >
              <SankeyHorizontal dataPath={sankeyPath} syncTransform={{ x: sankeyTransform.x, scale: sankeyTransform.scale }} />
            </div>
          </div>
          <div className="w-[25%] h-[25%] p-1 box-border">
            <div
              id="scatterplot"
              className="w-full h-full border border-zinc-400"
            />
          </div>
          <div className="w-[50%] h-[25%] p-1 box-border">
            <div
              id="instanceview"
              className="w-full h-full border border-zinc-400"
            />
          </div>
          <div className="w-[25%] h-[25%] p-1 box-border">
            <div
              id="patternview"
              className="w-full h-full border border-zinc-400"
            />
          </div>
          <div className="w-[25%] h-[25%] p-1 box-border">
            <div
              id="aicbic"
              className="w-full h-full border border-zinc-400"
            />
          </div>
        </div>
        
      </main>
    </div>
  );
}
