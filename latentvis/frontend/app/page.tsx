export default function Home() {
  return (
    <div className="flex min-h-screen">
      <main className="flex min-h-screen">
        <div className="h-[850px] w-[1280px] flex  flex-wrap">
          <div className="w-[75%] h-[50%] p-1 box-border">
            <div
              id="sankey2d"
              className="w-full h-full border border-zinc-400"
            />
          </div>
          <div className="w-[25%] h-[50%] p-1 box-border">
            <div
              id="sankeyVertical"
              className="w-full h-full border border-zinc-400"
            />
          </div>

          <div className="w-[75%] h-[25%] p-1 box-border">
            <div
              id="sankeyHorizontal"
              className="w-full h-full border border-zinc-400"
            />
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