import React, { useEffect, useState } from "react";
import SankeyDiagram from "../../frontend/components/SankeyDiagram"; 

function App() {
  const [data, setData] = useState<any[]>([]);
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const dataUrl = params.get("data") || "/sankey_notebook.json";
    fetch(dataUrl)
      .then(r => r.json())
      .then(rows => setData(rows))
      .catch(err => console.error("load data", err));
  }, []);
  return (
    <div style={{padding:8}}>
      <h3>Sankey (Notebook)</h3>
      <SankeyDiagram data={data} zoomFactor={1} />
    </div>
  );
}

export default App;