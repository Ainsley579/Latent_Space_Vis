import json
import uuid
from IPython.display import IFrame
from pathlib import Path

def layervis(nodes, links, sample_rows, split_map,
             template="sankey_template.html"):

    html = Path(template).read_text()
    html = html.replace("window.__DATA__", json.dumps({
        "nodes": nodes,
        "links": links,
        "sampleRows": sample_rows
    }))
    html = html.replace("window.__SPLIT_MAP__", json.dumps(split_map))

    out = Path("sankey_output.html")
    out.write_text(html)
    return IFrame(str(out), width="100%", height="620")