import json, uuid
from pathlib import Path
from IPython.display import IFrame

def scattervis(layers_dict, template="scatter_template.html"):
    html = Path(template).read_text()
    html = html.replace(
        "window.__SCATTER_DATA__ = {};",
        f"window.__SCATTER_DATA__ = {json.dumps(layers_dict)};"
    )

    out = Path(f"scatter_output_{uuid.uuid4().hex[:8]}.html")
    out.write_text(html)

    return IFrame(str(out), width="100%", height="620")