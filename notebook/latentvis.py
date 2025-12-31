from IPython.display import HTML, display
from pathlib import Path
import json, uuid

DIST_BUNDLE_PATH = Path("dist/notebook-vis.js") 

def render_sankey(df_or_json, elem_id=None, height=480, bundle_path=DIST_BUNDLE_PATH):
    """
    Render a Sankey in the notebook.
    - df_or_json: pandas.DataFrame or list/dict (JSON-serializable) or JSON string
    - elem_id: optional element id; if None a unique id is created
    - height: container height in px
    - bundle_path: path to IIFE bundle (default 'dist/notebook-vis.js')
    """
    # ensure elem_id
    if elem_id is None:
        elem_id = f"sankey-{uuid.uuid4().hex[:8]}"

    # prepare JSON string for data
    if hasattr(df_or_json, "to_json"):
        data_json = df_or_json.to_json(orient="records")
    else:
        # if user passed list/dict or string
        if isinstance(df_or_json, str):
            # assume it's already a JSON string
            data_json = df_or_json
        else:
            data_json = json.dumps(df_or_json)

    # make sure bundle exists for first-time inline; if missing we still try fallback
    bundle_exists = bundle_path.exists()

    # Build HTML that:
    # 1) creates a container div for the sankey
    # 2) if bundle not loaded yet, inline the bundle + register window.__renderSankey
    # 3) then call window.__renderSankey with the provided data and elem_id
    # Note: we inject a minimal process polyfill before executing bundle to avoid `process is not defined`.
    html_parts = []

    html_parts.append(f'<div id="{elem_id}" style="height:{height}px; border:1px solid #eee; margin:6px 0;"></div>')

    # Inline helper + bundle only if not loaded on the page yet.
    # We'll attempt to detect via window.__NOTEBK_SANK_LOADED flag.
    # The inline script will set that flag.
    if bundle_exists:
        bundle_code = bundle_path.read_text()
        inline_bootstrap = f"""
<script>
(function() {{
  // if already initialized, don't re-inline bundle (just call render)
  if (window.__NOTEBK_SANK_LOADED) {{
    return;
  }}

  // minimal runtime polyfill for process if some deps reference it
  if (typeof window.process === 'undefined') {{
    window.process = {{ env: {{}} }};
  }}
  if (typeof globalThis.process === 'undefined') {{
    globalThis.process = window.process;
  }}

  // inline the IIFE bundle now
  try {{
    {bundle_code}
  }} catch (e) {{
    console.error("Error executing inlined Sankey bundle:", e);
    return;
  }}

  // Now the bundle should have exposed window.mountSankeyApp or window.NotebookVis.mountSankeyApp
  // Provide a small helper that normalizes mount invocation and stores a registry of mounts
  window.__NOTEBK_SANK_LOADED = true;
  window.__renderSankey_calls = window.__renderSankey_calls || [];

  window.__renderSankey = function(data, elemId, opts) {{
    // if mountSankeyApp is not available, try to look under NotebookVis
    var mount = window.mountSankeyApp || (window.NotebookVis && window.NotebookVis.mountSankeyApp);
    if (!mount) {{
      console.error("mountSankeyApp not found after bundle execution. Make sure the bundle exports mountSankeyApp to window.");
      return null;
    }}
    // call mount with explicit initialData to avoid relying on global single value
    var handle = mount(elemId, {{ initialData: data, dataUrl: opts && opts.dataUrl }});
    // store handle for debugging/unmount
    window.__renderSankey_calls.push({{elemId: elemId, handle: handle}});
    return handle;
  }};
}})();
</script>
"""
        html_parts.append(inline_bootstrap)
    else:
        # if bundle missing, still create a small loader that will try to fetch via Jupyter's /files/ path
        inline_loader = f"""
<script>
(function() {{
  if (window.__NOTEBK_SANK_LOADED) return;
  if (typeof window.process === 'undefined') {{
    window.process = {{ env: {{}} }};
  }}
  if (typeof globalThis.process === 'undefined') {{
    globalThis.process = window.process;
  }}

  try {{
    // try synchronous XHR read from Jupyter files URL as fallback
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/files/{bundle_path.as_posix()}', false);
    xhr.send(null);
    if (xhr.status === 200) {{
      var script = document.createElement('script');
      script.type = 'text/javascript';
      script.text = xhr.responseText;
      document.head.appendChild(script);
      // expose helper as above
      window.__NOTEBK_SANK_LOADED = true;
      window.__renderSankey_calls = window.__renderSankey_calls || [];
      window.__renderSankey = function(data, elemId, opts) {{
        var mount = window.mountSankeyApp || (window.NotebookVis && window.NotebookVis.mountSankeyApp);
        if (!mount) {{ console.error("mountSankeyApp not found after loading bundle via /files"); return null; }}
        var handle = mount(elemId, {{ initialData: data, dataUrl: opts && opts.dataUrl }});
        window.__renderSankey_calls.push({{elemId: elemId, handle: handle}});
        return handle;
      }};
    }} else {{
      console.warn('Could not load bundle via /files/; please run npm run build and ensure dist/notebook-vis.js exists.'); 
    }}
  }} catch (e) {{
    console.warn('Exception while trying to load bundle via /files/:', e);
  }}
}})();
</script>
"""
        html_parts.append(inline_loader)

    # Finally add a small script that calls the helper with the actual data (this will work whether bundle was inlined now or had been inlined earlier)
    call_script = f"""
<script>
(function() {{
  var data = {data_json};
  var attempts = 0;
  function tryRender() {{
    if (window.__renderSankey) {{
      try {{
        window.__renderSankey(data, "{elem_id}", {{}});
      }} catch (e) {{
        console.error("renderSankey call error:", e);
      }}
    }} else {{
      attempts++;
      if (attempts < 30) {{
        // wait a bit and retry (bundle may be executing)
        setTimeout(tryRender, 100);
      }} else {{
        console.error("renderSankey helper not available after waiting; please rerun the initialization cell that inlines the bundle.");
      }}
    }}
  }}
  tryRender();
}})();
</script>
"""
    html_parts.append(call_script)

    display(HTML("\n".join(html_parts)))
    return elem_id  # return the DOM id used
def render_scatter(df_or_json, elem_id=None, height=480, bundle_path=DIST_BUNDLE_PATH):
    if elem_id is None:
        elem_id = f"scatter-{uuid.uuid4().hex[:8]}"

    if hasattr(df_or_json, "to_json"):
        data_json = df_or_json.to_json(orient="records")
    else:
        if isinstance(df_or_json, str):
            data_json = df_or_json
        else:
            data_json = json.dumps(df_or_json)

    bundle_exists = bundle_path.exists()
    html_parts = []
    html_parts.append(f'<div id="{elem_id}" style="height:{height}px; border:1px solid #eee; margin:6px 0;"></div>')

    # inline bundle (if exists) and register a robust __renderScatter that probes many names
    if bundle_exists:
        bundle_code = bundle_path.read_text()
        inline_bootstrap = f"""
<script>
(function() {{
  if (window.__NOTEBK_SANK_LOADED && window.__renderScatter) return;

  if (typeof window.process === 'undefined') {{ window.process = {{ env: {{}} }}; }}
  if (typeof globalThis.process === 'undefined') {{ globalThis.process = window.process; }}

  try {{
    {bundle_code}
  }} catch (e) {{
    console.error("Error executing inlined Scatter bundle:", e);
    return;
  }}

  window.__NOTEBK_SANK_LOADED = true;
  window.__renderScatter_calls = window.__renderScatter_calls || [];

  function findScatterMount() {{
    // explicit candidate names
    var candidates = [
      window.mountScatterApp, window.mountScatter,
      (window.NotebookVis && window.NotebookVis.mountScatterApp),
      (window.NotebookVis && window.NotebookVis.mountScatter),
      window.mountApp, // legacy
      window.mountSankeyApp
    ];
    for (var i=0;i<candidates.length;i++) {{
      if (typeof candidates[i] === 'function') return candidates[i];
    }}
    // scan NotebookVis for any mount function whose name includes 'Scatter'
    if (window.NotebookVis) {{
      for (var k in window.NotebookVis) {{
        try {{
          if (typeof window.NotebookVis[k] === 'function' && /scatter/i.test(k)) {{
            return window.NotebookVis[k];
          }}
        }} catch (e) {{ /* ignore */ }}
      }}
    }}
    // scan globals for any function with 'Scatter' in its name (last resort)
    for (var g in window) {{
      try {{
        if (typeof window[g] === 'function' && /scatter/i.test(g)) return window[g];
      }} catch(e) {{ }}
    }}
    return null;
  }}

  window.__renderScatter = function(data, elemId, opts) {{
    var mount = findScatterMount();
    if (!mount) {{
      console.error("mountScatter not found after bundle execution. Available NotebookVis keys:", Object.keys(window.NotebookVis||{{}}));
      return null;
    }}
    var handle = mount(elemId, {{ initialData: data, dataUrl: opts && opts.dataUrl }});
    window.__renderScatter_calls.push({{elemId: elemId, handle: handle}});
    return handle;
  }};
}})();
</script>
"""
        html_parts.append(inline_bootstrap)
    else:
        # fallback loader - attempts /files/ path, same robust finder
        inline_loader = f"""
<script>
(function() {{
  if (window.__NOTEBK_SANK_LOADED && window.__renderScatter) return;

  if (typeof window.process === 'undefined') {{ window.process = {{ env: {{}} }}; }}
  if (typeof globalThis.process === 'undefined') {{ globalThis.process = window.process; }}

  try {{
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/files/{bundle_path.as_posix()}', false);
    xhr.send(null);
    if (xhr.status === 200) {{
      var script = document.createElement('script');
      script.type = 'text/javascript';
      script.text = xhr.responseText;
      document.head.appendChild(script);
      window.__NOTEBK_SANK_LOADED = true;
    }} else {{
      console.warn('Could not load bundle via /files/; please run npm run build and ensure {bundle_path.as_posix()} exists.'); 
    }}
  }} catch (e) {{
    console.warn('Exception while trying to load bundle via /files/:', e);
  }}

  // same finder logic (kept simple here)
  function findScatterMount() {{
    var candidates = [
      window.mountScatterApp, window.mountScatter,
      (window.NotebookVis && window.NotebookVis.mountScatterApp),
      (window.NotebookVis && window.NotebookVis.mountScatter),
      window.mountApp, window.mountSankeyApp
    ];
    for (var i=0;i<candidates.length;i++) {{
      if (typeof candidates[i] === 'function') return candidates[i];
    }}
    if (window.NotebookVis) {{
      for (var k in window.NotebookVis) {{
        try {{
          if (typeof window.NotebookVis[k] === 'function' && /scatter/i.test(k)) return window.NotebookVis[k];
        }} catch (e) {{ }}
      }}
    }}
    for (var g in window) {{
      try {{
        if (typeof window[g] === 'function' && /scatter/i.test(g)) return window[g];
      }} catch(e) {{ }}
    }}
    return null;
  }}

  window.__renderScatter_calls = window.__renderScatter_calls || [];
  window.__renderScatter = function(data, elemId, opts) {{
    var mount = findScatterMount();
    if (!mount) {{
      console.error("mountScatter not found after loading bundle via /files; NotebookVis keys:", Object.keys(window.NotebookVis||{{}}));
      return null;
    }}
    var handle = mount(elemId, {{ initialData: data, dataUrl: opts && opts.dataUrl }});
    window.__renderScatter_calls.push({{elemId: elemId, handle: handle}});
    return handle;
  }};
}})();
</script>
"""
        html_parts.append(inline_loader)

    call_script = f"""
<script>
(function() {{
  var data = {data_json};
  var attempts = 0;
  function tryRender() {{
    if (window.__renderScatter) {{
      try {{
        window.__renderScatter(data, "{elem_id}", {{}});
      }} catch (e) {{
        console.error("renderScatter call error:", e);
      }}
    }} else {{
      attempts++;
      if (attempts < 30) {{
        setTimeout(tryRender, 100);
      }} else {{
        console.error("renderScatter helper not available after waiting; please ensure the bundle exposes mountScatterApp or NotebookVis.mountScatterApp.");
      }}
    }}
  }}
  tryRender();
}})();
</script>
"""
    html_parts.append(call_script)
    display(HTML("\n".join(html_parts)))
    return elem_id


# Optional demo when running as script (quick test)
if __name__ == "__main__":
    try:
        import pandas as pd

        demo_df = pd.DataFrame(
            [
                {"source": "A", "target": "B", "value": 10},
                {"source": "A", "target": "C", "value": 5},
            ]
        )
        print("Running demo render_sankey (must be in a Jupyter notebook to see output).")
        render_sankey(demo_df)
    except Exception as e:
        print("Demo failed (are you in a notebook?). Error:", e)