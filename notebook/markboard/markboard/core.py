import plotly.graph_objs as go
from ipywidgets import VBox, HBox, Button, ColorPicker, Text, Output, HTML
from IPython.display import display
import numpy as np
from copy import deepcopy


class MarkBoard:
    """
    Generic MarkBoard that can be attached to ANY Plotly Figure / FigureWidget.

    Usage in a notebook:

        import plotly.express as px
        from markboard import MarkBoard

        fig = px.scatter(df, x="a", y="b")
        board = MarkBoard.attach(fig)
        board.show()   # displays fig + right panel

    Marks work for all traces that support point-level interaction (click / selection).
    Hover is NOT recorded.
    """

    def __init__(self, fig_widget: go.FigureWidget):
        # Always store a FigureWidget internally
        self.fig: go.FigureWidget = fig_widget
        self.traces = list(self.fig.data)

        if not self.traces:
            raise ValueError("Figure has no traces; cannot attach MarkBoard.")

        # Per-trace point counts
        self.trace_lengths = []
        for tr in self.traces:
            n = 0
            if hasattr(tr, "x") and tr.x is not None:
                try:
                    n = len(tr.x)
                except TypeError:
                    n = 0
            elif hasattr(tr, "y") and tr.y is not None:
                try:
                    n = len(tr.y)
                except TypeError:
                    n = 0
            self.trace_lengths.append(n)

        # Cache original opacity so we can restore
        self.base_marker_opacity = []
        self.base_trace_opacity = []
        for i, tr in enumerate(self.traces):
            marker = getattr(tr, "marker", None)
            if marker is not None and hasattr(marker, "opacity") and marker.opacity is not None:
                self.base_marker_opacity.append(deepcopy(marker.opacity))
            else:
                self.base_marker_opacity.append(None)

            self.base_trace_opacity.append(getattr(tr, "opacity", None))

        # Mark related state
        self.marks = []              # list of mark dicts
        self.mark_counter = 0
        self.active_mark = None

        # Pending selection from last click / lasso / box
        # List of {trace_index, point_inds}
        self.pending_selection = []

        # Widgets (right-side panel)
        self.color_picker = ColorPicker(
            description="Color",
            value="#e41a1c",
        )

        self.label_text = Text(
            description="Label",
            value="mark-1",
        )

        self.annotation_text = Text(
            description="Note",
            placeholder="annotation (optional)",
        )

        self.create_btn = Button(
            description="Create Mark",
            button_style="success",
        )

        self.mark_list = VBox([])
        self.info = Output()

        # Bind events for all traces: click + selection (no hover)
        for ti, tr in enumerate(self.traces):
            # Selection (lasso/box)
            try:
                tr.on_selection(self._make_on_selection(ti))
            except Exception:
                pass
            # Click
            try:
                tr.on_click(self._make_on_click(ti))
            except Exception:
                pass

    # -------- Classmethods / constructors --------

    @classmethod
    def attach(cls, fig) -> "MarkBoard":
        """
        Attach MarkBoard to an existing Plotly figure.

        If `fig` is a go.Figure, it will be wrapped into a FigureWidget.
        If it is already a FigureWidget, it is used directly.

        Returns a MarkBoard instance; call `.show()` to display.
        """
        if isinstance(fig, go.FigureWidget):
            fw = fig
        else:
            # Wrap any go.Figure / px figure into FigureWidget
            fw = go.FigureWidget(fig)

        # Ensure default dragmode is lasso if not specified
        if getattr(fw.layout, "dragmode", None) is None:
            fw.layout.dragmode = "lasso"

        return cls(fw)

    @classmethod
    def from_xy(cls, x, y, **scatter_kwargs) -> "MarkBoard":
        """
        Backwards-compatible helper: create a new scatter FigureWidget
        from x, y and attach MarkBoard to it.
        """
        fig = go.FigureWidget(
            data=[
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(size=8),
                    **scatter_kwargs,
                )
            ],
            layout=go.Layout(
                title="MarkBoard",
                dragmode="lasso",
                height=500,
                width=700,
            ),
        )
        return cls(fig)

    # -------- Trace event handlers --------

    def _make_on_selection(self, trace_index):
        def _handler(trace, points, selector):
            inds = list(points.point_inds) if points.point_inds is not None else []
            # Latest selection overrides previous pending selection for simplicity
            if inds:
                self.pending_selection = [
                    {"trace_index": trace_index, "point_inds": inds}
                ]
            else:
                self.pending_selection = []

            with self.info:
                print(f"Selection on trace {trace_index}: {len(inds)} point(s).")

        return _handler

    def _make_on_click(self, trace_index):
        def _handler(trace, points, selector):
            inds = list(points.point_inds) if points.point_inds is not None else []
            # If user clicks, we treat this as a new pending selection
            if inds:
                self.pending_selection = [
                    {"trace_index": trace_index, "point_inds": inds}
                ]
            else:
                self.pending_selection = []

            with self.info:
                print(f"Click on trace {trace_index}: {len(inds)} point(s).")

        return _handler

    # -------- Mark creation & list --------

    def _handle_create_mark(self, btn):
        if not self.pending_selection:
            with self.info:
                print("No points selected (via click or lasso).")
            return

        self.mark_counter += 1
        label = self.label_text.value or f"mark-{self.mark_counter}"
        color = self.color_picker.value
        annotation = self.annotation_text.value

        mark = {
            "id": self.mark_counter,
            "label": label,
            "color": color,
            "annotation": annotation,
            "selections": deepcopy(self.pending_selection),
        }
        self.marks.append(mark)

        # Add a button for this mark
        icon_btn = Button(
            description=label,
            layout={"width": "auto"},
        )
        icon_btn.style.button_color = color

        # Bind highlight callback
        icon_btn.on_click(lambda b, mid=self.mark_counter: self._highlight_mark(mid))

        current = list(self.mark_list.children)
        current.append(icon_btn)
        self.mark_list.children = tuple(current)

        # Prepare for next mark
        self.label_text.value = f"mark-{self.mark_counter + 1}"
        with self.info:
            print(f"Created mark {self.mark_counter} with {len(self.pending_selection)} selection group(s).")

    # -------- Highlight / reset logic --------

    def _reset_highlight(self):
        """
        Restore all traces' opacity to their original state.
        """
        for i, tr in enumerate(self.traces):
            n = self.trace_lengths[i]
            marker = getattr(tr, "marker", None)

            # Restore point-wise marker opacity if we had it
            if marker is not None and n > 0:
                base_op = self.base_marker_opacity[i]
                if isinstance(base_op, (list, tuple)):
                    # Already list-like
                    marker.opacity = deepcopy(base_op)
                elif base_op is not None:
                    # Scalar
                    marker.opacity = base_op
                else:
                    # If the trace is Bar-like or has no original per‑point opacity,
                    # restore to full opacity value (scalar 1.0) instead of a list.
                    # This correctly resets bar opacity and avoids invalid states.
                    try:
                        marker.opacity = 1.0
                    except Exception:
                        marker.opacity = [1.0] * n

            # Restore trace-level opacity if any
            if self.base_trace_opacity[i] is not None:
                tr.opacity = self.base_trace_opacity[i]
            else:
                # If it was None originally, keep as None
                if hasattr(tr, "opacity"):
                    tr.opacity = None

    def _apply_highlight(self, mark):
        """
        Default highlight strategy:
        - Dim all traces to low opacity (0.2).
        - For the mark's selected points, set those points back to full opacity (1.0).
        - Works on any trace that has a marker; falls back to trace-level opacity otherwise.
        Enhanced: For bar traces, highlight the entire bar trace, not individual bars.
        """
        # First dim everything
        for i, tr in enumerate(self.traces):
            n = self.trace_lengths[i]
            marker = getattr(tr, "marker", None)
            if marker is not None and n > 0:
                marker.opacity = [0.2] * n
            else:
                # No point-wise marker: dim the whole trace
                if hasattr(tr, "opacity"):
                    tr.opacity = 0.2

        # Then brighten selected points for this mark
        for sel in mark["selections"]:
            ti = sel["trace_index"]
            inds = sel["point_inds"]

            if ti < 0 or ti >= len(self.traces):
                continue

            tr = self.traces[ti]
            n = self.trace_lengths[ti]
            marker = getattr(tr, "marker", None)

            # ---- Special handling for Bar traces ----
            if getattr(tr, "type", None) == "bar":
                # Entire bar trace should be highlighted, not individual points.
                # Full opacity for selected bar trace:
                tr.opacity = 1.0
                continue

            # ---- Default highlight behavior for marker-based traces ----
            if marker is not None and n > 0:
                current_op = marker.opacity
                if isinstance(current_op, (list, tuple)):
                    op_list = list(current_op)
                elif current_op is not None:
                    op_list = [current_op] * n
                else:
                    op_list = [0.2] * n

                for idx in inds:
                    if 0 <= idx < n:
                        op_list[idx] = 1.0

                marker.opacity = op_list
                continue

            # ---- Fallback: brighten entire trace ----
            if hasattr(tr, "opacity"):
                tr.opacity = 1.0

    def _highlight_mark(self, mark_id):
        """
        Toggle highlight for a given mark.
        """
        # If clicking the active mark, reset everything
        if self.active_mark == mark_id:
            self._reset_highlight()
            self.active_mark = None
            with self.info:
                print(f"Cleared highlight for mark {mark_id}.")
            return

        # Find mark
        mark = next((m for m in self.marks if m["id"] == mark_id), None)
        if mark is None:
            with self.info:
                print(f"Mark {mark_id} not found.")
            return

        # Apply highlight
        self._apply_highlight(mark)
        self.active_mark = mark_id

        with self.info:
            total_points = sum(len(s["point_inds"]) for s in mark["selections"])
            print(f"Highlighted mark {mark_id} ({total_points} points).")

    # -------- Public API --------

    def show(self):
        """
        Display the figure + right-hand control panel in a notebook.
        """
        self.create_btn.on_click(self._handle_create_mark)

        right_panel = VBox(
            [
                self.color_picker,
                self.label_text,
                self.annotation_text,
                self.create_btn,
                HTML("<b>Marks:</b>"),
                self.mark_list,
                self.info,
            ]
        )
        display(HBox([self.fig, right_panel]))