import plotly.graph_objs as go
from ipywidgets import VBox, HBox, Button, ColorPicker, Text, Output, HTML
import numpy as np

class MarkBoard:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.N = len(x)

        # 原始颜色（随机）
        self.original_colors = [
            f"#{np.random.randint(0, 0xFFFFFF):06x}" for _ in range(self.N)
        ]
        self.colors = list(self.original_colors)

        # plot
        self.fig = go.FigureWidget(
            data=[
                go.Scatter(
                    x=self.x,
                    y=self.y,
                    mode="markers",
                    marker=dict(size=8, color=self.colors)
                )
            ],
            layout=go.Layout(
                title="MarkBoard",
                dragmode="lasso",
                height=500,
                width=700
            )
        )
        
        self.scatter = self.fig.data[0]

        # 存储 mark
        self.marks = []
        self.mark_counter = 0
        self.last_selected_indices = []

        # tooltip 面板（右侧 UI）
        self.color_picker = ColorPicker(
            description='Color',
            value='#e41a1c'
        )

        self.label_text = Text(
            description='Label',
            value='mark-1'
        )

        self.annotation_text = Text(
            description='Note',
            placeholder='annotation (optional)'
        )

        self.create_btn = Button(
            description='Create Mark',
            button_style='success'
        )

        self.mark_list = VBox([])

        self.info = Output()

        # 绑定事件
        self.scatter.on_selection(self._handle_selection)
        # Removed the following line as per instructions
        # self.create_btn.on_click(self._handle_create_mark)

    # 回调：用户在图上选点
    def _handle_selection(self, trace, points, selector):
        self.last_selected_indices = points.point_inds
        n = len(self.last_selected_indices)

        with self.info:
            print(f"Selected {n} points.")

        # Only exit lasso if 0 points are selected
        if n == 0:
            self.fig.layout.dragmode = "zoom"

    # 回调：用户创建 mark
    def _handle_create_mark(self, btn):
        print("BUTTON CLICKED")
        if not self.last_selected_indices:
            with self.info:
                print("No points selected.")
            return
        
        self.mark_counter += 1
        label = self.label_text.value
        color = self.color_picker.value
        annotation = self.annotation_text.value

        # (no longer update colors here)

        mark = {
            "id": self.mark_counter,
            "label": label,
            "color": color,
            "annotation": annotation,
            "indices": list(self.last_selected_indices)
        }
        self.marks.append(mark)

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

    # 高亮某个 mark（点击 icon 调用）
    def _highlight_mark(self, mark_id):
        # toggle behavior
        if hasattr(self, "active_mark") and self.active_mark == mark_id:
            # reset
            with self.fig.batch_update():
                self.scatter.marker.opacity = [1.0] * self.N
            self.active_mark = None
            return

        mark = next(m for m in self.marks if m["id"] == mark_id)
        indices = mark["indices"]

        new_opacity = [0.2] * self.N
        for idx in indices:
            new_opacity[idx] = 1.0

        with self.fig.batch_update():
            self.scatter.marker.opacity = new_opacity

        self.active_mark = mark_id

    # 展示 UI
    def show(self):
        self.create_btn.on_click(self._handle_create_mark)
        right_panel = VBox([
            self.color_picker,
            self.label_text,
            self.annotation_text,
            self.create_btn,
            HTML("<b>Marks:</b>"),
            self.mark_list,
            self.info
        ])

        display(HBox([self.fig, right_panel]))