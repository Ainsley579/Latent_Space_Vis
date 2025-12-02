import plotly.graph_objs as go
from ipywidgets import VBox, HBox, Button, ColorPicker, Text, Output, HTML
import numpy as np

class MarkBoard:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.N = len(x)

        # 颜色初始化
        self.colors = ["lightgray"] * self.N

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
        self.create_btn.on_click(self._handle_create_mark)

    # 回调：用户在图上选点
    def _handle_selection(self, trace, points, selector):
        self.last_selected_indices = points.point_inds
        with self.info:
            print(f"Selected {len(self.last_selected_indices)} points.")

    # 回调：用户创建 mark
    def _handle_create_mark(self, btn):
        if not self.last_selected_indices:
            with self.info:
                print("No points selected.")
            return
        
        self.mark_counter += 1
        label = self.label_text.value
        color = self.color_picker.value
        annotation = self.annotation_text.value

        # 更新颜色
        for idx in self.last_selected_indices:
            self.colors[idx] = color

        with self.fig.batch_update():
            self.scatter.marker.color = self.colors

        mark = {
            "id": self.mark_counter,
            "label": label,
            "color": color,
            "annotation": annotation,
            "indices": list(self.last_selected_indices)
        }
        self.marks.append(mark)

        # 在右侧加一个 icon
        icon = HTML(f"""
        <div style="display:flex;align-items:center;cursor:pointer;"
             onclick="IPython.notebook.kernel.execute('mb._highlight_mark({self.mark_counter})')">
            <div style="width:12px;height:12px;background:{color};border-radius:50%;margin-right:8px;"></div>
            <span>{label}</span>
        </div>
        """)

        self.mark_list.children += (icon,)

    # 高亮某个 mark（点击 icon 调用）
    def _highlight_mark(self, mark_id):
        mark = next(m for m in self.marks if m["id"] == mark_id)
        indices = mark["indices"]
        color = mark["color"]

        # 将所有点变暗
        new_colors = ["#cccccc"] * self.N
        for idx in indices:
            new_colors[idx] = color

        with self.fig.batch_update():
            self.scatter.marker.color = new_colors

    # 展示 UI
    def show(self):
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