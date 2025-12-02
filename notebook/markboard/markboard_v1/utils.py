    def _handle_selection(self, trace, points, selector):
        ...
        n = len(self.last_selected_indices)

        with self.info:
            print(f"Selected {n} points.")

        # Only exit lasso if 0 points are selected
        if n == 0:
            self.fig.layout.dragmode = "zoom"

    def _handle_create_mark(self, b):
        ...
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
