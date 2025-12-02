@@ -_handle_selection
-        # ... existing code ...
-
-        # Auto-exit lasso mode
-        self.fig.layout.dragmode = "zoom"
-
-        return
+        n = len(self.last_selected_indices)
+
+        with self.info:
+            print(f"Selected {n} points.")
+
+        # Only exit lasso if 0 points are selected
+        if n == 0:
+            self.fig.layout.dragmode = "zoom"
+
+        return
