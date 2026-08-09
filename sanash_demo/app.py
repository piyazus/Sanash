from __future__ import annotations

import queue
import sys
import threading
import traceback
from pathlib import Path
from tkinter import BOTH, DISABLED, END, HORIZONTAL, LEFT, NORMAL, RIGHT, X, filedialog, messagebox
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

from .inference import (
    IMAGE_EXTENSIONS,
    MODEL_EXTENSIONS,
    PassengerCounter,
    PredictionResult,
    find_default_image_source,
    find_default_model,
    list_images,
    load_rgb_image,
)


COLORS = {
    "app_bg": "#edf2f1",
    "surface": "#f8faf9",
    "surface_alt": "#eef3f2",
    "border": "#d3dddb",
    "text": "#17211f",
    "muted": "#66736f",
    "header": "#10201e",
    "header_text": "#f7fbf9",
    "header_muted": "#a9bbb6",
    "accent": "#0f9f8f",
    "accent_hover": "#0b7f73",
    "accent_soft": "#d9f3ef",
    "warning": "#d97706",
    "canvas": "#111817",
    "canvas_grid": "#20312e",
}


class ImagePane(ttk.Frame):
    def __init__(self, parent: tk.Widget, title: str, empty_text: str) -> None:
        super().__init__(parent, style="Pane.TFrame")
        self.image: Image.Image | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.empty_text = empty_text
        self.title_var = tk.StringVar(value=title)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Label(self, textvariable=self.title_var, style="PaneTitle.TLabel").grid(
            row=0,
            column=0,
            sticky="w",
            padx=12,
            pady=(10, 6),
        )
        self.canvas = tk.Canvas(
            self,
            bg=COLORS["canvas"],
            bd=0,
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["accent"],
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.canvas.bind("<Configure>", lambda _event: self._redraw())

    def set_title(self, title: str) -> None:
        self.title_var.set(title)

    def set_image(self, image: Image.Image | None) -> None:
        self.image = image
        self._redraw()

    def _redraw(self) -> None:
        self.canvas.delete("all")
        canvas_width = max(1, self.canvas.winfo_width())
        canvas_height = max(1, self.canvas.winfo_height())
        if self.image is None:
            self._draw_empty_state(canvas_width, canvas_height)
            return

        image_width, image_height = self.image.size
        scale = min(canvas_width / image_width, canvas_height / image_height)
        display_size = (max(1, int(image_width * scale)), max(1, int(image_height * scale)))
        display = self.image.resize(display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        x = (canvas_width - display_size[0]) // 2
        y = (canvas_height - display_size[1]) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.photo)

    def _draw_empty_state(self, canvas_width: int, canvas_height: int) -> None:
        inset = 22
        if canvas_width > inset * 2 and canvas_height > inset * 2:
            self.canvas.create_rectangle(
                inset,
                inset,
                canvas_width - inset,
                canvas_height - inset,
                outline=COLORS["canvas_grid"],
                width=1,
            )
        self.canvas.create_text(
            canvas_width // 2,
            canvas_height // 2,
            text=self.empty_text,
            fill="#8da19c",
            font=("Segoe UI", 13, "bold"),
        )


class SanashApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sanash Passenger Counter")
        self.geometry("1320x860")
        self.minsize(1120, 740)
        self.configure(bg=COLORS["app_bg"])

        self.project_dir = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
        self.counter = PassengerCounter()
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_result: PredictionResult | None = None
        self.image_paths: list[Path] = []
        self.current_index = 0
        self.loaded_model_path: Path | None = None
        self.busy = False

        self.model_var = tk.StringVar()
        self.source_var = tk.StringVar()
        self.model_display_var = tk.StringVar(value="Sanash P2PNet")
        self.model_detail_var = tk.StringVar(value="Built-in model")
        self.source_display_var = tk.StringVar(value="Demo samples")
        self.threshold_var = tk.DoubleVar(value=0.40)
        self.threshold_text = tk.StringVar(value="0.40")
        self.status_var = tk.StringVar(value="Ready")
        self.count_var = tk.StringVar(value="-")
        self.time_var = tk.StringVar(value="-")
        self.image_var = tk.StringVar(value="-")
        self.backend_var = tk.StringVar(value="-")

        default_model = find_default_model(self.project_dir)
        if default_model is not None:
            self.model_var.set(str(default_model))
            self.model_display_var.set("Sanash P2PNet")
            self.model_detail_var.set("Built-in passenger counting model")
        else:
            self.model_display_var.set("Model not found")
            self.model_detail_var.set("Add the model archive next to the application")
        default_source = find_default_image_source(self.project_dir)
        if default_source is not None:
            self.source_var.set(str(default_source))
            self._load_source(default_source)

        self._configure_style()
        self._set_window_icon()
        self._build_ui()
        self._bind_shortcuts()
        self._show_current_image()
        self.after(100, self._poll_worker)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=COLORS["app_bg"], foreground=COLORS["text"])
        style.configure("App.TFrame", background=COLORS["app_bg"])
        style.configure("Card.TFrame", background=COLORS["surface"])
        style.configure("Pane.TFrame", background=COLORS["surface"])
        style.configure("Header.TFrame", background=COLORS["header"])
        style.configure(
            "Header.TLabel",
            background=COLORS["header"],
            foreground=COLORS["header_text"],
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "SubHeader.TLabel",
            background=COLORS["header"],
            foreground=COLORS["header_muted"],
            font=("Segoe UI", 10),
        )
        style.configure(
            "HeaderStatus.TLabel",
            background=COLORS["header"],
            foreground="#d9f3ef",
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Card.TLabelframe",
            background=COLORS["surface"],
            bordercolor=COLORS["border"],
            relief="solid",
        )
        style.configure(
            "Card.TLabelframe.Label",
            background=COLORS["surface"],
            foreground=COLORS["muted"],
            font=("Segoe UI", 10, "bold"),
        )
        style.configure("TEntry", fieldbackground="#ffffff", bordercolor=COLORS["border"], lightcolor=COLORS["accent"])
        style.configure("TButton", padding=(10, 7), background="#eef3f2", foreground=COLORS["text"])
        style.map("TButton", background=[("active", "#e0e9e7"), ("disabled", "#edf2f1")])
        style.configure("Primary.TButton", background=COLORS["accent"], foreground="#ffffff", padding=(12, 9))
        style.map(
            "Primary.TButton",
            background=[("active", COLORS["accent_hover"]), ("disabled", "#93c8bf")],
            foreground=[("disabled", "#eef7f5")],
        )
        style.configure("MetricLabel.TLabel", background=COLORS["surface"], foreground=COLORS["muted"], font=("Segoe UI", 9, "bold"))
        style.configure("MetricValue.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=("Segoe UI", 10))
        style.configure("ModelName.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=("Segoe UI", 14, "bold"))
        style.configure("InfoText.TLabel", background=COLORS["surface"], foreground=COLORS["muted"], font=("Segoe UI", 9))
        style.configure("Count.TLabel", background=COLORS["surface"], foreground=COLORS["accent"], font=("Segoe UI", 44, "bold"))
        style.configure("PaneTitle.TLabel", background=COLORS["surface"], foreground=COLORS["text"], font=("Segoe UI", 11, "bold"))
        style.configure("Status.TLabel", background="#dfe8e6", foreground=COLORS["muted"], padding=(12, 6))
        style.configure(
            "Horizontal.TProgressbar",
            background=COLORS["accent"],
            troughcolor=COLORS["surface_alt"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["accent"],
            darkcolor=COLORS["accent"],
        )

    def _set_window_icon(self) -> None:
        icon = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon, "RGBA")
        draw.rounded_rectangle((5, 5, 59, 59), radius=15, fill=(15, 159, 143, 255))
        draw.rounded_rectangle((14, 17, 50, 47), radius=8, fill=(16, 32, 30, 255))
        draw.ellipse((21, 22, 31, 32), fill=(248, 250, 249, 255))
        draw.ellipse((35, 22, 45, 32), fill=(248, 250, 249, 255))
        draw.rectangle((18, 39, 47, 43), fill=(217, 243, 239, 255))
        self._app_icon = ImageTk.PhotoImage(icon)
        self.iconphoto(True, self._app_icon)

    def _build_ui(self) -> None:
        header = ttk.Frame(self, style="Header.TFrame")
        header.pack(fill=X)

        brand = ttk.Frame(header, style="Header.TFrame")
        brand.pack(fill=X, padx=20, pady=16)
        mark = tk.Canvas(brand, width=42, height=42, bg=COLORS["header"], bd=0, highlightthickness=0)
        mark.pack(side=LEFT, padx=(0, 12))
        mark.create_rectangle(2, 2, 40, 40, fill=COLORS["accent"], outline="")
        mark.create_oval(10, 13, 18, 21, fill="#f8faf9", outline="")
        mark.create_oval(24, 13, 32, 21, fill="#f8faf9", outline="")
        mark.create_rectangle(10, 28, 32, 31, fill="#d9f3ef", outline="")

        title_block = ttk.Frame(brand, style="Header.TFrame")
        title_block.pack(side=LEFT, fill=X, expand=True)
        ttk.Label(title_block, text="Sanash Passenger Counter", style="Header.TLabel").pack(anchor="w")
        ttk.Label(title_block, text="Local P2PNet inference workspace", style="SubHeader.TLabel").pack(anchor="w")
        ttk.Label(brand, textvariable=self.status_var, style="HeaderStatus.TLabel").pack(side=RIGHT, padx=(16, 0))

        content = ttk.Frame(self, style="App.TFrame")
        content.pack(fill=BOTH, expand=True, padx=18, pady=16)

        sidebar = ttk.Frame(content, style="App.TFrame", width=354)
        sidebar.pack(side=LEFT, fill="y", padx=(0, 14))
        sidebar.pack_propagate(False)

        model_frame = ttk.LabelFrame(sidebar, text="Model", style="Card.TLabelframe", padding=(12, 10))
        model_frame.pack(fill=X, pady=(0, 12))
        ttk.Label(model_frame, textvariable=self.model_display_var, style="ModelName.TLabel").pack(anchor="w")
        ttk.Label(model_frame, textvariable=self.model_detail_var, style="InfoText.TLabel", wraplength=300).pack(
            anchor="w",
            pady=(2, 10),
        )
        model_buttons = ttk.Frame(model_frame, style="Card.TFrame")
        model_buttons.pack(fill=X)
        self.inspect_button = ttk.Button(model_buttons, text="Model details", command=self._inspect_model_clicked)
        self.inspect_button.pack(fill=X)

        source_frame = ttk.LabelFrame(sidebar, text="Images", style="Card.TLabelframe", padding=(12, 10))
        source_frame.pack(fill=X, pady=(0, 12))
        ttk.Label(source_frame, textvariable=self.source_display_var, style="ModelName.TLabel", wraplength=300).pack(
            anchor="w",
            pady=(0, 10),
        )
        source_buttons = ttk.Frame(source_frame, style="Card.TFrame")
        source_buttons.pack(fill=X)
        self.open_image_button = ttk.Button(source_buttons, text="Open Image", command=self._browse_image)
        self.open_folder_button = ttk.Button(source_buttons, text="Open Folder", command=self._browse_folder)
        for column, button in enumerate((self.open_image_button, self.open_folder_button)):
            source_buttons.columnconfigure(column, weight=1, uniform="source_buttons")
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 4, 0 if column == 1 else 4))

        action_frame = ttk.LabelFrame(sidebar, text="Inference", style="Card.TLabelframe", padding=(12, 10))
        action_frame.pack(fill=X, pady=(0, 12))
        self.run_button = ttk.Button(action_frame, text="Run Inference", style="Primary.TButton", command=self._run_inference)
        self.run_button.pack(fill=X, pady=(0, 9))
        nav_buttons = ttk.Frame(action_frame, style="Card.TFrame")
        nav_buttons.pack(fill=X, pady=(0, 9))
        self.prev_button = ttk.Button(nav_buttons, text="Previous", command=self._previous_image)
        self.next_button = ttk.Button(nav_buttons, text="Next", command=self._next_image)
        for column, button in enumerate((self.prev_button, self.next_button)):
            nav_buttons.columnconfigure(column, weight=1, uniform="nav_buttons")
            button.grid(row=0, column=column, sticky="ew", padx=(0 if column == 0 else 4, 0 if column == 1 else 4))
        self.save_button = ttk.Button(action_frame, text="Save Annotated", command=self._save_annotated)
        self.save_button.pack(fill=X, pady=(0, 12))

        threshold_header = ttk.Frame(action_frame, style="Card.TFrame")
        threshold_header.pack(fill=X)
        ttk.Label(threshold_header, text="Threshold", style="MetricLabel.TLabel").pack(side=LEFT)
        ttk.Label(threshold_header, textvariable=self.threshold_text, style="MetricValue.TLabel").pack(side=RIGHT)
        self.threshold_control = ttk.Scale(
            action_frame,
            from_=0.05,
            to=0.95,
            orient=HORIZONTAL,
            variable=self.threshold_var,
            command=self._threshold_changed,
        )
        self.threshold_control.pack(fill=X, pady=(3, 10))
        self.progress = ttk.Progressbar(action_frame, mode="indeterminate", style="Horizontal.TProgressbar")
        self.progress.pack(fill=X)

        metrics_frame = ttk.LabelFrame(sidebar, text="Result", style="Card.TLabelframe", padding=(12, 10))
        metrics_frame.pack(fill=X)
        ttk.Label(metrics_frame, text="Count", style="MetricLabel.TLabel").pack(anchor="w")
        ttk.Label(metrics_frame, textvariable=self.count_var, style="Count.TLabel").pack(anchor="w", pady=(0, 6))
        self._add_metric(metrics_frame, "Inference", self.time_var)
        self._add_metric(metrics_frame, "Backend", self.backend_var)
        self._add_metric(metrics_frame, "Image", self.image_var)

        main = ttk.Frame(content, style="App.TFrame")
        main.pack(side=LEFT, fill=BOTH, expand=True)
        panes = ttk.PanedWindow(main, orient=HORIZONTAL)
        panes.pack(fill=BOTH, expand=True)
        self.original_pane = ImagePane(panes, "Original", "No image")
        self.annotated_pane = ImagePane(panes, "Annotated", "Awaiting result")
        panes.add(self.original_pane, weight=1)
        panes.add(self.annotated_pane, weight=1)

        diagnostics_frame = ttk.LabelFrame(main, text="Diagnostics", style="Card.TLabelframe", padding=(8, 7))
        diagnostics_frame.pack(fill=X, pady=(12, 0))
        self.diagnostics = tk.Text(
            diagnostics_frame,
            height=4,
            bg="#fbfdfc",
            fg="#35423f",
            relief="flat",
            bd=0,
            wrap="word",
            font=("Consolas", 9),
            insertwidth=0,
        )
        self.diagnostics.pack(fill=X)
        self.diagnostics.insert(END, "")
        self.diagnostics.configure(state=DISABLED)

        status = ttk.Label(self, textvariable=self.status_var, style="Status.TLabel", anchor="w")
        status.pack(fill=X, side="bottom")
        self._refresh_buttons()

    def _add_metric(self, parent: tk.Widget, label: str, variable: tk.StringVar) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill=X, pady=3)
        ttk.Label(row, text=label, style="MetricLabel.TLabel", width=10).pack(side=LEFT)
        ttk.Label(row, textvariable=variable, style="MetricValue.TLabel", wraplength=205).pack(side=LEFT, fill=X, expand=True)

    def _bind_shortcuts(self) -> None:
        self.bind("<Left>", lambda _event: self._previous_image())
        self.bind("<Right>", lambda _event: self._next_image())
        self.bind("<Return>", lambda _event: self._run_inference())

    def _threshold_changed(self, _value: str) -> None:
        self.threshold_text.set(f"{self.threshold_var.get():.2f}")

    def _browse_model(self) -> None:
        extensions = " ".join(f"*{ext}" for ext in sorted(MODEL_EXTENSIONS))
        path = filedialog.askopenfilename(title="Select model", filetypes=[("Model files", extensions), ("All files", "*.*")])
        if path:
            self.model_var.set(path)
            self.loaded_model_path = None
            self.counter = PassengerCounter()
            self.model_display_var.set("Custom P2PNet model")
            self.model_detail_var.set(Path(path).name)
            self.backend_var.set("-")
            self._set_diagnostics("")
            self._set_status("Model selected")

    def _browse_image(self) -> None:
        extensions = " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS))
        path = filedialog.askopenfilename(title="Select image", filetypes=[("Image files", extensions), ("All files", "*.*")])
        if path:
            image_path = Path(path)
            self.source_var.set(str(image_path))
            self.source_display_var.set(f"{image_path.name} (1 image)")
            self.image_paths = [image_path]
            self.current_index = 0
            self._show_current_image()

    def _browse_folder(self) -> None:
        path = filedialog.askdirectory(title="Select image folder")
        if path:
            self.source_var.set(path)
            self._load_source(Path(path))
            self._show_current_image()

    def _load_source(self, path: Path) -> None:
        try:
            if path.is_file():
                self.image_paths = [path]
            else:
                self.image_paths = list_images(path)
            self.current_index = 0
            if not self.image_paths:
                self.source_display_var.set("No images found")
                self._set_status("No images found")
            else:
                source_name = "Demo samples" if path.name == "demo_samples" else path.name
                image_word = "image" if len(self.image_paths) == 1 else "images"
                self.source_display_var.set(f"{source_name} ({len(self.image_paths)} {image_word})")
                self._set_status(f"Loaded {len(self.image_paths)} image(s)")
        except Exception as exc:
            messagebox.showerror("Image Error", str(exc))

    def _show_current_image(self) -> None:
        self.current_result = None
        self.annotated_pane.set_image(None)
        self.count_var.set("-")
        self.time_var.set("-")
        self._set_diagnostics("")
        if not self.image_paths:
            self.original_pane.set_image(None)
            self.image_var.set("-")
            self.original_pane.set_title("Original")
            self.annotated_pane.set_title("Annotated")
            self._refresh_buttons()
            return
        path = self.image_paths[self.current_index]
        try:
            image = load_rgb_image(path)
            self.original_pane.set_image(image)
            suffix = f"{self.current_index + 1} / {len(self.image_paths)}"
            self.original_pane.set_title(f"Original - {path.name} ({suffix})")
            self.annotated_pane.set_title("Annotated")
            self.image_var.set(path.name)
            self._set_status(f"Ready: {path.name}")
        except Exception as exc:
            self.original_pane.set_image(None)
            messagebox.showerror("Image Error", str(exc))
        self._refresh_buttons()

    def _previous_image(self) -> None:
        if self.busy or not self.image_paths:
            return
        self.current_index = (self.current_index - 1) % len(self.image_paths)
        self._show_current_image()

    def _next_image(self) -> None:
        if self.busy or not self.image_paths:
            return
        self.current_index = (self.current_index + 1) % len(self.image_paths)
        self._show_current_image()

    def _load_model_clicked(self) -> None:
        self._start_worker("load", self._load_model_worker)

    def _inspect_model_clicked(self) -> None:
        self._start_worker("inspect", self._inspect_model_worker)

    def _run_inference(self) -> None:
        if not self.image_paths:
            source = Path(self.source_var.get().strip()) if self.source_var.get().strip() else None
            if source is not None and source.exists():
                self._load_source(source)
            if not self.image_paths:
                messagebox.showwarning("Images", "Select an image or folder first.")
                return
        self._start_worker("predict", self._predict_worker)

    def _save_annotated(self) -> None:
        if self.current_result is None:
            messagebox.showwarning("Save", "Run inference before saving.")
            return
        source = self.current_result.image_path
        default_name = f"{source.stem}_sanash_count{self.current_result.count}.jpg"
        path = filedialog.asksaveasfilename(
            title="Save annotated image",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.current_result.annotated_image.save(path, quality=94)
            self._set_status(f"Saved {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def _current_model_path(self) -> Path:
        text = self.model_var.get().strip()
        if not text:
            raise ValueError("Select a model file first.")
        path = Path(text)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        return path

    def _load_model_worker(self) -> object:
        path = self._current_model_path()
        summary = self.counter.load_model(path)
        self.loaded_model_path = path.resolve()
        return summary

    def _inspect_model_worker(self) -> object:
        if self.loaded_model_path != self._current_model_path().resolve() or self.counter.summary is None:
            self._load_model_worker()
        return self.counter.summary

    def _predict_worker(self) -> object:
        model_path = self._current_model_path().resolve()
        if self.loaded_model_path != model_path or self.counter.summary is None:
            self._load_model_worker()
        image_path = self.image_paths[self.current_index]
        return self.counter.predict_image(image_path, threshold=float(self.threshold_var.get()))

    def _start_worker(self, label: str, func) -> None:
        if self.busy:
            return
        self.busy = True
        self.progress.start(8)
        self._refresh_buttons()
        self._set_status({"load": "Loading model...", "inspect": "Inspecting model...", "predict": "Running inference..."}[label])

        def run() -> None:
            try:
                result = func()
                self.worker_queue.put((label, result))
            except Exception as exc:
                self.worker_queue.put(("error", (exc, traceback.format_exc())))

        threading.Thread(target=run, daemon=True).start()

    def _poll_worker(self) -> None:
        try:
            while True:
                label, payload = self.worker_queue.get_nowait()
                self.busy = False
                self.progress.stop()
                self._handle_worker_result(label, payload)
                self._refresh_buttons()
        except queue.Empty:
            pass
        self.after(100, self._poll_worker)

    def _handle_worker_result(self, label: str, payload: object) -> None:
        if label == "error":
            exc, trace = payload
            self._set_status("Error")
            self._set_diagnostics(str(trace))
            messagebox.showerror("Sanash Demo", str(exc))
            return
        if label == "load":
            summary = payload
            self.backend_var.set(summary.backend)
            self.model_display_var.set("Sanash P2PNet")
            self.model_detail_var.set(f"Loaded with {summary.backend}")
            self._set_status(f"Loaded model: {summary.resolved_path.name}")
            self._set_diagnostics(summary.details)
            return
        if label == "inspect":
            summary = payload
            self.backend_var.set(summary.backend)
            self.model_display_var.set("Sanash P2PNet")
            self.model_detail_var.set(f"Loaded with {summary.backend}")
            self._set_status("Model inspection ready")
            self._set_diagnostics(summary.details)
            self._show_summary_dialog(summary.details)
            return
        if label == "predict":
            result = payload
            self.current_result = result
            self.annotated_pane.set_image(result.annotated_image)
            self.annotated_pane.set_title(f"Annotated - {result.image_path.name}")
            self.count_var.set(str(result.count))
            self.time_var.set(f"{result.inference_ms:.0f} ms")
            self.backend_var.set(result.backend)
            self.model_display_var.set("Sanash P2PNet")
            self.model_detail_var.set(f"Loaded with {result.backend}")
            self._set_status(f"Done: {result.image_path.name}")
            self._set_diagnostics(result.diagnostics)

    def _show_summary_dialog(self, details: str) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Model Inspection")
        dialog.geometry("740x440")
        dialog.configure(bg=COLORS["app_bg"])
        text = tk.Text(dialog, wrap="word", bg="#fbfdfc", fg=COLORS["text"], relief="flat", font=("Consolas", 10))
        text.pack(fill=BOTH, expand=True, padx=14, pady=14)
        text.insert(END, details)
        text.configure(state=DISABLED)
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=(0, 14))
        dialog.transient(self)
        dialog.grab_set()

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _set_diagnostics(self, text: str) -> None:
        self.diagnostics.configure(state=NORMAL)
        self.diagnostics.delete("1.0", END)
        self.diagnostics.insert(END, text)
        self.diagnostics.configure(state=DISABLED)

    def _refresh_buttons(self) -> None:
        state = DISABLED if self.busy else NORMAL
        buttons = (
            self.inspect_button,
            self.open_image_button,
            self.open_folder_button,
            self.prev_button,
            self.run_button,
            self.next_button,
            self.save_button,
        )
        for button in buttons:
            button.configure(state=state)
        self.threshold_control.configure(state=state)
        if not self.image_paths or len(self.image_paths) <= 1 or self.busy:
            self.prev_button.configure(state=DISABLED)
            self.next_button.configure(state=DISABLED)
        if self.current_result is None or self.busy:
            self.save_button.configure(state=DISABLED)


def main() -> None:
    app = SanashApp()
    app.mainloop()


if __name__ == "__main__":
    main()
