#!/usr/bin/env python
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     # Only what importing the scripts costs; the runner carries their own deps.
#     "numpy",
# ]
# ///
"""A window over the scripts in this folder: pick a model, point it at images, run it.

    python gui.py
    uv run examples/gui.py

tkinter, which is in the standard library, so this adds nothing to install.

The form is built from each script's own `arguments()` parser, so a flag added to a
script appears here with no edit. Only settings moved off their default are passed, and
the command is shown before it runs. Outputs are named from the output folder and the
input's name: `cells.leishmania-seg.png`, `-labels.tif`, `-rois.zip`.

A folder is one run per image in it, and the scripts name each image's outputs
themselves, so for a folder the window hands them the output folder alone.

The runner is this interpreter where it can already import what the script needs, else
`uv run`, which reads the script's inline dependencies.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from fetch import IMAGE_SUFFIXES
from outputs import OUTPUT_SET

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# module -> (what it has to be able to import to run, one line for the picker,
#            the framework and task whose index entries it can run)
SCRIPTS = {
    "detect": ("ultralytics", "a box round every cell", ("ultralytics", "detect")),
    "segment": ("ultralytics", "one mask per cell, split into body and flagellum",
                ("ultralytics", "segment")),
    "pose": ("ultralytics", "eight ordered points, head to flagellar tip",
             ("ultralytics", "pose")),
    "mask2former": ("transformers", "Mask2Former, body and flagellum",
                    ("hf-transformers", None)),
    "cellpose_seg": ("cellpose", "cellpose 4, one mask per cell", ("cellpose", None)),
    "guide": ("ultralytics", "the tile plan alone, without segmenting anything",
              ("ultralytics", "detect")),
}

# Fields the index suggests values for. Editable comboboxes, filled in the background:
# the index lists what is published, not what may be typed.
FROM_INDEX = {"id", "version", "guide_version"}

DETECTOR = "leishmania-detect"  # what --guide runs, whatever script is in front of it

# Flags this window lays out itself instead of generating. The outputs come from
# outputs.py, so adding a fourth there is enough.
TAKEN = {"help", "image", "frames"} | {name for name, _, _ in OUTPUT_SET}

# Built from the scripts' own suffix list, so the picker cannot disagree with them.
IMAGE_TYPES = [("images", " ".join("*" + suffix for suffix in IMAGE_SUFFIXES)),
               ("all files", "*.*")]

# Fields that hold a path on disk, and so get a file picker beside the entry.
BROWSE = {"weights"}


def state_file() -> Path:
    """Beside the weights cache, so one environment variable moves both."""

    from fetch import CACHE

    return CACHE / "gui.json"


def kind_of(action) -> str:
    """Which widget an argparse action wants.

    A BooleanOptionalAction (--x with a --no-x) is tristate: its third state, neither
    given, leaves the script's own default alone and has to stay reachable.
    """
    if action.nargs != 0:
        return "value"
    if len(action.option_strings) > 1 and any(o.startswith("--no-")
                                              for o in action.option_strings):
        return "tristate"
    return "switch"


def default_text(action) -> str:
    """What an untouched field shows, which is also what counts as "not given"."""
    if action.default is None:
        return ""
    if isinstance(action.default, (list, tuple)):
        return " ".join(str(item) for item in action.default)
    return str(action.default)


def quoted(command) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in command)


def scrollable(parent) -> ttk.Frame:
    """A frame that scrolls, which tkinter has no widget for. Returns the inner frame.

    Call `takes_wheel` on it once its rows exist.
    """
    canvas = tk.Canvas(parent, highlightthickness=0, borderwidth=0)
    bar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    inner = ttk.Frame(canvas)
    window = canvas.create_window((0, 0), window=inner, anchor="nw")
    canvas.configure(yscrollcommand=bar.set)
    canvas.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    inner.bind("<Configure>",
               lambda event: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.bind("<Configure>",
                lambda event: canvas.itemconfigure(window, width=event.width))
    inner.canvas = canvas
    return inner


def takes_wheel(inner) -> None:
    """Let the wheel scroll `inner`'s canvas from anywhere inside it.

    Bound per widget, not via bind_all on <Enter>: the rows are children of the canvas,
    so the pointer moving onto one fires <Leave> on the canvas, and a wheel event walks
    the target's bindtags rather than its parents.
    """
    canvas = inner.canvas

    def turned(event):
        canvas.yview_scroll(-(event.delta // 120), "units")
        return "break"

    def bind(widget):
        widget.bind("<MouseWheel>", turned)
        for child in widget.winfo_children():
            bind(child)

    bind(canvas)
    bind(inner)


class Field:
    """One argparse action as a row: its flag, a widget, and its help beside it.

    The widget starts at the action's own default and adds nothing to the command line
    until it is moved off it, so a run here is the equivalent one-line command.
    """

    def __init__(self, parent, action, row: int, changed, offered: bool = False):
        self.action = action
        self.flag = action.option_strings[0]
        self.kind = kind_of(action)
        ttk.Label(parent, text=self.flag, anchor="e").grid(
            row=row, column=0, sticky="e", padx=(6, 8), pady=4)

        if self.kind == "switch":
            self.var = tk.BooleanVar(value=bool(action.default))
            self.widget = ttk.Checkbutton(parent, variable=self.var)
        elif self.kind == "tristate":
            self.var = tk.StringVar(value="")
            self.widget = ttk.Combobox(parent, textvariable=self.var, width=12,
                                       state="readonly", values=("", "yes", "no"))
        elif action.choices:
            values = [str(choice) for choice in action.choices]
            if action.default is None:
                values.insert(0, "")
            self.var = tk.StringVar(value=default_text(action))
            self.widget = ttk.Combobox(parent, textvariable=self.var, width=14,
                                       state="readonly", values=values)
        elif offered:
            # Editable: the index lists what is published, but a version missing from
            # this copy of it still has to be typeable.
            self.var = tk.StringVar(value=default_text(action))
            self.widget = ttk.Combobox(parent, textvariable=self.var, width=26)
        elif action.dest in BROWSE:
            # The picker only fills the entry in, so a path can still be typed. A
            # mask2former checkpoint FOLDER is named by picking any file inside it.
            self.var = tk.StringVar(value=default_text(action))
            mount = ttk.Frame(parent)
            self.widget = ttk.Entry(mount, textvariable=self.var, width=34)
            self.widget.pack(side="left")
            ttk.Button(mount, text="file...", width=7,
                       command=self.browse).pack(side="left", padx=(4, 0))
        else:
            self.var = tk.StringVar(value=default_text(action))
            self.widget = ttk.Entry(parent, textvariable=self.var, width=18)
        (mount if action.dest in BROWSE else self.widget).grid(
            row=row, column=1, sticky="w")

        self.base = (action.help or "").strip()
        if action.nargs == 2:
            self.base = f"two numbers, e.g. {default_text(action) or '1 99'}. {self.base}"
        if self.kind == "tristate":
            self.base = f"blank leaves the script's default. {self.base}"
        self.note = ttk.Label(parent, text=self.base, wraplength=540, justify="left",
                              foreground="#586069")
        self.note.grid(row=row, column=2, sticky="w", padx=(12, 6), pady=4)
        self.var.trace_add("write", lambda *_: changed())

    def browse(self) -> None:
        chosen = filedialog.askopenfilename(title=self.flag)
        if chosen:
            self.var.set(chosen)

    def offer(self, values) -> None:
        """The values the index has for this field, where the widget can show a list."""
        if isinstance(self.widget, ttk.Combobox) and str(self.widget["state"]) != "readonly":
            self.widget.configure(values=list(values))

    def hint(self, text: str) -> None:
        """What the index records for this setting, said beside the flag that sets it."""
        self.note.configure(text=f"{self.base}  ·  {text}" if text else self.base)

    def argv(self) -> list:
        """What this field adds to the command, which is nothing until it is moved."""
        if self.kind == "switch":
            return [self.flag] if self.var.get() else []
        text = str(self.var.get()).strip()
        if self.kind == "tristate":
            if not text:
                return []
            negative = next(o for o in self.action.option_strings
                            if o.startswith("--no-"))
            return [self.flag if text == "yes" else negative]
        if not text or text == default_text(self.action):
            return []
        if self.action.nargs == 2:
            return [self.flag, *text.split()]
        return [self.flag, text]

    def get(self):
        return self.var.get()

    def set(self, value) -> None:
        try:
            self.var.set(value)
        except tk.TclError:  # a saved value of the wrong shape for this widget
            pass


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("leishmania-models")
        self.modules: dict = {}
        self.fields: dict = {}
        # Which script the fields on screen belong to. The picker's value has already
        # moved on when the form is rebuilt, so settings cannot be keyed on it.
        self.showing = None
        self.parser = None
        self.process = None
        self.lines: queue.Queue = queue.Queue()
        self.index = None            # the published models, once they have arrived
        self.arriving: queue.Queue = queue.Queue()
        self.saved = self._load_state()

        self.script = tk.StringVar(value=self.saved.get("script", "segment"))
        self.image = tk.StringVar(value=self.saved.get("image", ""))
        self.folder = tk.StringVar(value=self.saved.get("folder", ""))
        self.frames = tk.StringVar(value=self.saved.get("frames", "0"))
        self.runner = tk.StringVar()
        self.wants = {name: tk.BooleanVar(
            value=self.saved.get("wants", {}).get(name, name == "out"))
            for name, _, _ in OUTPUT_SET}

        self._build()
        self.select()
        self.read_index()
        # self.script is absent on purpose: <<ComboboxSelected>> already calls select().
        # A trace here ran first, pairing the new name with the old script's fields and
        # importing from inside a Tcl callback, where failures escape select().
        for var in (self.image, self.folder, self.frames, self.runner,
                    *self.wants.values()):
            var.trace_add("write", lambda *_: self.refresh())
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    # building the window

    def _build(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        top = ttk.LabelFrame(outer, text="model", padding=8)
        top.pack(fill="x")
        picker = ttk.Combobox(top, textvariable=self.script, state="readonly", width=18,
                              values=list(SCRIPTS))
        picker.grid(row=0, column=0, sticky="w")
        picker.bind("<<ComboboxSelected>>", lambda event: self.select())
        self.about = ttk.Label(top, text="", foreground="#586069")
        self.about.grid(row=0, column=1, sticky="w", padx=10)
        ttk.Label(top, text="run with").grid(row=0, column=2, sticky="e", padx=(10, 4))
        self.runners = ttk.Combobox(top, textvariable=self.runner, state="readonly",
                                    width=26)
        self.runners.grid(row=0, column=3, sticky="e")
        self.entry_note = ttk.Label(top, text="", foreground="#586069",
                                    wraplength=780, justify="left")
        self.entry_note.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Button(top, text="Reload index", width=14,
                   command=self.read_index).grid(row=1, column=3, sticky="e",
                                                 pady=(6, 0))
        top.columnconfigure(1, weight=1)

        files = ttk.LabelFrame(outer, text="input and output", padding=8)
        files.pack(fill="x", pady=(8, 0))
        ttk.Label(files, text="input").grid(row=0, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(files, textvariable=self.image).grid(row=0, column=1, sticky="ew")
        ttk.Button(files, text="file...", width=8,
                   command=self.pick_file).grid(row=0, column=2, padx=(6, 0))
        ttk.Button(files, text="folder...", width=9,
                   command=self.pick_input_folder).grid(row=0, column=3, padx=(4, 0))
        ttk.Label(files, text="a stack, one image, or a folder: one run per image in it",
                  foreground="#586069").grid(row=1, column=1, sticky="w", pady=(0, 6))

        ttk.Label(files, text="output").grid(row=2, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(files, textvariable=self.folder).grid(row=2, column=1, sticky="ew")
        ttk.Button(files, text="open", width=8,
                   command=self.open_output).grid(row=2, column=2, padx=(6, 0))
        ttk.Button(files, text="folder...", width=9,
                   command=self.pick_output).grid(row=2, column=3, padx=(4, 0))
        writes = ttk.Frame(files)
        writes.grid(row=3, column=1, sticky="w", pady=(2, 6))
        self.boxes = {}
        for name, _, text in OUTPUT_SET:
            self.boxes[name] = ttk.Checkbutton(writes, text=text,
                                               variable=self.wants[name])
            self.boxes[name].pack(side="left", padx=(0, 12))

        ttk.Label(files, text="frames").grid(row=4, column=0, sticky="e", padx=(0, 6))
        ttk.Entry(files, textvariable=self.frames, width=18).grid(row=4, column=1,
                                                                  sticky="w")
        self.frames_note = ttk.Label(files, text="", foreground="#586069")
        self.frames_note.grid(row=5, column=1, sticky="w")
        files.columnconfigure(1, weight=1)

        # Packed to the bottom first, so they keep their place whatever the screen is;
        # the settings and the log then share what is left, across a draggable divider.
        buttons = ttk.Frame(outer)
        buttons.pack(side="bottom", fill="x", pady=(8, 0))

        preview = ttk.LabelFrame(outer, text="the command this runs", padding=8)
        preview.pack(side="bottom", fill="x", pady=(8, 0))
        self.command_box = tk.Text(preview, height=3, wrap="word", relief="flat",
                                   background="#f6f8fa", foreground="#24292e")
        self.command_box.pack(fill="x")
        self.command_box.configure(state="disabled")

        split = ttk.PanedWindow(outer, orient="vertical")
        split.pack(fill="both", expand=True, pady=(8, 0))
        self.book = ttk.Notebook(split)
        split.add(self.book, weight=3)
        self.run_button = ttk.Button(buttons, text="Run", command=self.run)
        self.run_button.pack(side="left")
        self.stop_button = ttk.Button(buttons, text="Stop", command=self.stop,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=6)
        ttk.Button(buttons, text="Copy command",
                   command=self.copy).pack(side="left", padx=6)
        ttk.Button(buttons, text="Reset settings",
                   command=lambda: self.select(reset=True)).pack(side="left", padx=6)
        self.status = ttk.Label(buttons, text="", foreground="#586069")
        self.status.pack(side="left", padx=12)

        log = ttk.LabelFrame(split, text="output", padding=6)
        split.add(log, weight=2)
        self.log_box = tk.Text(log, height=8, wrap="word", relief="flat",
                               background="#101418", foreground="#d6dae0")
        bar = ttk.Scrollbar(log, orient="vertical", command=self.log_box.yview)
        self.log_box.configure(yscrollcommand=bar.set, state="disabled")
        self.log_box.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")

    # the script, and the form built from its parser

    def module(self):
        name = self.script.get()
        if name not in self.modules:
            self.modules[name] = importlib.import_module(name)
        return self.modules[name]

    def select(self, reset: bool = False) -> None:
        """Rebuild the form for the chosen script, one tab per group of its parser."""
        self.remember()
        for tab in self.book.tabs():
            self.book.forget(tab)
        self.fields = {}
        name = self.script.get()
        if reset:
            self.saved.get("settings", {}).pop(name, None)
        self.showing = name
        self.about.configure(text=SCRIPTS.get(name, ("", ""))[1])
        try:
            self.parser = self.module().arguments()
        except Exception as broken:
            # Carry on with parser None: a bare return would leave the boxes, the
            # runner list and the preview describing the previous script.
            self.parser = None
            self.say(f"{name}.py: could not read its settings: {broken}\n")

        # _action_groups is the only place --help's grouping lives; no public API.
        for group in (self.parser._action_groups if self.parser else ()):
            actions = [action for action in group._group_actions
                       if action.dest not in TAKEN and action.option_strings]
            if not actions:
                continue
            page = ttk.Frame(self.book)
            self.book.add(page, text="model" if group.title == "options" else group.title)
            rows = scrollable(page)
            rows.columnconfigure(2, weight=1)
            for row, action in enumerate(actions):
                self.fields[action.dest] = Field(rows, action, row, self.refresh,
                                                 offered=action.dest in FROM_INDEX)
            takes_wheel(rows)

        offered = self.parser._option_string_actions if self.parser else {}
        frames = offered.get("--frames")
        self.frames_note.configure(text=(frames.help if frames else
                                         "this script runs one frame at a time"))
        self.writes_state()
        if not reset:
            for dest, value in self.saved.get("settings", {}).get(name, {}).items():
                if dest in self.fields:
                    self.fields[dest].set(value)
        self.runners.configure(values=self.runner_choices())
        if self.runner.get() not in self.runner_choices():
            self.runner.set(self.runner_choices()[0])
        self.refresh()

    def writes_state(self) -> None:
        """Grey the box for any output the script has no flag for; guide.py has none.

        The box is disabled and the tick left alone: clearing it would turn that output
        off for every other script too, and `close` would save that.
        """
        offered = self.parser._option_string_actions if self.parser else {}
        for name, box in self.boxes.items():
            box.configure(state="normal" if f"--{name}" in offered else "disabled")

    # what the index knows

    def read_index(self) -> None:
        """Fetch the index in the background and fill in what it says.

        In the background because it is a URL by default: everything it supplies is a
        suggestion beside a field that already works without it.
        """
        self.entry_note.configure(text="reading the index...")

        def work():
            try:
                from fetch import load_index

                self.arriving.put(load_index()["models"])
            except Exception as failed:  # a URL, a file, or neither: all the same here
                self.arriving.put(failed)

        threading.Thread(target=work, daemon=True).start()
        self.root.after(150, self._check_index)

    def _check_index(self) -> None:
        try:
            arrived = self.arriving.get_nowait()
        except queue.Empty:
            self.root.after(150, self._check_index)
            return
        if isinstance(arrived, Exception):
            self.index = None
            self.entry_note.configure(
                text=f"the index could not be read ({arrived}). Every field still "
                     f"works; nothing is filled in from it.")
            return
        self.index = arrived
        self.apply_index()

    def published(self) -> list:
        """The index entries the chosen script is the one that runs."""
        if not self.index:
            return []
        framework, task = SCRIPTS.get(self.script.get(), ("", "", (None, None)))[2]
        return [entry for entry in self.index
                if entry.get("framework") == framework
                and (task is None or entry.get("task") == task)]

    def versions_of(self, model_id: str) -> list:
        return [entry["version"] for entry in (self.index or [])
                if entry.get("id") == model_id]

    def entry(self) -> dict:
        """The entry a run would resolve to right now: the id in force, at the version
        in force, else the newest. Via `fetch.newest`, so this and the run agree."""
        from fetch import newest

        matches = [item for item in (self.index or [])
                   if item.get("id") == self.model_name()]
        asked = self.fields["version"].get().strip() if "version" in self.fields else ""
        if asked:
            matches = [item for item in matches if item.get("version") == asked]
        return newest(matches) if matches else {}

    def tiling_is_the_entry_s(self) -> bool:
        """Whether this script's --tile means the tile the resolved entry records.

        True except for guide.py, whose --tile is the segmenter's while the entry in
        force is the detector's.
        """
        return self.script.get() != "guide"

    def apply_index(self) -> None:
        """Offer the ids and versions there are, and say what the entry already sets.

        The hints matter more than the lists: `--tile` shows 0 where the entry records
        640, resolved inside the run where nothing had been showing it.
        """
        if self.index is None or self.parser is None:
            return
        if "id" in self.fields:
            self.fields["id"].offer(sorted({e["id"] for e in self.published()}))
        entry = self.entry()
        if "version" in self.fields:
            self.fields["version"].offer([""] + sorted(self.versions_of(self.model_name())))
        if "guide_version" in self.fields:
            self.fields["guide_version"].offer([""] + sorted(self.versions_of(DETECTOR)))

        config = entry.get("config") or {}
        tiling = (config.get("tiling") or {}) if self.tiling_is_the_entry_s() else {}
        records = {}
        for dest, value in (("tile", tiling.get("tile")),
                            ("overlap", tiling.get("overlap")),
                            ("imgsz", (config.get("detection") or {}).get("imgsz")),
                            ("conf", (config.get("detection") or {}).get("conf"))):
            if value is not None:
                records[dest] = f"the index records {value} for this model"
        for key, value in (config.get("cellpose") or {}).items():
            if key in self.fields:
                records[key] = f"the index sets {value} for this model"
        for dest, field in self.fields.items():
            field.hint(records.get(dest, ""))

        local = self.fields.get("weights")
        if local is not None and local.get().strip():
            self.entry_note.configure(
                text=f"a checkpoint on disk: {local.get().strip()}. The index is "
                     f"not consulted for it.")
        elif entry:
            scale = (entry.get("assumes") or {}).get("pixel_size_um")
            # A list means trained across magnifications; only its ends say anything.
            if isinstance(scale, (list, tuple)) and scale:
                scale = (f"{min(scale):g}-{max(scale):g}" if len(scale) > 1
                         else f"{scale[0]:g}")
            elif scale is not None:
                scale = f"{scale:g}"
            self.entry_note.configure(
                text=f"{entry['id']} {entry['version']} — "
                     f"{entry.get('notes') or entry.get('framework', '')}"
                     + (f"  ·  {scale} um/px" if scale else ""))
        else:
            known = ", ".join(sorted({e["id"] for e in self.published()})) or "nothing"
            self.entry_note.configure(
                text=f"{self.model_name()} is not in the index. This script runs: "
                     f"{known}." + ("  --weights runs an unpublished checkpoint."
                                    if "weights" in self.fields else ""))

    # what to run it with

    def runner_choices(self) -> list:
        """This interpreter first where it can already import what the script needs.

        A conda environment with ultralytics in it should not have uv fetch a second
        copy of torch. Otherwise `uv run`, which reads the inline dependency block.
        """
        needs = SCRIPTS.get(self.script.get(), ("", ""))[0]
        try:
            available = importlib.util.find_spec(needs) is not None
        except (ImportError, ValueError):
            available = False
        this = f"this python ({Path(sys.executable).name})"
        uv = "uv run (fetches what it needs)"
        if not shutil.which("uv"):
            return [this]
        return [this, uv] if available else [uv, this]

    def command(self) -> list:
        if self.parser is None:
            return []
        run = ["uv", "run"] if self.runner.get().startswith("uv") else [sys.executable]
        command = run + [str(HERE / f"{self.script.get()}.py")]
        if self.image.get().strip():
            command.append(str(self.here(self.image.get())))
        frames = self.parser._option_string_actions.get("--frames")
        if frames and self.frames.get().strip() not in ("", frames.default):
            command += ["--frames", self.frames.get().strip()]
        for field in self.fields.values():
            command += field.argv()
        return command + self.output_argv()

    def output_argv(self) -> list:
        """The output paths, named from the output folder and the input's name:
        `<folder>/<input stem>.<model>.png` and its two neighbours.

        For a folder of images the scripts name each image's outputs themselves, so
        they get the output folder alone -- unless --sequence runs the folder as one
        movie, which is named after the folder like any one input.
        """
        folder = self.folder.get().strip()
        image = self.image.get().strip()
        if not folder or not image or self.parser is None:
            return []
        stem = Path(image).stem or Path(image).name
        model = self.model_name()
        where = self.here(folder)
        sequence = self.fields.get("sequence")
        per_image = self.input_is_folder() and not (sequence is not None
                                                    and sequence.get())
        command = []
        for name, suffix, _ in OUTPUT_SET:
            if self.wants[name].get() and f"--{name}" in self.parser._option_string_actions:
                command += [f"--{name}", str(where if per_image
                                             else where / f"{stem}.{model}{suffix}")]
        return command

    def here(self, path) -> Path:
        """A path as the RUN will see it.

        The child is launched with cwd=examples/, and the copied command could be
        pasted anywhere, so everything on the command line is absolute.
        """
        return Path(str(path).strip()).expanduser().resolve()

    def model_name(self) -> str:
        """What to call the files: --weights' stem, else --id, else the script's own
        MODEL -- the order the scripts name their outputs in."""
        local = self.fields.get("weights")
        if local is not None and local.get().strip():
            name = Path(local.get().strip()).stem
            if name:
                return name
        chosen = self.fields.get("id")
        if chosen is not None and chosen.get().strip():
            return chosen.get().strip()
        return getattr(self.module(), "MODEL", self.script.get())

    def input_is_folder(self) -> bool:
        text = self.image.get().strip()
        try:
            return bool(text) and self.here(text).is_dir()
        except (OSError, ValueError):  # half-typed, or not a path at all
            return False

    # running it

    def refresh(self) -> None:
        # Recomputed here, not only on a script change: the hints follow --id/--version.
        self.apply_index()
        self.command_box.configure(state="normal")
        self.command_box.delete("1.0", "end")
        self.command_box.insert("1.0", quoted(self.command()))
        self.command_box.configure(state="disabled")

    def run(self) -> None:
        if self.process is not None:
            return
        image = Path(self.image.get().strip()) if self.image.get().strip() else None
        if image is None or not image.exists():
            self.say("pick an input first: a .tif stack, an image, or a folder.\n")
            return
        if self.output_argv():
            self.here(self.folder.get()).mkdir(parents=True, exist_ok=True)
        command = self.command()
        self.clear()
        self.say(quoted(command) + "\n\n")
        try:
            self.process = subprocess.Popen(
                command, cwd=str(HERE), stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1,
                # Unbuffered, or the output arrives in one lump at the end.
                env=dict(os.environ, PYTHONUNBUFFERED="1"),
                # Its own process group, so `stop` can reach the whole tree: the
                # runner may be `uv run`, whose python is a grandchild.
                **({} if sys.platform == "win32" else {"start_new_session": True}))
        except (OSError, ValueError) as failed:
            self.say(f"could not start it: {failed}\n")
            self.process = None
            return
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status.configure(text="running...")
        threading.Thread(target=self.pump, args=(self.process,), daemon=True).start()
        self.root.after(80, self.drain)

    def pump(self, process) -> None:
        for line in process.stdout:
            self.lines.put(line)
        process.stdout.close()
        self.lines.put(("done", process.wait()))

    def drain(self) -> None:
        finished = None
        while True:
            try:
                item = self.lines.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                finished = item[1]
            else:
                self.say(item)
        if finished is None:
            self.root.after(80, self.drain)
            return
        self.process = None
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status.configure(text="done" if finished == 0
                              else f"exit {finished}")
        self.say(f"\n[{'finished' if finished == 0 else f'exit {finished}'}]\n")

    def stop(self, quietly: bool = False) -> None:
        """End the run, and everything it started.

        `terminate` alone is not enough: `uv run` is a launcher, and killing it leaves
        the python it spawned holding the pipe, so the reader never sees EOF and the
        window sits on "running..." for the rest of the session.
        """
        if self.process is None:
            return
        if not quietly:
            self.say("\n[stopping]\n")
        try:
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.process.pid)],
                               capture_output=True)
            else:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        except (OSError, subprocess.SubprocessError, PermissionError):
            pass
        self.process.terminate()  # whatever the tree kill could not reach

    # odds and ends

    def pick_file(self) -> None:
        chosen = filedialog.askopenfilename(title="an image or a stack",
                                            filetypes=IMAGE_TYPES,
                                            initialdir=self._near())
        if chosen:
            self.image.set(chosen)

    def pick_input_folder(self) -> None:
        chosen = filedialog.askdirectory(title="a folder of images",
                                         initialdir=self._near())
        if chosen:
            self.image.set(chosen)

    def open_output(self) -> None:
        """Show where the outputs land, in the system's own file browser.

        The output folder where one is set, else the input's parent, which is where
        the scripts write when no path is given.
        """
        target = self.folder.get().strip()
        if not target and self.image.get().strip():
            target = str(self.here(self.image.get()).parent)
        if not target:
            self.status.configure(text="nothing to open: pick an input or an "
                                       "output folder")
            return
        where = self.here(target)
        if not where.is_dir():
            self.status.configure(text=f"not there yet: {where}")
            return
        if sys.platform == "win32":
            os.startfile(str(where))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(where)])
        else:
            subprocess.Popen(["xdg-open", str(where)])

    def pick_output(self) -> None:
        chosen = filedialog.askdirectory(title="where the outputs go",
                                         initialdir=self.folder.get() or self._near())
        if chosen:
            self.folder.set(chosen)

    def _near(self) -> str:
        current = Path(self.image.get().strip() or ".")
        return str(current if current.is_dir() else current.parent)

    def copy(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(quoted(self.command()))
        self.status.configure(text="command copied")

    def say(self, text: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", text)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def clear(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def remember(self) -> None:
        """Save the fields on screen before the form is torn down.

        Under `showing`, the script they were built for, not the picker's value.
        """
        if not self.fields or self.showing is None:
            return
        self.saved.setdefault("settings", {})[self.showing] = {
            dest: field.get() for dest, field in self.fields.items()}

    def _load_state(self) -> dict:
        try:
            return json.loads(state_file().read_text(encoding="utf-8"))
        except Exception:
            # A missing or half-written file just means a window with its defaults.
            return {}

    def close(self) -> None:
        self.remember()
        self.saved.update(script=self.script.get(), image=self.image.get(),
                          folder=self.folder.get(), frames=self.frames.get(),
                          wants={name: var.get() for name, var in self.wants.items()})
        try:
            state_file().parent.mkdir(parents=True, exist_ok=True)
            state_file().write_text(json.dumps(self.saved, indent=2), encoding="utf-8")
        except OSError:
            pass
        self.stop(quietly=True)
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    elif "clam" in style.theme_names():
        style.theme_use("clam")
    # Sized to the screen: a window taller than the desktop puts the Run button under
    # the taskbar. Tk is already told the scaling factor, so nothing here reapplies it.
    wide = min(1100, root.winfo_screenwidth() - 80)
    tall = min(820, root.winfo_screenheight() - 120)
    root.geometry(f"{wide}x{tall}")
    root.minsize(min(840, wide), min(560, tall))
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
