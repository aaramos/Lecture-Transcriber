import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import (
    AudioQuality,
    BatchConfig,
    RecordingSpeed,
    SlideSensitivity,
    TranscriptionEngine,
)
from .errors import LectureProcessorError
from .media import ensure_media_tools
from .models import FileStatus
from .pipeline import BatchProcessor, discover_mov_files
from .slides import SlideExtractor
from .transcription import build_transcriber


class LectureProcessorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Lecture Processor")
        self.geometry("860x620")
        self.minsize(760, 540)
        self.result_queue = queue.Queue()
        self.worker = None

        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.recording_speed = tk.StringVar(value=RecordingSpeed.NORMAL.value)
        self.audio_quality = tk.StringVar(value=AudioQuality.FAST.value)
        self.save_normalized = tk.BooleanVar(value=True)
        self.transcription_engine = tk.StringVar(value=TranscriptionEngine.AUTO.value)
        self.whisper_model = tk.StringVar(value="large-v3")
        self.slide_sensitivity = tk.StringVar(value=SlideSensitivity.MEDIUM.value)
        self.concurrent_files = tk.IntVar(value=4)

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=(18, 16, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Lecture Processor", font=("Helvetica", 20, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Batch-process .mov lectures into playback-speed videos, transcripts, and slide images.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        controls = ttk.LabelFrame(self, text="Batch setup", padding=14)
        controls.grid(row=1, column=0, sticky="ew", padx=18, pady=8)
        controls.columnconfigure(1, weight=1)

        ttk.Label(controls, text="Lecture folder").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.input_dir).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(controls, text="Browse", command=self._browse_input).grid(row=0, column=2)

        ttk.Label(controls, text="Output folder").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(controls, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(controls, text="Browse", command=self._browse_output).grid(row=1, column=2)

        settings = ttk.Frame(controls)
        settings.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        for column in range(4):
            settings.columnconfigure(column, weight=1)

        self._combo(settings, "Recording speed", self.recording_speed, [item.value for item in RecordingSpeed], 0, 0)
        self._combo(settings, "Audio quality", self.audio_quality, [item.value for item in AudioQuality], 0, 1)
        self._combo(
            settings,
            "Transcription",
            self.transcription_engine,
            [item.value for item in TranscriptionEngine],
            0,
            2,
        )
        self._combo(
            settings,
            "Slide sensitivity",
            self.slide_sensitivity,
            [item.value for item in SlideSensitivity],
            0,
            3,
        )

        ttk.Label(settings, text="Whisper model").grid(row=2, column=0, sticky="w", pady=(10, 2))
        ttk.Combobox(
            settings,
            textvariable=self.whisper_model,
            values=["large-v3", "medium", "small"],
            state="readonly",
        ).grid(row=3, column=0, sticky="ew", padx=(0, 8))

        ttk.Label(settings, text="Concurrent files").grid(row=2, column=1, sticky="w", pady=(10, 2))
        ttk.Spinbox(settings, from_=1, to=8, textvariable=self.concurrent_files, width=6).grid(
            row=3, column=1, sticky="w"
        )

        ttk.Checkbutton(
            settings,
            text="Save normalized video",
            variable=self.save_normalized,
        ).grid(row=3, column=2, sticky="w")

        action_row = ttk.Frame(self, padding=(18, 0, 18, 10))
        action_row.grid(row=3, column=0, sticky="ew")
        action_row.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(action_row, mode="indeterminate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 12))
        self.open_output_button = ttk.Button(action_row, text="Open Output", command=self._open_output, state="disabled")
        self.open_output_button.grid(row=0, column=1, padx=(0, 8))
        self.start_button = ttk.Button(action_row, text="Start", command=self._start)
        self.start_button.grid(row=0, column=2)

        output = ttk.LabelFrame(self, text="Run log", padding=10)
        output.grid(row=2, column=0, sticky="nsew", padx=18, pady=8)
        output.rowconfigure(0, weight=1)
        output.columnconfigure(0, weight=1)
        self.log = tk.Text(output, wrap="word", height=14)
        self.log.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(output, command=self.log.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scrollbar.set)

    def _combo(self, parent, label, variable, values, row, column) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=(0, 2))
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly").grid(
            row=row + 1, column=column, sticky="ew", padx=(0, 8)
        )

    def _browse_input(self) -> None:
        folder = filedialog.askdirectory(title="Choose lecture folder")
        if folder:
            self.input_dir.set(folder)
            if not self.output_dir.get():
                path = Path(folder)
                self.output_dir.set(str(path.parent / f"{path.name}_processed"))

    def _browse_output(self) -> None:
        folder = filedialog.askdirectory(title="Choose output folder")
        if folder:
            self.output_dir.set(folder)

    def _start(self) -> None:
        try:
            config = self._build_config()
        except LectureProcessorError as exc:
            messagebox.showerror("Cannot start", str(exc))
            return

        if config.recording_speed is RecordingSpeed.DOUBLE:
            confirmed = messagebox.askyesno(
                "Confirm 2x normalization",
                "Speed normalization will double the duration of eligible files. "
                "Confirm that this batch was recorded at 2x speed.",
            )
            if not confirmed:
                return
            config = BatchConfig(**{**config.__dict__, "confirm_normalization": True})

        self._set_running(True)
        self.log.delete("1.0", "end")
        self._write_log("Starting batch...\n")
        self.worker = threading.Thread(target=self._run_batch, args=(config,), daemon=True)
        self.worker.start()
        self.after(100, self._poll_worker)

    def _build_config(self) -> BatchConfig:
        if not self.input_dir.get():
            raise LectureProcessorError("Choose a lecture folder first.")
        input_dir = Path(self.input_dir.get()).expanduser().resolve()
        output_dir = Path(self.output_dir.get()).expanduser().resolve() if self.output_dir.get() else input_dir.parent / f"{input_dir.name}_processed"
        return BatchConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            recording_speed=RecordingSpeed(self.recording_speed.get()),
            confirm_normalization=False,
            concurrent_files=int(self.concurrent_files.get()),
            save_normalized_video=bool(self.save_normalized.get()),
            audio_quality=AudioQuality(self.audio_quality.get()),
            slide_sensitivity=SlideSensitivity(self.slide_sensitivity.get()),
            transcription_engine=TranscriptionEngine(self.transcription_engine.get()),
            whisper_model=self.whisper_model.get(),
        )

    def _run_batch(self, config: BatchConfig) -> None:
        try:
            config.validate()
            if not discover_mov_files(config.input_dir):
                raise LectureProcessorError("No .mov files found. Try a different folder.")
            transcriber = build_transcriber(config.transcription_engine, config.whisper_model)
            ensure_media_tools(
                ffprobe_path=config.ffprobe_path,
                ffmpeg_path=config.ffmpeg_path,
                needs_ffmpeg=config.recording_speed is RecordingSpeed.DOUBLE,
            )
            summary = BatchProcessor(
                config=config,
                transcriber=transcriber,
                slide_extractor=SlideExtractor(config.slide_sensitivity),
            ).run()
            self.result_queue.put(("success", config, summary))
        except Exception as exc:
            self.result_queue.put(("error", config, exc))

    def _poll_worker(self) -> None:
        try:
            kind, config, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.after(100, self._poll_worker)
            return

        self._set_running(False)
        if kind == "error":
            self._write_log(f"\nError: {payload}\n")
            messagebox.showerror("Batch failed", str(payload))
            return

        self.output_dir.set(str(config.output_dir))
        self.open_output_button.configure(state="normal")
        self._write_log(_summary_text(payload))

    def _set_running(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        if running:
            self.progress.start(10)
            self.open_output_button.configure(state="disabled")
        else:
            self.progress.stop()

    def _write_log(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")

    def _open_output(self) -> None:
        if self.output_dir.get():
            subprocess.run(["open", self.output_dir.get()], check=False)


def _summary_text(summary) -> str:
    lines = [
        "\nBatch finished",
        f"Attempted: {summary.attempted}",
        f"Completed: {summary.completed}",
        f"Failed:    {summary.failed}",
        f"Skipped:   {summary.skipped}",
        f"Stopped:   {summary.stopped}",
        "",
    ]
    for result in summary.results:
        if result.status is FileStatus.COMPLETED:
            detail = f"{result.word_count} words, {result.slide_count} slides"
        elif result.status is FileStatus.FAILED:
            detail = f"{result.failure_step}: {result.message}"
        else:
            detail = result.message
        lines.append(f"- {result.source.name}: {result.status.value} ({detail})")
    return "\n".join(lines) + "\n"


def main() -> int:
    app = LectureProcessorApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
