#!/usr/bin/env python3
"""
IkePicFall — Image to Spectrogram Audio Converter

Converts images into WAV audio files such that the original image
becomes visible when the audio is viewed on a spectrogram (waterfall display).
Each pixel column is encoded as an AM-modulated carrier at a specific frequency.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageGrab, ImageEnhance
import io
import os

SAMPLE_RATE = 11025
MAX_LINES = 230


def generate_audio(image_array, spectral_width, delta_f, center_freq, duration):
    """Generate audio signal from a grayscale image array.

    Args:
        image_array: 2D numpy array (height x width), values 0-255, grayscale.
        spectral_width: Total spectral width in Hz.
        delta_f: Carrier spacing in Hz.
        center_freq: Center frequency in Hz.
        duration: Total duration in seconds.

    Returns:
        1D numpy array of 16-bit PCM samples.
    """
    num_carriers = int(spectral_width / delta_f)
    num_lines = image_array.shape[0]

    # Resize image to match num_carriers width if needed
    if image_array.shape[1] != num_carriers:
        img = Image.fromarray(image_array)
        img = img.resize((num_carriers, num_lines), Image.LANCZOS)
        image_array = np.array(img, dtype=np.float64)
    else:
        image_array = image_array.astype(np.float64)

    # Normalize pixel values to 0-1
    image_array = image_array / 255.0

    f_min = center_freq - spectral_width / 2.0

    total_samples = int(SAMPLE_RATE * duration)
    samples_per_line = total_samples / num_lines

    # Build the modulation envelope for each carrier (time-domain)
    # For each carrier i, the envelope follows the pixel column i
    # with raised-cosine transitions between adjacent pixels
    t = np.arange(total_samples, dtype=np.float64) / SAMPLE_RATE

    signal = np.zeros(total_samples, dtype=np.float64)

    # Build envelope for all carriers
    for i in range(num_carriers):
        freq = f_min + i * delta_f

        # Build amplitude envelope from pixel column
        envelope = np.zeros(total_samples, dtype=np.float64)

        for line in range(num_lines):
            start_sample = int(line * samples_per_line)
            end_sample = int((line + 1) * samples_per_line)
            if end_sample > total_samples:
                end_sample = total_samples

            seg_len = end_sample - start_sample
            if seg_len <= 0:
                continue

            current_val = image_array[line, i]

            # Raised cosine transition zone (first ~10% of segment)
            if line > 0:
                prev_val = image_array[line - 1, i]
            else:
                prev_val = current_val

            transition_len = max(1, int(seg_len * 0.1))
            # Raised cosine: 0.5*(1 - cos(pi*n/N))
            rc = 0.5 * (1.0 - np.cos(np.pi * np.arange(transition_len) / transition_len))

            envelope[start_sample:start_sample + transition_len] = (
                prev_val + (current_val - prev_val) * rc
            )
            envelope[start_sample + transition_len:end_sample] = current_val

        # Generate carrier and modulate
        carrier = np.sin(2.0 * np.pi * freq * t)
        signal += envelope * carrier

    # Normalize to 16-bit range
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 32767.0

    return signal.astype(np.int16)


def apply_brightness_contrast(image_array, brightness, contrast):
    """Apply brightness and contrast adjustments to a grayscale image array."""
    img = Image.fromarray(image_array.astype(np.uint8))

    if brightness != 0:
        factor = 1.0 + brightness / 100.0
        img = ImageEnhance.Brightness(img).enhance(factor)

    if contrast != 0:
        factor = 1.0 + contrast / 100.0
        img = ImageEnhance.Contrast(img).enhance(factor)

    return np.array(img)


class IkePicFallApp:
    def __init__(self, root):
        self.root = root
        self.root.title("IkePicFall — Image to Spectrogram Audio")
        self.root.minsize(800, 600)

        self.source_image = None       # Original PIL Image
        self.processed_array = None    # Grayscale numpy array ready for audio gen
        self.audio_data = None         # Generated audio (int16 numpy array)
        self.inverted = False
        self.flipped = False

        self._build_menu()
        self._build_ui()
        self._update_freq_labels()

    def _build_menu(self):
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open...", command=self._open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Paste", command=self._paste_image, accelerator="Ctrl+V")
        file_menu.add_separator()
        file_menu.add_command(label="Save WAV...", command=self._save_wav, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_command(label="Invert", command=self._toggle_invert)
        view_menu.add_command(label="Flip Vertical", command=self._toggle_flip)
        menubar.add_cascade(label="View", menu=view_menu)

        self.root.config(menu=menubar)

        self.root.bind("<Control-o>", lambda e: self._open_image())
        self.root.bind("<Control-v>", lambda e: self._paste_image())
        self.root.bind("<Control-s>", lambda e: self._save_wav())

    def _build_ui(self):
        # Top row: source image + spectrum
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Source image panel
        src_frame = tk.LabelFrame(top_frame, text="Source Image")
        src_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self.source_canvas = tk.Canvas(src_frame, bg="black", width=250, height=230)
        self.source_canvas.pack(fill=tk.BOTH, expand=True)

        # Spectrum panel
        spec_frame = tk.LabelFrame(top_frame, text="Spectrum (accumulated)")
        spec_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(3, 0))
        self.spectrum_canvas = tk.Canvas(spec_frame, bg="black", width=250, height=230)
        self.spectrum_canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom row: spectrogram preview + controls
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Spectrogram preview panel
        spectro_frame = tk.LabelFrame(bottom_frame, text="Spectrogram Preview")
        spectro_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 3))
        self.spectro_canvas = tk.Canvas(spectro_frame, bg="black", width=250, height=230)
        self.spectro_canvas.pack(fill=tk.BOTH, expand=True)

        # Controls panel
        ctrl_frame = tk.LabelFrame(bottom_frame, text="Controls")
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(3, 0))

        # Slider definitions: (label, from, to, default, resolution, variable_name)
        slider_defs = [
            ("Spectral Width (Hz)", 1000, 2800, 2000, 100, "spectral_width"),
            ("Delta F (Hz)", 14, 20, 16, 1, "delta_f"),
            ("Center Freq (Hz)", 1300, 1700, 1500, 10, "center_freq"),
            ("Duration (s)", 1, 60, 10, 1, "duration"),
            ("Brightness", -100, 100, 0, 1, "brightness"),
            ("Contrast", -100, 100, 0, 1, "contrast"),
        ]

        self.sliders = {}
        for label, from_, to_, default, resolution, var_name in slider_defs:
            frame = tk.Frame(ctrl_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            tk.Label(frame, text=label, anchor="w").pack(fill=tk.X)
            var = tk.IntVar(value=default)
            slider = tk.Scale(
                frame, from_=from_, to=to_, orient=tk.HORIZONTAL,
                variable=var, resolution=resolution, length=200,
                command=lambda val, vn=var_name: self._on_slider_change(vn)
            )
            slider.pack(fill=tk.X)
            self.sliders[var_name] = var

        # Generate button
        btn_frame = tk.Frame(ctrl_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        self.generate_btn = tk.Button(
            btn_frame, text="Generate", command=self._generate,
            bg="#4CAF50", fg="white", font=("", 11, "bold"), height=2
        )
        self.generate_btn.pack(fill=tk.X)

        # Status bar
        status_frame = tk.Frame(self.root, relief=tk.SUNKEN, bd=1)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=5, pady=(0, 5))
        self.status_label = tk.Label(status_frame, text="Ready", anchor="w")
        self.status_label.pack(fill=tk.X, padx=5, pady=2)

    def _get_params(self):
        return {
            "spectral_width": self.sliders["spectral_width"].get(),
            "delta_f": self.sliders["delta_f"].get(),
            "center_freq": self.sliders["center_freq"].get(),
            "duration": self.sliders["duration"].get(),
            "brightness": self.sliders["brightness"].get(),
            "contrast": self.sliders["contrast"].get(),
        }

    def _update_freq_labels(self):
        p = self._get_params()
        f_min = p["center_freq"] - p["spectral_width"] / 2
        f_max = p["center_freq"] + p["spectral_width"] / 2
        num_carriers = int(p["spectral_width"] / p["delta_f"])
        self.status_label.config(
            text=f"f_min: {f_min:.0f} Hz | f_center: {p['center_freq']} Hz | "
                 f"f_max: {f_max:.0f} Hz | Carriers: {num_carriers}"
        )

    def _on_slider_change(self, var_name):
        self._update_freq_labels()
        if self.source_image is not None:
            self._update_previews()

    def _load_image(self, img):
        """Load a PIL Image into the app."""
        self.source_image = img
        self.inverted = False
        self.flipped = False
        self.audio_data = None
        self._update_previews()
        self.status_label.config(text="Image loaded. Adjust parameters and click Generate.")

    def _update_previews(self):
        """Update all preview canvases."""
        if self.source_image is None:
            return

        p = self._get_params()
        num_carriers = int(p["spectral_width"] / p["delta_f"])
        num_lines = min(MAX_LINES, num_carriers)  # Keep aspect reasonable

        # Convert to grayscale and resize
        gray = self.source_image.convert("L")
        resized = gray.resize((num_carriers, num_lines), Image.LANCZOS)
        arr = np.array(resized)

        # Apply brightness/contrast
        arr = apply_brightness_contrast(arr, p["brightness"], p["contrast"])

        if self.inverted:
            arr = 255 - arr

        if self.flipped:
            arr = arr[::-1, :]

        self.processed_array = arr

        # Update source image canvas
        self._draw_image_on_canvas(self.source_canvas, self.source_image.convert("L"))

        # Update spectrogram preview (how it would look)
        preview_img = Image.fromarray(arr)
        self._draw_image_on_canvas(self.spectro_canvas, preview_img)

        # Update spectrum display
        self._draw_spectrum(arr)

    def _draw_image_on_canvas(self, canvas, pil_image):
        """Draw a PIL image centered and scaled on a canvas."""
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = 250, 230

        img = pil_image.copy()
        img.thumbnail((cw, ch), Image.LANCZOS)

        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(img)
        canvas.delete("all")
        canvas.create_image(cw // 2, ch // 2, image=photo, anchor=tk.CENTER)
        # Keep reference to prevent garbage collection
        canvas._photo = photo

    def _draw_spectrum(self, image_array):
        """Draw accumulated spectrum on the spectrum canvas."""
        canvas = self.spectrum_canvas
        canvas.update_idletasks()
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 2 or ch < 2:
            cw, ch = 250, 230

        canvas.delete("all")

        # Accumulated spectrum: sum of all lines (rows)
        spectrum = np.mean(image_array.astype(np.float64), axis=0)
        if spectrum.max() > 0:
            spectrum = spectrum / spectrum.max()

        num_bins = len(spectrum)
        if num_bins < 2:
            return

        # Draw spectrum as line plot
        x_scale = cw / num_bins
        points = []
        for i, val in enumerate(spectrum):
            x = i * x_scale
            y = ch - val * (ch - 10)
            points.append((x, y))

        # Draw filled area
        fill_points = [(0, ch)] + points + [(cw, ch)]
        flat = [coord for pt in fill_points for coord in pt]
        canvas.create_polygon(flat, fill="#1a472a", outline="")

        # Draw line
        line_flat = [coord for pt in points for coord in pt]
        if len(line_flat) >= 4:
            canvas.create_line(line_flat, fill="#4CAF50", width=1)

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Open Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp"),
                ("All files", "*.*"),
            ]
        )
        if path:
            try:
                img = Image.open(path)
                self._load_image(img)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open image:\n{e}")

    def _paste_image(self):
        try:
            img = ImageGrab.grabclipboard()
            if img is None:
                messagebox.showinfo("Paste", "No image found in clipboard.")
                return
            self._load_image(img)
        except Exception as e:
            messagebox.showerror("Error", f"Could not paste image:\n{e}")

    def _toggle_invert(self):
        self.inverted = not self.inverted
        if self.source_image is not None:
            self._update_previews()

    def _toggle_flip(self):
        self.flipped = not self.flipped
        if self.source_image is not None:
            self._update_previews()

    def _generate(self):
        if self.processed_array is None:
            messagebox.showinfo("Generate", "Please open or paste an image first.")
            return

        p = self._get_params()
        self.generate_btn.config(state=tk.DISABLED, text="Generating...")
        self.root.update()

        try:
            self.audio_data = generate_audio(
                self.processed_array,
                spectral_width=p["spectral_width"],
                delta_f=p["delta_f"],
                center_freq=p["center_freq"],
                duration=p["duration"],
            )
            duration = len(self.audio_data) / SAMPLE_RATE
            self.status_label.config(
                text=f"Audio generated: {len(self.audio_data)} samples, "
                     f"{duration:.1f}s. Ready to save."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Audio generation failed:\n{e}")
        finally:
            self.generate_btn.config(state=tk.NORMAL, text="Generate")

    def _save_wav(self):
        if self.audio_data is None:
            messagebox.showinfo("Save", "Generate audio first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save WAV",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            try:
                from scipy.io import wavfile
                wavfile.write(path, SAMPLE_RATE, self.audio_data)
                self.status_label.config(text=f"Saved: {os.path.basename(path)}")
            except ImportError:
                messagebox.showerror(
                    "Error",
                    "scipy is required for WAV export.\n"
                    "Install with: pip install scipy"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Could not save WAV:\n{e}")


def main():
    root = tk.Tk()
    app = IkePicFallApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
