import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

class FFTGui:
    def __init__(self, root):
        self.root = root
        self.root.title("FFT Image Filtering + Glare Detection")

        self.img_color = None
        self.gray = None

        # ---------- CONTROLS ----------
        control = tk.Frame(root)
        control.pack(pady=10)

        tk.Button(control, text="Load Image", command=self.load_image).grid(row=0, column=0, padx=10)

        # Filter type
        self.filter_type = tk.StringVar(value="band")
        tk.Radiobutton(control, text="Low", variable=self.filter_type, value="low", command=self.update_filter).grid(row=0, column=1)
        tk.Radiobutton(control, text="High", variable=self.filter_type, value="high", command=self.update_filter).grid(row=0, column=2)
        tk.Radiobutton(control, text="Band", variable=self.filter_type, value="band", command=self.update_filter).grid(row=0, column=3)

        # Color mode
        self.color_mode = tk.StringVar(value="hsv")
        tk.Radiobutton(control, text="Gray", variable=self.color_mode, value="gray", command=self.update_filter).grid(row=1, column=1)
        tk.Radiobutton(control, text="RGB", variable=self.color_mode, value="rgb", command=self.update_filter).grid(row=1, column=2)
        tk.Radiobutton(control, text="HSV (V)", variable=self.color_mode, value="hsv", command=self.update_filter).grid(row=1, column=3)

        # Sliders
        self.outer_slider = tk.Scale(control, from_=10, to=300, orient=tk.HORIZONTAL, label="Outer Radius", command=self.update_filter)
        self.outer_slider.set(120)
        self.outer_slider.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10)

        self.inner_slider = tk.Scale(control, from_=1, to=200, orient=tk.HORIZONTAL, label="Inner Radius", command=self.update_filter)
        self.inner_slider.set(40)
        self.inner_slider.grid(row=2, column=2, columnspan=2, sticky="ew", padx=10)

        # Glare toggle
        self.glare_enabled = tk.BooleanVar(value=True)
        tk.Checkbutton(control, text="Measure Glare", variable=self.glare_enabled, command=self.update_filter).grid(row=3, column=0)

        # Glare stats
        self.glare_label = tk.Label(control, text="Glare Score: -")
        self.glare_label.grid(row=3, column=1, columnspan=3, sticky="w")

        # ---------- NEW: Glare attenuation slider ----------
        self.atten_slider = tk.Scale(control, from_=10, to=100, orient=tk.HORIZONTAL, label="Glare Attenuation (%)", command=self.update_filter)
        self.atten_slider.set(60)  # default 60%
        self.atten_slider.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10)

        # ---------- DISPLAY ----------
        image_frame = tk.Frame(root)
        image_frame.pack(pady=10)

        self.lbl_original = tk.Label(image_frame)
        self.lbl_fft = tk.Label(image_frame)
        self.lbl_filtered = tk.Label(image_frame)

        # ---------- NEW: Fourth image for glare-removed ----------
        self.lbl_removed = tk.Label(image_frame)

        self.lbl_original.pack(side=tk.LEFT, padx=10)
        self.lbl_fft.pack(side=tk.LEFT, padx=10)
        self.lbl_filtered.pack(side=tk.LEFT, padx=10)
        self.lbl_removed.pack(side=tk.LEFT, padx=10)

    # ---------- LOAD ----------
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not path:
            return

        self.img_color = cv2.imread(path)
        self.gray = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)
        self.update_filter()

    # ---------- MASK ----------
    def create_mask(self, shape):
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2

        mask = np.zeros((rows, cols), np.uint8)
        outer = self.outer_slider.get()
        inner = self.inner_slider.get()

        if self.filter_type.get() == "low":
            cv2.circle(mask, (ccol, crow), outer, 1, -1)
        elif self.filter_type.get() == "high":
            mask[:] = 1
            cv2.circle(mask, (ccol, crow), outer, 0, -1)
        elif self.filter_type.get() == "band":
            cv2.circle(mask, (ccol, crow), outer, 1, -1)
            cv2.circle(mask, (ccol, crow), inner, 0, -1)

        return mask

    # ---------- FFT ----------
    def fft_filter(self, img, mask):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        magnitude = 20 * np.log(np.abs(fshift) + 1)

        fshift_filtered = fshift * mask
        img_back = np.fft.ifft2(np.fft.ifftshift(fshift_filtered))
        img_back = np.abs(img_back)

        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return img_back, magnitude

    # ---------- GLARE MEASUREMENT ----------
    def measure_glare(self, glare_response):
        g = glare_response.astype(np.float32)
        g /= g.max() + 1e-6

        # Threshold
        glare_mask = (g > 0.8).astype(np.uint8) * 255

        # Cleanup
        glare_mask = cv2.morphologyEx(
            glare_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
        )

        # Metrics
        area_pct = 100.0 * np.sum(glare_mask > 0) / glare_mask.size
        intensity = g[glare_mask > 0].mean() if area_pct > 0 else 0.0
        score = np.sum(g ** 2)

        return glare_mask, score, area_pct, intensity

    # ---------- NEW: HSV-based glare attenuation ----------
    def attenuate_glare_hsv(self, img_bgr, glare_mask):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        factor = self.atten_slider.get() / 100.0

        v = v.astype(np.float32)
        v[glare_mask > 0] *= factor
        v = np.clip(v, 0, 255).astype(np.uint8)

        hsv_out = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

    # ---------- NEW: Draw ROI rectangle around glare ----------
    def draw_roi(self, img, glare_mask):
        ys, xs = np.where(glare_mask > 0)
        if len(xs) == 0:
            return img
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        img_box = img.copy()
        cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return img_box

    # ---------- UPDATE ----------
    def update_filter(self, event=None):
        if self.img_color is None:
            return

        hsv = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        mask = self.create_mask(v.shape)
        v_filtered, fft_img = self.fft_filter(v, mask)

        output = v_filtered.copy()

        # ----- Glare detection -----
        if self.glare_enabled.get():
            glare_mask, score, area, intensity = self.measure_glare(v_filtered)

            overlay = self.img_color.copy()
            overlay[glare_mask > 0] = [0, 0, 255]  # red overlay
            output = cv2.addWeighted(self.img_color, 0.7, overlay, 0.3, 0)
            overlay = self.draw_roi(output, glare_mask)

            # ----- Glare removal (HSV attenuation) -----
            glare_removed = self.attenuate_glare_hsv(self.img_color, glare_mask)
            glare_removed = self.draw_roi(glare_removed, glare_mask)

            self.glare_label.config(
                text=f"Glare Score: {score:.2f} | Area: {area:.2f}% | Intensity: {intensity:.2f}"
            )
        else:
            self.glare_label.config(text="Glare Score: -")
            glare_removed = self.img_color.copy()

        # ---------- DISPLAY ----------
        self.show_image(self.img_color, self.lbl_original)
        self.show_image(fft_img, self.lbl_fft, gray=True)
        self.show_image(overlay, self.lbl_filtered)
        self.show_image(glare_removed, self.lbl_removed)

    # ---------- DISPLAY ----------
    def show_image(self, img, label, gray=False):
        if not gray:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((256, 256))
        img_tk = ImageTk.PhotoImage(img_pil)
        label.configure(image=img_tk)
        label.image = img_tk


if __name__ == "__main__":
    root = tk.Tk()
    FFTGui(root)
    root.mainloop()
