import io
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageTk

# Try to import torch and Hi-SAM modules
try:
    import torch

    torch_available = True

    # Hi-SAM imports
    sys.path.append(os.path.join(os.path.dirname(__file__), "Hi-SAM"))
    from hi_sam.modeling.auto_mask_generator import AutoMaskGenerator
    from hi_sam.modeling.build import model_registry

    hisam_available = True

except ImportError as e:
    torch_available = False
    hisam_available = False
    print(f"Warning: PyTorch or Hi-SAM not available: {e}")
    print("The application will run without text line detection capability.")


class DocumentSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Image Segmentation App")
        self.root.geometry("1200x800")

        self.current_file_path = None
        self.current_page = 0
        self.total_pages = 0
        self.original_image = None
        self.processed_image = None
        self.displayed_image = None
        self.segmentation_masks = []  # セグメンテーションマスクのリスト
        self.original_segmentation_masks = []  # 元のセグメンテーションマスク
        self.photo = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.save_directory = None  # Save directory

        # Load Hi-SAM model
        self.hi_sam_model = None
        self.auto_mask_generator = None
        self.model_type = "vit_s"  # デフォルトは軽量モデル
        self.load_hi_sam_model()

        # GUI components creation
        self.create_widgets()

        # Variables for mask operations
        self.selected_mask_index = -1
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0

    def load_hi_sam_model(self):
        """Load Hi-SAM model"""
        if not torch_available or not hisam_available:
            print(
                "PyTorch or Hi-SAM not available. Text line detection will be disabled."
            )
            self.hi_sam_model = None
            self.auto_mask_generator = None
            return

        try:
            # Hi-SAMモデルのパラメータ設定（軽量版）
            class Args:
                def __init__(self):
                    # 利用可能なモデル:
                    # "vit_s" - Efficient Hi-SAM Small (最軽量、高速)
                    # "vit_b" - Hi-SAM Base (バランス型)
                    # "vit_l" - Hi-SAM Large (高精度、重い)
                    self.model_type = "vit_s"  # 軽量なEfficient Hi-SAM Smallを使用

                    # 学習済みチェックポイントを指定
                    checkpoint_dir = os.path.join(
                        os.path.dirname(__file__), "Hi-SAM", "pretrained_checkpoint"
                    )

                    # モデルタイプに応じてチェックポイントを選択
                    if self.model_type == "vit_s":
                        self.checkpoint = os.path.join(
                            checkpoint_dir, "efficient_hi_sam_s.pth"
                        )
                    elif self.model_type == "vit_b":
                        self.checkpoint = os.path.join(
                            checkpoint_dir, "sam_tss_b_textseg.pth"
                        )
                    elif self.model_type == "vit_l":
                        self.checkpoint = os.path.join(checkpoint_dir, "hi_sam_l.pth")
                    else:
                        self.checkpoint = None

                    self.hier_det = True
                    self.attn_layers = 1
                    self.prompt_len = 12
                    self.device = "cuda" if torch.cuda.is_available() else "cpu"

            args = Args()

            # チェックポイントファイルの存在確認
            if not os.path.exists(args.checkpoint):
                print(f"Warning: Checkpoint file not found: {args.checkpoint}")
                print("Using model without pre-trained weights.")
                args.checkpoint = None
            else:
                print(f"Loading checkpoint: {args.checkpoint}")

            # モデルを構築
            print("Building Hi-SAM model...")
            self.hi_sam_model = model_registry[args.model_type](args)
            self.hi_sam_model = self.hi_sam_model.to(args.device)
            self.hi_sam_model.eval()

            # AutoMaskGeneratorを初期化（モデルタイプに応じて設定）
            efficient_hisam = args.model_type in [
                "vit_s",
                "vit_t",
            ]  # Efficient Hi-SAMかどうか
            self.auto_mask_generator = AutoMaskGenerator(
                self.hi_sam_model, efficient_hisam=efficient_hisam
            )

            # モデルタイプを保存（後で使用）
            self.model_type = args.model_type

            if args.checkpoint:
                print(
                    f"Hi-SAM model loaded successfully on {args.device} with pre-trained weights"
                )
            else:
                print(
                    f"Hi-SAM model loaded successfully on {args.device} without pre-trained weights"
                )
                print(
                    "Note: For better results, please download and load appropriate checkpoints."
                )

        except Exception as e:
            import traceback

            print(f"Error loading Hi-SAM model: {e}")
            print(traceback.format_exc())
            messagebox.showwarning(
                "Warning",
                f"Failed to load Hi-SAM model: {e}\nThe application will run without text line detection capability.",
            )
            self.hi_sam_model = None
            self.auto_mask_generator = None

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel (controls)
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # Right panel (image display)
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # File selection
        file_frame = ttk.LabelFrame(left_panel, text="File Selection")
        file_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(file_frame, text="Open File", command=self.open_file).pack(pady=5)

        # Page navigation
        page_frame = ttk.LabelFrame(left_panel, text="Page Navigation")
        page_frame.pack(fill=tk.X, pady=(0, 10))

        page_control_frame = ttk.Frame(page_frame)
        page_control_frame.pack(pady=5)

        ttk.Button(page_control_frame, text="Previous", command=self.prev_page).pack(
            side=tk.LEFT, padx=2
        )
        self.page_label = ttk.Label(page_control_frame, text="0/0")
        self.page_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(page_control_frame, text="Next", command=self.next_page).pack(
            side=tk.LEFT, padx=2
        )

        # Display settings
        display_frame = ttk.LabelFrame(left_panel, text="Display Settings")
        display_frame.pack(fill=tk.X, pady=(0, 10))

        # Scale
        scale_frame = ttk.Frame(display_frame)
        scale_frame.pack(fill=tk.X, pady=2)
        ttk.Label(scale_frame, text="Scale:").pack(side=tk.LEFT)
        self.scale_var = tk.DoubleVar(value=1.0)
        self.scale_slider = ttk.Scale(
            scale_frame,
            from_=0.1,
            to=3.0,
            variable=self.scale_var,
            command=self.update_display,
        )
        self.scale_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # X position
        x_frame = ttk.Frame(display_frame)
        x_frame.pack(fill=tk.X, pady=2)
        ttk.Label(x_frame, text="X Position:").pack(side=tk.LEFT)
        self.x_var = tk.DoubleVar(value=0)
        self.x_slider = ttk.Scale(
            x_frame,
            from_=-500,
            to=500,
            variable=self.x_var,
            command=self.update_display,
        )
        self.x_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # Y position
        y_frame = ttk.Frame(display_frame)
        y_frame.pack(fill=tk.X, pady=2)
        ttk.Label(y_frame, text="Y Position:").pack(side=tk.LEFT)
        self.y_var = tk.DoubleVar(value=0)
        self.y_slider = ttk.Scale(
            y_frame,
            from_=-500,
            to=500,
            variable=self.y_var,
            command=self.update_display,
        )
        self.y_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # Preprocessing settings
        preprocess_frame = ttk.LabelFrame(left_panel, text="Preprocessing")
        preprocess_frame.pack(fill=tk.X, pady=(0, 10))

        # Binarization
        binary_frame = ttk.Frame(preprocess_frame)
        binary_frame.pack(fill=tk.X, pady=2)
        self.binary_enabled = tk.BooleanVar()
        ttk.Checkbutton(
            binary_frame,
            text="Binarization",
            variable=self.binary_enabled,
            command=self.apply_preprocessing,
        ).pack(side=tk.LEFT)

        binary_threshold_frame = ttk.Frame(preprocess_frame)
        binary_threshold_frame.pack(fill=tk.X, pady=2)
        ttk.Label(binary_threshold_frame, text="Threshold:").pack(side=tk.LEFT)
        self.binary_threshold = tk.DoubleVar(value=127)
        threshold_scale = ttk.Scale(
            binary_threshold_frame,
            from_=0,
            to=255,
            variable=self.binary_threshold,
            command=self.apply_preprocessing,
        )
        threshold_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # Rotation
        rotation_frame = ttk.Frame(preprocess_frame)
        rotation_frame.pack(fill=tk.X, pady=2)
        ttk.Label(rotation_frame, text="Rotation Angle:").pack(side=tk.LEFT)
        self.rotation_angle = tk.DoubleVar(value=0)
        rotation_scale = ttk.Scale(
            rotation_frame,
            from_=-180,
            to=180,
            variable=self.rotation_angle,
            command=self.apply_preprocessing,
        )
        rotation_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # Distortion correction (simple processing)
        distortion_frame = ttk.Frame(preprocess_frame)
        distortion_frame.pack(fill=tk.X, pady=2)
        self.distortion_enabled = tk.BooleanVar()
        ttk.Checkbutton(
            distortion_frame,
            text="Distortion Correction",
            variable=self.distortion_enabled,
            command=self.apply_preprocessing,
        ).pack(side=tk.LEFT)

        # Recognition
        recognition_frame = ttk.LabelFrame(left_panel, text="Recognition")
        recognition_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(
            recognition_frame, text="Run Recognition", command=self.run_recognition
        ).pack(pady=5)

        # Segmentation mask adjustment
        mask_frame = ttk.LabelFrame(left_panel, text="Segmentation Mask Adjustment")
        mask_frame.pack(fill=tk.X, pady=(0, 10))

        # Mask operations
        mask_operations_frame = ttk.Frame(mask_frame)
        mask_operations_frame.pack(fill=tk.X, pady=5)
        ttk.Button(
            mask_operations_frame,
            text="Delete Selected",
            command=self.delete_selected_mask,
        ).pack(side=tk.LEFT, padx=(0, 5))

        # Image saving
        save_frame = ttk.LabelFrame(left_panel, text="Image Saving")
        save_frame.pack(fill=tk.X, pady=(0, 10))

        # Save directory selection
        dir_frame = ttk.Frame(save_frame)
        dir_frame.pack(fill=tk.X, pady=2)
        ttk.Button(
            dir_frame, text="Select Directory", command=self.select_save_directory
        ).pack(side=tk.LEFT, padx=(0, 5))
        self.save_dir_label = ttk.Label(
            dir_frame, text="Not selected", foreground="gray"
        )
        self.save_dir_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Save buttons
        save_buttons_frame = ttk.Frame(save_frame)
        save_buttons_frame.pack(fill=tk.X, pady=2)
        ttk.Button(
            save_buttons_frame, text="Save All", command=self.save_all_crops
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            save_buttons_frame, text="Save Selected", command=self.save_selected_crop
        ).pack(side=tk.LEFT)

        # Image display canvas
        canvas_frame = ttk.Frame(right_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas with scrollbars
        self.canvas = tk.Canvas(canvas_frame, bg="white")
        v_scrollbar = ttk.Scrollbar(
            canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview
        )
        h_scrollbar = ttk.Scrollbar(
            canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview
        )
        self.canvas.configure(
            yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set
        )

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Mouse event bindings
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)

    def open_file(self):
        """Open file"""
        file_path = filedialog.askopenfilename(
            title="Select File",
            filetypes=[
                ("All supported files", "*.pdf *.png *.jpg *.jpeg"),
                ("PDF", "*.pdf"),
                ("Images", "*.png *.jpg *.jpeg"),
                ("All files", "*.*"),
            ],
        )

        if file_path:
            self.current_file_path = file_path
            self.current_page = 0
            self.load_document()

    def load_document(self):
        """Load document"""
        if not self.current_file_path:
            return

        try:
            file_ext = os.path.splitext(self.current_file_path)[1].lower()

            if file_ext == ".pdf":
                # Load PDF
                doc = fitz.open(self.current_file_path)
                self.total_pages = len(doc)
                page = doc[self.current_page]

                # Convert PDF page to image
                mat = fitz.Matrix(2.0, 2.0)  # Convert with high resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("ppm")
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                self.original_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                doc.close()

            else:
                # Load image
                self.original_image = cv2.imread(self.current_file_path)
                self.total_pages = 1

            self.update_page_label()
            self.apply_preprocessing()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def prev_page(self):
        """Previous page"""
        if self.current_page > 0:
            self.current_page -= 1
            self.load_document()

    def next_page(self):
        """Next page"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.load_document()

    def update_page_label(self):
        """Update page label"""
        current = self.current_page + 1
        total = self.total_pages
        self.page_label.config(text=f"{current}/{total}")

    def apply_preprocessing(self, *args):
        """Apply preprocessing"""
        if self.original_image is None:
            return

        image = self.original_image.copy()

        # Rotation
        angle = self.rotation_angle.get()
        if angle != 0:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image,
                rotation_matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255),
            )

        # Distortion correction (simple trapezoid correction)
        if self.distortion_enabled.get():
            # Simple distortion correction (more advanced processing needed in practice)
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)

        # Binarization
        if self.binary_enabled.get():
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            threshold = int(self.binary_threshold.get())
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

        self.processed_image = image
        self.update_display()

    def update_display(self, *args):
        """Update display"""
        if self.processed_image is None:
            return

        # Get scale and position
        scale = self.scale_var.get()
        offset_x = int(self.x_var.get())
        offset_y = int(self.y_var.get())

        # Resize image
        height, width = self.processed_image.shape[:2]
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_image = cv2.resize(self.processed_image, (new_width, new_height))

        # Draw segmentation masks
        display_image = resized_image.copy()

        for i, mask in enumerate(self.segmentation_masks):
            # マスクをスケールに合わせてリサイズ
            mask_scaled = cv2.resize(
                mask.astype(np.uint8),
                (new_width, new_height),
                interpolation=cv2.INTER_NEAREST,
            )

            # Selected mask highlighting
            is_selected = i == self.selected_mask_index

            # マスクを可視化するための色
            if is_selected:
                color = [255, 0, 0]  # 選択されたマスクは赤
                alpha = 0.6
            else:
                color = [0, 255, 0]  # その他のマスクは緑
                alpha = 0.4

            # マスクのオーバーレイを作成
            mask_overlay = np.zeros_like(display_image)
            mask_overlay[mask_scaled > 0] = color

            # マスクをオーバーレイで合成
            display_image = cv2.addWeighted(display_image, 1, mask_overlay, alpha, 0)

            # マスクの輪郭を描画
            contours, _ = cv2.findContours(
                mask_scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour_color = (0, 0, 255) if is_selected else (0, 255, 0)
            cv2.drawContours(display_image, contours, -1, contour_color, 2)

        # Convert image to Tkinter format
        self.displayed_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(self.displayed_image)
        self.photo = ImageTk.PhotoImage(img_pil)

        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def run_recognition(self):
        """Run text line segmentation with Hi-SAM model"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        if self.hi_sam_model is None or self.auto_mask_generator is None:
            messagebox.showerror("Error", "Hi-SAM model not loaded")
            return

        try:
            # CUDAメモリクリーンアップ（処理前）
            if torch_available and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(
                    f"CUDA memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()} bytes"
                )

            # 画像をRGB形式に変換
            image = self.processed_image.copy()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Hi-SAMで画像を設定
            self.auto_mask_generator.set_image(image_rgb)

            # テキストライン検出のパラメータ（ライン検出に最適化）
            if self.model_type == "vit_s":
                # Efficient Hi-SAM Small用 - ライン検出設定
                fg_points_num = 120  # ライン検出のため増加
                batch_points_num = 20  # メモリ効率維持
                score_thresh = 0.5  # CTW1500に適した閾値
                nms_thresh = 0.4  # NMS閾値
            elif self.model_type == "vit_b":
                # Hi-SAM Base用のライン検出設定
                fg_points_num = 200
                batch_points_num = 40
                score_thresh = 0.6
                nms_thresh = 0.5
            else:  # vit_l
                # Hi-SAM Large用のライン検出設定
                fg_points_num = 300
                batch_points_num = 60
                score_thresh = 0.7
                nms_thresh = 0.6

            zero_shot = False  # 学習済みモデルを使用
            dataset = "ctw1500"  # CTW1500データセット（直接ライン検出を行う）

            # テキストライン検出を実行
            print("Running Hi-SAM text line segmentation...")
            masks, scores = self.auto_mask_generator.predict_text_detection(
                from_low_res=True,  # 低解像度から開始してより大きなマスクを生成
                fg_points_num=fg_points_num,
                batch_points_num=batch_points_num,
                score_thresh=score_thresh,
                nms_thresh=nms_thresh,
                zero_shot=zero_shot,
                dataset=dataset,
            )

            # CUDAメモリクリーンアップ（処理後）
            if torch_available and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if masks is not None and len(masks) > 0:
                print(f"Found {len(masks)} text line masks (directly from Hi-SAM)")

                # セグメンテーションマスクを保存（直接ライン単位のマスクを使用）
                self.segmentation_masks = []
                self.original_segmentation_masks = []

                for i, mask in enumerate(masks):
                    # マスクが3次元の場合、2次元に変換
                    if len(mask.shape) == 3:
                        mask = mask[0]  # 最初のチャンネルを使用

                    # マスクをブール型に変換
                    mask_bool = mask > 0.5

                    self.segmentation_masks.append(mask_bool)
                    self.original_segmentation_masks.append(mask_bool.copy())

                # 表示を更新
                self.update_display()
                messagebox.showinfo(
                    "Complete", f"Detected {len(masks)} text line regions"
                )

            else:
                messagebox.showinfo("Information", "No text line regions detected")

        except Exception as e:
            import traceback

            print(f"Error in Hi-SAM recognition: {e}")
            print(traceback.format_exc())

            # CUDAメモリクリーンアップ
            if torch_available and torch.cuda.is_available():
                torch.cuda.empty_cache()

            messagebox.showerror("Error", f"Error occurred during recognition: {e}")

    def delete_selected_mask(self):
        """Delete selected segmentation mask"""
        if self.selected_mask_index < 0:
            messagebox.showwarning(
                "Warning", "Please select a segmentation mask to delete"
            )
            return

        if len(self.segmentation_masks) == 0:
            messagebox.showwarning("Warning", "No segmentation masks to delete")
            return

        # Confirm deletion
        result = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete segmentation mask {self.selected_mask_index + 1}?",
        )

        if result:
            # Remove the selected mask
            del self.segmentation_masks[self.selected_mask_index]

            # Also remove from original masks if it exists
            if self.selected_mask_index < len(self.original_segmentation_masks):
                del self.original_segmentation_masks[self.selected_mask_index]

            # Adjust selection index if necessary
            if self.selected_mask_index >= len(self.segmentation_masks):
                self.selected_mask_index = len(self.segmentation_masks) - 1
            if len(self.segmentation_masks) == 0:
                self.selected_mask_index = -1

            # Update display
            self.update_display()

            messagebox.showinfo("Success", "Segmentation mask deleted successfully")

    def select_save_directory(self):
        """Select save directory"""
        directory = filedialog.askdirectory(title="Select Save Directory")
        if directory:
            self.save_directory = directory
            # Display the end of the path if it's too long
            display_path = directory
            if len(display_path) > 30:
                display_path = "..." + display_path[-27:]
            self.save_dir_label.config(text=display_path, foreground="black")

    def save_all_crops(self):
        """Save all images within segmentation masks"""
        if not self.segmentation_masks:
            messagebox.showwarning("Warning", "No segmentation masks to save")
            return

        if not self.save_directory:
            messagebox.showwarning("Warning", "Please select a save directory")
            return

        if self.processed_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        try:
            saved_count = 0
            for i, mask in enumerate(self.segmentation_masks):
                success = self._save_masked_image(mask, i)
                if success:
                    saved_count += 1

            messagebox.showinfo("Complete", f"Saved {saved_count} images")

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred during saving: {e}")

    def save_selected_crop(self):
        """Save image within selected segmentation mask"""
        if self.selected_mask_index < 0:
            messagebox.showwarning("Warning", "Please select a segmentation mask")
            return

        if not self.save_directory:
            messagebox.showwarning("Warning", "Please select a save directory")
            return

        if self.processed_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        try:
            mask = self.segmentation_masks[self.selected_mask_index]
            success = self._save_masked_image(mask, self.selected_mask_index)

            if success:
                messagebox.showinfo("Complete", "Image saved")

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred during saving: {e}")

    def _save_masked_image(self, mask, index):
        """Extract and save the image within the segmentation mask"""
        try:
            # マスクをnumpy配列に変換（必要に応じて）
            if torch_available and torch.is_tensor(mask):
                mask = mask.cpu().numpy()

            mask = mask.astype(np.uint8)

            # マスクの境界ボックスを取得
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if len(contours) == 0:
                print(f"No contours found in mask {index}")
                return False

            # 最大の輪郭を選択
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # 画像サイズ内に制限
            height, width = self.processed_image.shape[:2]
            x = max(0, min(x, width))
            y = max(0, min(y, height))
            w = min(w, width - x)
            h = min(h, height - y)

            if w <= 0 or h <= 0:
                print(f"Invalid mask dimensions {index}: ({x}, {y}, {w}, {h})")
                return False

            # マスクの該当部分を切り出し
            mask_crop = mask[y : y + h, x : x + w]
            image_crop = self.processed_image[y : y + h, x : x + w]

            # マスクを適用して背景を白にする
            masked_image = image_crop.copy()
            masked_image[mask_crop == 0] = [255, 255, 255]  # 白背景

            if masked_image.size == 0:
                print(f"Empty masked image {index}")
                return False

            # ファイル名を生成
            base_name = "segmented_text"
            if self.current_file_path:
                file_name = os.path.splitext(os.path.basename(self.current_file_path))[
                    0
                ]
                base_name = f"{file_name}_seg"

            # ページ番号を追加（PDFの場合）
            if self.total_pages > 1:
                base_name += f"_page{self.current_page + 1:03d}"

            # インデックスを追加
            output_filename = f"{base_name}_{index:03d}.png"
            output_path = os.path.join(self.save_directory, output_filename)

            # 画像を保存
            cv2.imwrite(output_path, masked_image)
            print(f"Save complete: {output_path}")
            return True

        except Exception as e:
            print(f"Masked image save error {index}: {e}")
            return False

    def on_canvas_click(self, event):
        """Canvas click handling for segmentation masks"""
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Coordinate transformation considering display scale
        scale = self.scale_var.get()
        offset_x = int(self.x_var.get())
        offset_y = int(self.y_var.get())

        img_x = int((x - offset_x) / scale)
        img_y = int((y - offset_y) / scale)

        # セグメンテーションマスクの選択
        self.selected_mask_index = -1

        # 画像の範囲内かチェック
        if self.processed_image is not None:
            height, width = self.processed_image.shape[:2]
            if 0 <= img_x < width and 0 <= img_y < height:
                # 各マスクをチェックして、クリック位置がマスク内かどうか確認
                for i, mask in enumerate(self.segmentation_masks):
                    if torch_available and torch.is_tensor(mask):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = mask

                    if mask_np[img_y, img_x] > 0:
                        self.selected_mask_index = i
                        break

        self.update_display()

    def on_canvas_drag(self, event):
        """Canvas drag handling - currently disabled for masks"""
        # セグメンテーションマスクは通常はドラッグで編集しないため、
        # このメソッドは基本的に何もしない
        pass

    def on_canvas_release(self, event):
        """Canvas release handling"""
        self.dragging = False

    def on_canvas_motion(self, event):
        """Change cursor on mouse movement"""
        if self.dragging:
            return

        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Coordinate transformation considering display scale
        scale = self.scale_var.get()
        offset_x = int(self.x_var.get())
        offset_y = int(self.y_var.get())

        img_x = int((x - offset_x) / scale)
        img_y = int((y - offset_y) / scale)

        # セグメンテーションマスクの場合はカーソルを変更
        cursor = "arrow"
        if self.processed_image is not None:
            height, width = self.processed_image.shape[:2]
            if 0 <= img_x < width and 0 <= img_y < height:
                # マスクの上にカーソルがある場合は手のカーソルを表示
                for mask in self.segmentation_masks:
                    if torch_available and torch.is_tensor(mask):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = mask

                    if mask_np[img_y, img_x] > 0:
                        cursor = "hand2"
                        break

        self.canvas.config(cursor=cursor)


def main():
    # Check for required module imports
    try:
        import io

        globals()["io"] = io
    except ImportError:
        pass

    root = tk.Tk()
    DocumentSegmentationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
