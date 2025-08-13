import io
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import fitz  # PyMuPDF
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageTk


class DocumentSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Document Image Segmentation App")
        self.root.geometry("1200x800")

        # Application state
        self.current_file_path = None
        self.current_page = 0
        self.total_pages = 0
        self.original_image = None
        self.processed_image = None
        self.displayed_image = None
        self.bounding_boxes = []
        self.original_bounding_boxes = []  # Original bounding boxes
        self.photo = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.save_directory = None  # Save directory

        # Load ONNX model
        try:
            self.onnx_session = ort.InferenceSession("model.onnx")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load ONNX model: {e}")
            self.onnx_session = None

        # GUI components creation
        self.create_widgets()

        # Variables for bounding box operations
        self.selected_box_index = -1
        self.dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_edge = None  # 'left', 'right', 'top', 'bottom', 'move'

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

        # Bounding box adjustment
        bbox_frame = ttk.LabelFrame(left_panel, text="Bounding Box Adjustment")
        bbox_frame.pack(fill=tk.X, pady=(0, 10))

        # Width adjustment
        bbox_width_frame = ttk.Frame(bbox_frame)
        bbox_width_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bbox_width_frame, text="Width:").pack(side=tk.LEFT)
        self.bbox_width_scale = tk.DoubleVar(value=1.0)
        bbox_width_slider = ttk.Scale(
            bbox_width_frame,
            from_=0.5,
            to=2.0,
            variable=self.bbox_width_scale,
            command=self.update_bbox_scale,
        )
        bbox_width_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        # Height adjustment
        bbox_height_frame = ttk.Frame(bbox_frame)
        bbox_height_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bbox_height_frame, text="Height:").pack(side=tk.LEFT)
        self.bbox_height_scale = tk.DoubleVar(value=1.0)
        bbox_height_slider = ttk.Scale(
            bbox_height_frame,
            from_=0.5,
            to=2.0,
            variable=self.bbox_height_scale,
            command=self.update_bbox_scale,
        )
        bbox_height_slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

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

        # Draw bounding boxes
        display_image = resized_image.copy()
        for i, bbox in enumerate(self.bounding_boxes):
            x1, y1, x2, y2 = bbox
            # Adjust coordinates according to scale
            x1_scaled = int(x1 * scale)
            y1_scaled = int(y1 * scale)
            x2_scaled = int(x2 * scale)
            y2_scaled = int(y2 * scale)

            # Draw bounding box
            is_selected = i == self.selected_box_index
            color = (0, 255, 0) if not is_selected else (0, 0, 255)
            cv2.rectangle(
                display_image, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2
            )

            # Draw corner manipulation points
            if is_selected:
                point_size = 5
                point_color = (255, 0, 0)
                cv2.circle(
                    display_image, (x1_scaled, y1_scaled), point_size, point_color, -1
                )
                cv2.circle(
                    display_image, (x2_scaled, y1_scaled), point_size, point_color, -1
                )
                cv2.circle(
                    display_image, (x1_scaled, y2_scaled), point_size, point_color, -1
                )
                cv2.circle(
                    display_image, (x2_scaled, y2_scaled), point_size, point_color, -1
                )

        # Convert image to Tkinter format
        self.displayed_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(self.displayed_image)
        self.photo = ImageTk.PhotoImage(img_pil)

        # Display on canvas
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.photo)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def run_recognition(self):
        """Run recognition with ONNX model"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        if self.onnx_session is None:
            messagebox.showerror("Error", "ONNX model not loaded")
            return

        try:
            # Preprocess image (adjust according to model requirements)
            image = self.processed_image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Get model input size dynamically
            input_shape = self.onnx_session.get_inputs()[0].shape
            # Input shape is 3-dimensional [channels, height, width]
            if len(input_shape) == 3:
                if input_shape[0] == 3:  # CHW format
                    input_height, input_width = input_shape[1], input_shape[2]
                else:  # HWC format
                    input_height, input_width = input_shape[0], input_shape[1]
            elif len(input_shape) == 4:
                # 4-dimensional case (with batch dimension)
                if input_shape[1] == 3:  # NCHW format
                    input_height, input_width = input_shape[2], input_shape[3]
                else:  # NHWC format
                    input_height, input_width = input_shape[1], input_shape[2]
            else:
                # Use default values
                input_height, input_width = 640, 640

            height, width = image.shape[:2]
            scale_x = input_width / width
            scale_y = input_height / height

            resized_image = cv2.resize(image, (input_width, input_height))

            # Normalize and add batch dimension
            input_data = resized_image.astype(np.float32)
            input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW

            # Run inference
            input_name = self.onnx_session.get_inputs()[0].name

            output_names = [output.name for output in self.onnx_session.get_outputs()]
            outputs = self.onnx_session.run(output_names, {input_name: input_data})

            print("Inference results:", outputs)
            # Assume first element of output is bounding boxes
            predictions = outputs[0]

            # Extract bounding boxes (format depends on model)
            self.bounding_boxes = []
            self.original_bounding_boxes = []  # Save originals

            if predictions.shape[0] == 0:
                messagebox.showinfo("Information", "No recognition results")
                return

            # Temporary processing: adjust according to output format
            if len(predictions.shape) >= 2:
                for detection in predictions:  # Get first batch
                    print("Detection result:", detection)
                    print("Detection result shape:", detection.shape)
                    if detection.shape[0] >= 4:
                        # Convert coordinates to original image size
                        x1 = int(detection[0] / scale_x)
                        y1 = int(detection[1] / scale_y)
                        x2 = int(detection[2] / scale_x)
                        y2 = int(detection[3] / scale_y)

                        # Confidence check (if confidence is included)
                        confidence = detection[4] if len(detection) > 4 else 1.0
                        if confidence > 0.5:  # Threshold
                            bbox = [x1, y1, x2, y2]
                            self.bounding_boxes.append(bbox)
                            self.original_bounding_boxes.append(
                                bbox.copy()
                            )  # Save original

            # Reflect results in display
            self.update_display()
            detection_count = len(self.bounding_boxes)
            messagebox.showinfo("Complete", f"Detected {detection_count} text regions")

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred during recognition: {e}")

    def update_bbox_scale(self, *args):
        """Update bounding box scale"""
        if not self.original_bounding_boxes:
            return

        width_scale = self.bbox_width_scale.get()
        height_scale = self.bbox_height_scale.get()

        # Adjust each bounding box from original size
        for i, original_bbox in enumerate(self.original_bounding_boxes):
            x1, y1, x2, y2 = original_bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            original_width = x2 - x1
            original_height = y2 - y1

            # Calculate new size
            new_width = original_width * width_scale
            new_height = original_height * height_scale

            self.bounding_boxes[i] = [
                int(center_x - new_width / 2),
                int(center_y - new_height / 2),
                int(center_x + new_width / 2),
                int(center_y + new_height / 2),
            ]

        self.update_display()

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
        """Save all images within bounding boxes"""
        if not self.bounding_boxes:
            messagebox.showwarning("Warning", "No bounding boxes to save")
            return

        if not self.save_directory:
            messagebox.showwarning("Warning", "Please select a save directory")
            return

        if self.processed_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        try:
            saved_count = 0
            for i, bbox in enumerate(self.bounding_boxes):
                success = self._save_crop_image(bbox, i)
                if success:
                    saved_count += 1

            messagebox.showinfo("Complete", f"Saved {saved_count} images")

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred during saving: {e}")

    def save_selected_crop(self):
        """Save image within selected bounding box"""
        if self.selected_box_index < 0:
            messagebox.showwarning("Warning", "Please select a bounding box")
            return

        if not self.save_directory:
            messagebox.showwarning("Warning", "Please select a save directory")
            return

        if self.processed_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        try:
            bbox = self.bounding_boxes[self.selected_box_index]
            success = self._save_crop_image(bbox, self.selected_box_index)

            if success:
                messagebox.showinfo("Complete", "Image saved")

        except Exception as e:
            messagebox.showerror("Error", f"Error occurred during saving: {e}")

    def _save_crop_image(self, bbox, index):
        """Crop and save the image within the bounding box"""
        try:
            x1, y1, x2, y2 = bbox

            # Limit coordinates within image size
            height, width = self.processed_image.shape[:2]
            x1 = max(0, min(x1, width))
            y1 = max(0, min(y1, height))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            # Check coordinate order
            if x1 >= x2 or y1 >= y2:
                print(f"Invalid bounding box {index}: ({x1}, {y1}, {x2}, {y2})")
                return False

            # Crop image
            cropped_image = self.processed_image[y1:y2, x1:x2]

            if cropped_image.size == 0:
                print(f"Empty cropped image {index}")
                return False

            # Generate filename
            base_name = "cropped_text"
            if self.current_file_path:
                file_name = os.path.splitext(os.path.basename(self.current_file_path))[
                    0
                ]
                base_name = f"{file_name}_crop"

            # Add page number (for PDF)
            if self.total_pages > 1:
                base_name += f"_page{self.current_page + 1}"

            # Add index
            output_filename = f"{base_name}_{index:03d}.png"
            output_path = os.path.join(self.save_directory, output_filename)

            # Save image
            cv2.imwrite(output_path, cropped_image)
            print(f"Save complete: {output_path}")
            return True

        except Exception as e:
            print(f"Crop save error {index}: {e}")
            return False

    def on_canvas_click(self, event):
        """Canvas click handling"""
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Coordinate transformation considering display scale
        scale = self.scale_var.get()
        offset_x = int(self.x_var.get())
        offset_y = int(self.y_var.get())

        img_x = int((x - offset_x) / scale)
        img_y = int((y - offset_y) / scale)

        # Bounding box selection
        self.selected_box_index = -1
        for i, bbox in enumerate(self.bounding_boxes):
            x1, y1, x2, y2 = bbox
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                self.selected_box_index = i
                break

        if self.selected_box_index >= 0:
            self.dragging = True
            self.drag_start_x = img_x
            self.drag_start_y = img_y

            # Determine which edge to drag
            bbox = self.bounding_boxes[self.selected_box_index]
            x1, y1, x2, y2 = bbox

            edge_threshold = 10
            if abs(img_x - x1) < edge_threshold:
                self.drag_edge = "left"
            elif abs(img_x - x2) < edge_threshold:
                self.drag_edge = "right"
            elif abs(img_y - y1) < edge_threshold:
                self.drag_edge = "top"
            elif abs(img_y - y2) < edge_threshold:
                self.drag_edge = "bottom"
            else:
                self.drag_edge = "move"

        self.update_display()

    def on_canvas_drag(self, event):
        """Canvas drag handling"""
        if not self.dragging or self.selected_box_index < 0:
            return

        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

        # Coordinate transformation considering display scale
        scale = self.scale_var.get()
        offset_x = int(self.x_var.get())
        offset_y = int(self.y_var.get())

        img_x = int((x - offset_x) / scale)
        img_y = int((y - offset_y) / scale)

        # Update bounding box
        bbox = self.bounding_boxes[self.selected_box_index]
        x1, y1, x2, y2 = bbox

        if self.drag_edge == "left":
            x1 = img_x
        elif self.drag_edge == "right":
            x2 = img_x
        elif self.drag_edge == "top":
            y1 = img_y
        elif self.drag_edge == "bottom":
            y2 = img_y
        elif self.drag_edge == "move":
            dx = img_x - self.drag_start_x
            dy = img_y - self.drag_start_y
            x1 += dx
            y1 += dy
            x2 += dx
            y2 += dy
            self.drag_start_x = img_x
            self.drag_start_y = img_y

        # Keep coordinate order correct
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        self.bounding_boxes[self.selected_box_index] = [x1, y1, x2, y2]
        self.update_display()

    def on_canvas_release(self, event):
        """Canvas release handling"""
        self.dragging = False
        self.drag_edge = None

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

        # Change cursor
        cursor = "arrow"
        for bbox in self.bounding_boxes:
            x1, y1, x2, y2 = bbox
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                edge_threshold = 10
                if abs(img_x - x1) < edge_threshold or abs(img_x - x2) < edge_threshold:
                    cursor = "sb_h_double_arrow"
                elif (
                    abs(img_y - y1) < edge_threshold or abs(img_y - y2) < edge_threshold
                ):
                    cursor = "sb_v_double_arrow"
                else:
                    cursor = "fleur"
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
