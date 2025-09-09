import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import os
import json
from pathlib import Path
from .file_managment import load_annotations_json, save_annotations_obb_dota_json
from .transformer import Transformer

class Labeler:
    def __init__(self, root):
        self.root = root
        self.root.title("Bounding Box Labeling Tool")
        self.root.geometry("1200x800")

        # Data storage
        self.image_folder = None
        self.image_files = []
        self.current_index = 0
        self.annotations = {}  # {filename: [{class, x1, y1, x2, y2}, ...]}
        self.class_names = []  # Default classes
        self.current_class = ""

        # Drawing state
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        self.scale_factor = 1.0
        self.image_offset_x = 0
        self.image_offset_y = 0
        self.original_image_size = (0, 0)

        # Display state
        self.canvas_image = None
        self.display_image = None
        self.photo_image = None

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 5))

        # Folder and navigation controls
        ttk.Button(control_frame, text="Select Folder",
                  command=self.select_folder).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Button(control_frame, text="Previous",
                  command=self.previous_image).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(control_frame, text="Next",
                  command=self.next_image).pack(side=tk.LEFT, padx=(0, 10))

        self.progress_label = ttk.Label(control_frame, text="No folder selected")
        self.progress_label.pack(side=tk.LEFT, padx=(0, 20))

        # Save/Load controls
        ttk.Button(control_frame, text="Save Annotations",
                  command=self.save_annotations).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(control_frame, text="Load Annotations",
                  command=self.load_annotations).pack(side=tk.RIGHT, padx=(10, 0))
        ttk.Button(control_frame, text="Transform Images",
                  command=self.transform_images_and_labels).pack(side=tk.RIGHT, padx=(10, 0))

        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel - Image canvas
        canvas_frame = ttk.LabelFrame(content_frame, text="Image", padding="5")
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Canvas with scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(canvas_container, bg="white", width=800, height=600)

        h_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL, command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas bindings for drawing
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.finish_drawing)
        self.canvas.bind("<Button-3>", self.delete_box)  # Right click

        # Right panel - Controls
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # Current image info
        info_frame = ttk.LabelFrame(right_panel, text="Current Image", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.filename_label = ttk.Label(info_frame, text="No image loaded", wraplength=200)
        self.filename_label.pack(anchor=tk.W)

        self.image_size_label = ttk.Label(info_frame, text="Size: -")
        self.image_size_label.pack(anchor=tk.W)

        self.box_count_label = ttk.Label(info_frame, text="Boxes: 0")
        self.box_count_label.pack(anchor=tk.W)

        # Class selection
        self.class_frame = ttk.LabelFrame(right_panel, text="Current Class", padding="10")
        self.class_frame.pack(fill=tk.X, pady=(0, 10))

        self.class_var = tk.StringVar(value=self.current_class)

        for class_name in self.class_names:
            ttk.Radiobutton(self.class_frame, text=class_name, variable=self.class_var,
                           value=class_name, command=self.class_changed).pack(anchor=tk.W)

        # Custom class
        custom_frame = ttk.Frame(self.class_frame)
        custom_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(custom_frame, text="Custom:").pack(anchor=tk.W)
        self.custom_class_var = tk.StringVar()
        custom_entry = ttk.Entry(custom_frame, textvariable=self.custom_class_var, width=15)
        custom_entry.pack(fill=tk.X, pady=2)
        custom_entry.bind('<Return>', self.add_custom_class)

        ttk.Button(custom_frame, text="Add Class",
                  command=self.add_custom_class).pack(pady=2)

        # Class select keybind 1-9
        for i in range(1, 10):
            self.root.bind(f'<KeyPress-{i}>', lambda e, num=i: self.num_class_changed(num))

        # Bounding boxes list
        boxes_frame = ttk.LabelFrame(right_panel, text="Bounding Boxes", padding="10")
        boxes_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Listbox with scrollbar
        listbox_frame = ttk.Frame(boxes_frame)
        listbox_frame.pack(fill=tk.BOTH, expand=True)

        self.boxes_listbox = tk.Listbox(listbox_frame, height=10)
        listbox_scrollbar = ttk.Scrollbar(listbox_frame, orient=tk.VERTICAL,
                                         command=self.boxes_listbox.yview)

        self.boxes_listbox.configure(yscrollcommand=listbox_scrollbar.set)
        self.boxes_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        listbox_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Bind listbox selection
        self.boxes_listbox.bind('<Double-Button-1>', self.edit_box)

        # Box management buttons
        box_buttons = ttk.Frame(boxes_frame)
        box_buttons.pack(fill=tk.X, pady=(5, 0))

        ttk.Button(box_buttons, text="Delete Selected",
                  command=self.delete_selected_box).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(box_buttons, text="Clear All",
                  command=self.clear_all_boxes).pack(side=tk.LEFT)

        # Instructions
        instructions_frame = ttk.LabelFrame(right_panel, text="Instructions", padding="10")
        instructions_frame.pack(fill=tk.X)

        instructions = """• Select class first
• Click and drag to draw box
• Right-click box to delete
• Double-click list item to edit
• Use arrow keys to navigate"""

        ttk.Label(instructions_frame, text=instructions, justify=tk.LEFT,
                 font=("Arial", 8)).pack(anchor=tk.W)

        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.previous_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<Delete>', lambda e: self.delete_selected_box())
        self.root.focus_set()

    def select_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.image_folder = folder
            self.load_images()

    def load_images(self):
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

        self.image_files = []
        for ext in extensions:
            self.image_files.extend(Path(self.image_folder).glob(f'*{ext}'))
            # self.image_files.extend(Path(self.image_folder).glob(f'*{ext.upper()}'))

        self.image_files = sorted(self.image_files)

        if self.image_files:
            self.current_index = 0
            self.display_current_image()
        else:
            messagebox.showwarning("No Images", "No supported image files found.")

    def display_current_image(self):
        if not self.image_files:
            return

        image_path = self.image_files[self.current_index]
        filename = image_path.name

        try:
            # Load original image
            self.original_image = Image.open(image_path)
            self.original_image_size = self.original_image.size

            # Calculate scale to fit canvas while maintaining aspect ratio
            canvas_width = self.canvas.winfo_width() or 800
            canvas_height = self.canvas.winfo_height() or 600

            scale_x = canvas_width / self.original_image_size[0]
            scale_y = canvas_height / self.original_image_size[1]
            self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up

            # Resize image for display
            new_size = (int(self.original_image_size[0] * self.scale_factor),
                    int(self.original_image_size[1] * self.scale_factor))

            self.display_image = self.original_image.resize(new_size, Image.Resampling.LANCZOS)

            # Calculate centering offsets
            self.image_offset_x = max(0, (canvas_width - new_size[0]) // 2)
            self.image_offset_y = max(0, (canvas_height - new_size[1]) // 2)

            # Convert to PhotoImage
            self.photo_image = ImageTk.PhotoImage(self.display_image)

            # Clear canvas and display image
            self.canvas.delete("all")
            self.canvas_image = self.canvas.create_image(
                self.image_offset_x, self.image_offset_y,
                anchor=tk.NW, image=self.photo_image
            )

            # Set scroll region
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

            # Draw existing annotations
            self.draw_annotations()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

        self.update_ui_info()

    def draw_annotations(self):
        if not self.image_files:
            return

        filename = self.image_files[self.current_index].name
        annotations = self.annotations.get(filename, [])

        colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "cyan"]
        class_colors = {}

        for i, ann in enumerate(annotations):
            class_name = ann['class']
            if class_name not in class_colors:
                class_colors[class_name] = colors[len(class_colors) % len(colors)]

            color = class_colors[class_name]

            # Convert all 4 corner coordinates to display coordinates
            x1 = ann['x1'] * self.scale_factor + self.image_offset_x
            y1 = ann['y1'] * self.scale_factor + self.image_offset_y
            x2 = ann['x2'] * self.scale_factor + self.image_offset_x
            y2 = ann['y2'] * self.scale_factor + self.image_offset_y
            x3 = ann['x3'] * self.scale_factor + self.image_offset_x
            y3 = ann['y3'] * self.scale_factor + self.image_offset_y
            x4 = ann['x4'] * self.scale_factor + self.image_offset_x
            y4 = ann['y4'] * self.scale_factor + self.image_offset_y

            # Draw polygon using all 4 corners (clockwise from top-left)
            points = [x1, y1, x2, y2, x3, y3, x4, y4]
            poly_id = self.canvas.create_polygon(points, outline=color, fill='', width=2, tags=f"box_{i}")

            # Draw label at the top-left corner
            label_text = f"{class_name} ({i+1})"
            self.canvas.create_text(x1, y1-10, text=label_text, anchor=tk.SW,
                                fill=color, font=("Arial", 9, "bold"), tags=f"label_{i}")

    def start_drawing(self, event):
        if not self.canvas_image:
            return

        # Check if click is on the image
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        if (self.image_offset_x <= canvas_x <= self.image_offset_x + self.display_image.width and
            self.image_offset_y <= canvas_y <= self.image_offset_y + self.display_image.height):

            self.drawing = True
            self.start_x = canvas_x
            self.start_y = canvas_y

    def draw_rectangle(self, event):
        if not self.drawing:
            return

        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Remove previous temporary rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)

        # Draw new temporary rectangle
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, canvas_x, canvas_y,
            outline="yellow", width=2, tags="temp"
        )

    def finish_drawing(self, event):
        if not self.drawing:
            return
        self.drawing = False
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        # Remove temporary rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
        # Calculate bounding box in original image coordinates
        x1 = min(self.start_x, canvas_x) - self.image_offset_x
        y1 = min(self.start_y, canvas_y) - self.image_offset_y
        x2 = max(self.start_x, canvas_x) - self.image_offset_x
        y2 = max(self.start_y, canvas_y) - self.image_offset_y
        # Convert to original image scale
        x1 = max(0, x1 / self.scale_factor)
        y1 = max(0, y1 / self.scale_factor)
        x2 = min(self.original_image_size[0], x2 / self.scale_factor)
        y2 = min(self.original_image_size[1], y2 / self.scale_factor)
        # Only add if box has minimum size
        if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
            # Pass corners to add_annotation (it will handle 4-corner conversion)
            self.add_annotation(x1, y1, x2, y2, self.current_class)


    def add_annotation(self, x1, y1, x2, y2, class_name):
        if not self.image_files:
            return

        filename = self.image_files[self.current_index].name

        if filename not in self.annotations:
            self.annotations[filename] = []

        # Store as 4 corners format (clockwise from top-left)
        # x1,y1 = top-left, x2,y2 = top-right, x3,y3 = bottom-right, x4,y4 = bottom-left
        annotation = {
            'class': class_name,
            'x1': round(min(x1, x2)),  # top-left x
            'y1': round(min(y1, y2)),  # top-left y
            'x2': round(max(x1, x2)),  # top-right x
            'y2': round(min(y1, y2)),  # top-right y
            'x3': round(max(x1, x2)),  # bottom-right x
            'y3': round(max(y1, y2)),  # bottom-right y
            'x4': round(min(x1, x2)),  # bottom-left x
            'y4': round(max(y1, y2))   # bottom-left y
        }

        self.annotations[filename].append(annotation)
        self.display_current_image()  # Refresh display

    def delete_box(self, event):
        """Delete box on right-click"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Find clicked item
        item = self.canvas.find_closest(canvas_x, canvas_y)[0]
        tags = self.canvas.gettags(item)

        for tag in tags:
            if tag.startswith("box_"):
                box_index = int(tag.split("_")[1])
                self.delete_annotation(box_index)
                break

    def delete_annotation(self, index):
        if not self.image_files:
            return

        filename = self.image_files[self.current_index].name
        annotations = self.annotations.get(filename, [])

        if 0 <= index < len(annotations):
            del annotations[index]
            self.display_current_image()  # Refresh display

    def delete_selected_box(self):
        selection = self.boxes_listbox.curselection()
        if selection:
            self.delete_annotation(selection[0])

    def clear_all_boxes(self):
        if not self.image_files:
            return

        result = messagebox.askyesno("Confirm", "Delete all bounding boxes for this image?")
        if result:
            filename = self.image_files[self.current_index].name
            self.annotations[filename] = []
            self.display_current_image()

    def class_changed(self):
        self.current_class = self.class_var.get()

    def num_class_changed(self, num):
        index = min(num - 1, len(self.class_names) - 1)
        if index >= 0 and self.class_names:
            self.current_class = self.class_names[index]
            self.class_var.set(self.current_class)

    def add_custom_class(self, event=None):
        custom_class = self.custom_class_var.get().strip()
        if custom_class and custom_class not in self.class_names:
            self.class_names.append(custom_class)
            self.current_class = custom_class
            self.class_var.set(custom_class)
            self.custom_class_var.set("")

            # Rebuild class selection UI
            ttk.Radiobutton(self.class_frame, text=custom_class, variable=self.class_var,
                           value=custom_class, command=self.class_changed).pack(anchor=tk.W)


    def edit_box(self, event):
        selection = self.boxes_listbox.curselection()
        if not selection:
            return

        index = selection[0]
        filename = self.image_files[self.current_index].name
        annotations = self.annotations.get(filename, [])

        if index < len(annotations):
            ann = annotations[index]
            new_class = simpledialog.askstring("Edit Class", f"Current class: {ann['class']}\nNew class:", initialvalue=ann['class'])
            if new_class:
                ann['class'] = new_class
                self.display_current_image()


    def update_ui_info(self):
        if not self.image_files:
            return

        filename = self.image_files[self.current_index].name
        annotations = self.annotations.get(filename, [])

        # Update labels
        self.filename_label.configure(text=f"File: {filename}")
        self.image_size_label.configure(text=f"Size: {self.original_image_size[0]}x{self.original_image_size[1]}")
        self.progress_label.configure(text=f"Image {self.current_index + 1} of {len(self.image_files)}")
        self.box_count_label.configure(text=f"Boxes: {len(annotations)}")

        # Update boxes listbox - show as rectangles using corner coordinates
        self.boxes_listbox.delete(0, tk.END)
        for i, ann in enumerate(annotations):
            # Calculate width and height from corners
            width = abs(ann['x3'] - ann['x1'])
            height = abs(ann['y3'] - ann['y1'])
            bbox_text = f"{i+1}. {ann['class']} ({ann['x1']},{ann['y1']}) {width}x{height}"
            self.boxes_listbox.insert(tk.END, bbox_text)

    def previous_image(self):
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()

    def next_image(self):
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_current_image()

    def save_annotations(self):
        if not self.annotations:
            messagebox.showwarning("No Annotations", "No annotations to save.")
            return

        file_path = filedialog.askdirectory(title="Save Annotations")
        save_annotations_obb_dota_json(self.annotations, self.image_files, file_path)

    def load_annotations(self):
        file_path = filedialog.askopenfilename(
            title="Load Annotations",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )

        if file_path:
            try:
                self.annotations = load_annotations_json(file_path=file_path)

                # Extract all classes from loaded annotations
                all_classes = set()
                for annotations in self.annotations.values():
                    for ann in annotations:
                        all_classes.add(ann['class'])

                # Add new classes to class list
                for class_name in all_classes:
                    if class_name not in self.class_names:
                        self.class_names.append(class_name)
                        ttk.Radiobutton(self.class_frame, text=class_name, variable=self.class_var,
                            value=class_name, command=self.class_changed).pack(anchor=tk.W)

                self.display_current_image()

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load annotations:\n{str(e)}")

    def transform_images_and_labels(self):
        """Open the data augmentation transformer"""

        # Check if there are any annotations
        if not self.annotations:
            messagebox.showwarning("No Annotations",
                                "No annotations found. Please annotate some images before proceeding.")
            return

        try:
            image_sizes = {}
            if self.image_files:
                from PIL import Image
                for image_file in self.image_files:
                    try:
                        with Image.open(image_file) as img:
                            image_sizes[image_file.name] = img.size
                    except Exception as e:
                        print(f"Warning: Could not get size for {image_file.name}: {e}")
                        image_sizes[image_file.name] = (1920, 1088)  # default size

            transformer = Transformer(
                parent=self.root,
                annotations=self.annotations,
                image_files=self.image_files,
                image_sizes=image_sizes
            )

        except Exception as e:
            messagebox.showerror("Error", f"Failed to create transformer:\n{str(e)}")
