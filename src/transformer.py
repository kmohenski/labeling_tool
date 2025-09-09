import random
import math
import cv2

import numpy as np
import tkinter as tk

from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter

from .file_managment import save_annotations_yolo

class Transformer:
    def __init__(self, parent=None, annotations=None, image_files=None, image_sizes=None):
        self.parent = parent
        self.window = tk.Toplevel(parent) if parent else tk.Tk()
        self.window.title("Data Augmentation Transformer")
        self.window.geometry("700x950")
        self.window.resizable(True, True)

        # Make window modal
        if parent:
            self.window.transient(parent)
            self.window.grab_set()

        # Data from parent labeler
        self.original_annotations = annotations or {}
        self.original_image_files = image_files or []
        self.image_sizes = image_sizes or {}

        # Configuration variables
        self.output_folder = None
        self.standard_size = [tk.IntVar(value=600), tk.IntVar(value=600)]  # width, height
        self.generation_multiplier = tk.DoubleVar(value=2.0)  # 2x = 200%
        self.train_split = tk.IntVar(value=60)
        self.val_split = tk.IntVar(value=20)
        self.test_split = tk.IntVar(value=20)
        self.test_mode = tk.BooleanVar(value=False)

        # Transform states
        self.transforms = {}

        # Multi-image transforms (first priority)
        self.multi_image_transforms = {
            'Mosaic': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.3)},
            'Mixup': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.2), 'scale': tk.DoubleVar(value=0.15)}
        }

        # Single image transforms
        self.single_image_transforms = {
            'Horizontal Flip': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.5)},
            'Vertical Flip': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.2)},
            'Random Rotate': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.3), 'degrees': tk.IntVar(value=90)},
            'HSV': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.4), 'h_gain': tk.IntVar(value=30), 's_gain': tk.IntVar(value=20), 'v_gain': tk.IntVar(value=20)},
            'Random Brightness': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.3), 'factor': tk.DoubleVar(value=0.2)},
            'Random Contrast': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.3), 'factor': tk.DoubleVar(value=0.2)},
            'Gaussian Blur': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.2), 'kernel': tk.IntVar(value=3)},
            'Random Affine': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.3), 'scale': tk.DoubleVar(value=0.9)},
            'Noise': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.2), 'strength': tk.DoubleVar(value=0.05)},
            'CutOut': {'enabled': tk.BooleanVar(), 'prob': tk.DoubleVar(value=0.2), 'size': tk.DoubleVar(value=0.1)}
        }

        self.setup_ui()

        # Center window on parent
        if parent:
            self.center_window()

    def center_window(self):
        self.window.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() // 2) - (self.window.winfo_width() // 2)
        y = self.parent.winfo_y() + (self.parent.winfo_height() // 2) - (self.window.winfo_height() // 2)
        self.window.geometry(f"+{x}+{y}")

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(main_frame, text="Data Augmentation Transformer",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 15))

        # Dataset info
        info_frame = ttk.LabelFrame(main_frame, text="Dataset Information", padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        info_text = f"Loaded: {len(self.original_image_files)} images"
        ttk.Label(info_frame, text=info_text, font=("Arial", 10, "bold")).pack(anchor=tk.W)

        # Output folder selection
        folder_frame = ttk.LabelFrame(main_frame, text="Output Settings", padding="10")
        folder_frame.pack(fill=tk.X, pady=(0, 10))

        folder_select_frame = ttk.Frame(folder_frame)
        folder_select_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(folder_select_frame, text="Select Output Folder",
                  command=self.select_output_folder).pack(side=tk.LEFT, padx=(0, 10))

        self.folder_label = ttk.Label(folder_select_frame, text="No folder selected",
                                     foreground="gray")
        self.folder_label.pack(side=tk.LEFT)

        # Standard size setting
        size_frame = ttk.Frame(folder_frame)
        size_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(size_frame, text="Standard output size:").pack(side=tk.LEFT)
        ttk.Spinbox(size_frame, from_=224, to=2048, width=6,
                   textvariable=self.standard_size[0]).pack(side=tk.LEFT, padx=(5, 2))
        ttk.Label(size_frame, text="×").pack(side=tk.LEFT)
        ttk.Spinbox(size_frame, from_=224, to=2048, width=6,
                   textvariable=self.standard_size[1]).pack(side=tk.LEFT, padx=(2, 10))
        ttk.Label(size_frame, text="pixels").pack(side=tk.LEFT)

        # Generation settings
        gen_frame = ttk.LabelFrame(main_frame, text="Generation Settings", padding="10")
        gen_frame.pack(fill=tk.X, pady=(0, 10))

        # Generation multiplier
        multiplier_frame = ttk.Frame(gen_frame)
        multiplier_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(multiplier_frame, text="Generate images:").pack(side=tk.LEFT)

        def update_multiplier_label(*args):
            mult = self.generation_multiplier.get()
            original_count = len(self.original_image_files)
            total_generated = int(original_count * mult)
            percentage = int((mult - 1) * 100)
            self.multiplier_label.config(text=f"{total_generated} images ({percentage}% more)")

        multiplier_scale = ttk.Scale(multiplier_frame, from_=1.0, to=6.0,
                                   variable=self.generation_multiplier,
                                   orient=tk.HORIZONTAL, length=200)
        multiplier_scale.pack(side=tk.LEFT, padx=(10, 10))

        self.multiplier_label = ttk.Label(multiplier_frame, text="")
        self.multiplier_label.pack(side=tk.LEFT)

        self.generation_multiplier.trace('w', update_multiplier_label)
        update_multiplier_label()  # Initial update

        # Train/Test/Val split
        split_frame = ttk.Frame(gen_frame)
        split_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(split_frame, text="Dataset split:").pack(side=tk.LEFT)

        # Train split
        train_frame = ttk.Frame(split_frame)
        train_frame.pack(side=tk.LEFT, padx=(20, 10))
        ttk.Label(train_frame, text="Train").pack()
        train_scale = ttk.Scale(train_frame, from_=0, to=100, variable=self.train_split,
                               orient=tk.VERTICAL, length=80)
        train_scale.pack()
        self.train_label = ttk.Label(train_frame, text="60%")
        self.train_label.pack()

        # Val split
        val_frame = ttk.Frame(split_frame)
        val_frame.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(val_frame, text="Val").pack()
        val_scale = ttk.Scale(val_frame, from_=0, to=100, variable=self.val_split,
                             orient=tk.VERTICAL, length=80)
        val_scale.pack()
        self.val_label = ttk.Label(val_frame, text="20%")
        self.val_label.pack()

        # Test split
        test_frame = ttk.Frame(split_frame)
        test_frame.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(test_frame, text="Test").pack()
        test_scale = ttk.Scale(test_frame, from_=0, to=100, variable=self.test_split,
                              orient=tk.VERTICAL, length=80)
        test_scale.pack()
        self.test_label = ttk.Label(test_frame, text="20%")
        self.test_label.pack()

        # Bind split updates
        def update_splits(*args):
            train = self.train_split.get()
            val = self.val_split.get()
            test = self.test_split.get()

            total = train + val + test
            if total != 100:
                if total > 100:
                    factor = 100 / total
                    self.train_split.set(int(train * factor))
                    self.val_split.set(int(val * factor))
                    self.test_split.set(100 - self.train_split.get() - self.val_split.get())

            self.train_label.config(text=f"{self.train_split.get()}%")
            self.val_label.config(text=f"{self.val_split.get()}%")
            self.test_label.config(text=f"{self.test_split.get()}%")

        for var in [self.train_split, self.val_split, self.test_split]:
            var.trace('w', update_splits)

        # Test mode checkbox
        test_mode_frame = ttk.Frame(gen_frame)
        test_mode_frame.pack(fill=tk.X)

        ttk.Checkbutton(test_mode_frame, text="Test mode (save transform examples)",
                       variable=self.test_mode).pack(side=tk.LEFT)

        # Transforms section
        transforms_frame = ttk.LabelFrame(main_frame, text="Transformations", padding="10")
        transforms_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create scrollable frame for transforms
        canvas_frame = ttk.Frame(transforms_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        self.create_transforms_ui()

        # Bottom buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(button_frame, text="Generate Augmented Dataset",
                  command=self.generate_dataset).pack(side=tk.RIGHT, padx=(10, 0))

        ttk.Button(button_frame, text="Close",
                  command=self.window.destroy).pack(side=tk.RIGHT)

    def create_transforms_ui(self):
        # Multi-image transforms section
        multi_frame = ttk.LabelFrame(self.scrollable_frame, text="Multi-Image Transforms", padding="10")
        multi_frame.pack(fill=tk.X, pady=(0, 10), padx=5)

        for transform_name, config in self.multi_image_transforms.items():
            self.create_transform_control(multi_frame, transform_name, config)

        # Single image transforms section
        single_frame = ttk.LabelFrame(self.scrollable_frame, text="Single Image Transforms", padding="10")
        single_frame.pack(fill=tk.X, pady=(0, 10), padx=5)

        for transform_name, config in self.single_image_transforms.items():
            self.create_transform_control(single_frame, transform_name, config)

    def create_transform_control(self, parent, transform_name, config):
        # Main frame for this transform
        transform_frame = ttk.Frame(parent)
        transform_frame.pack(fill=tk.X, pady=2)

        # Checkbox
        checkbox = ttk.Checkbutton(transform_frame, text=transform_name,
                                  variable=config['enabled'])
        checkbox.pack(side=tk.LEFT, anchor=tk.W)

        # Parameters frame (shown when enabled)
        params_frame = ttk.Frame(transform_frame)
        params_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(20, 0))

        # Probability control
        ttk.Label(params_frame, text="Prob:").pack(side=tk.LEFT)
        prob_scale = ttk.Scale(params_frame, from_=0.0, to=1.0, variable=config['prob'],
                              orient=tk.HORIZONTAL, length=100)
        prob_scale.pack(side=tk.LEFT, padx=(5, 10))

        prob_label = ttk.Label(params_frame, text=f"{config['prob'].get():.2f}")
        prob_label.pack(side=tk.LEFT, padx=(0, 15))

        def update_prob_label(*args):
            prob_label.config(text=f"{config['prob'].get():.2f}")

        config['prob'].trace('w', update_prob_label)

        # Additional parameters based on transform type
        if transform_name == 'Random Rotate':
            ttk.Label(params_frame, text="Degrees:").pack(side=tk.LEFT)
            ttk.Spinbox(params_frame, from_=0, to=180, width=6,
                       textvariable=config['degrees']).pack(side=tk.LEFT, padx=(5, 0))

        if transform_name == 'HSV':
            for param, label in [('h_gain', 'H:'), ('s_gain', 'S:'), ('v_gain', 'V:')]:
                ttk.Label(params_frame, text=label).pack(side=tk.LEFT, padx=(10, 0))
                ttk.Spinbox(params_frame, from_=0, to=50, width=4,
                           textvariable=config[param]).pack(side=tk.LEFT, padx=(2, 0))

        elif transform_name in ['Random Brightness', 'Random Contrast']:
            ttk.Label(params_frame, text="Factor:").pack(side=tk.LEFT)
            ttk.Scale(params_frame, from_=0.0, to=0.8, variable=config['factor'],
                     orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT, padx=(5, 0))

        elif transform_name == 'Gaussian Blur':
            ttk.Label(params_frame, text="Kernel:").pack(side=tk.LEFT)
            ttk.Spinbox(params_frame, from_=1, to=15, width=4,
                       textvariable=config['kernel']).pack(side=tk.LEFT, padx=(5, 0))

        elif transform_name == 'Random Affine':
            ttk.Label(params_frame, text="Scale:").pack(side=tk.LEFT)
            ttk.Scale(params_frame, from_=0.7, to=1.0, variable=config['scale'],
                     orient=tk.HORIZONTAL, length=60).pack(side=tk.LEFT, padx=(5, 0))

        elif transform_name == 'Noise':
            ttk.Label(params_frame, text="Strength:").pack(side=tk.LEFT)
            ttk.Scale(params_frame, from_=0.0, to=0.2, variable=config['strength'],
                     orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT, padx=(5, 0))

        elif transform_name == 'CutOut':
            ttk.Label(params_frame, text="Size:").pack(side=tk.LEFT)
            ttk.Scale(params_frame, from_=0.0, to=0.3, variable=config['size'],
                     orient=tk.HORIZONTAL, length=80).pack(side=tk.LEFT, padx=(5, 0))

        elif transform_name == 'Mixup':
            ttk.Label(params_frame, text="Scale:").pack(side=tk.LEFT)
            ttk.Scale(params_frame, from_=0.0, to=0.5, variable=config['scale'],
                     orient=tk.HORIZONTAL, length=60).pack(side=tk.LEFT, padx=(5, 0))

    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder for Augmented Dataset")
        if folder:
            self.output_folder = folder
            self.folder_label.config(text=f"Output: {Path(folder).name}", foreground="black")

    def generate_dataset(self):
        if not self.output_folder:
            messagebox.showerror("No Output Folder", "Please select an output folder first.")
            return

        if not self.original_annotations:
            messagebox.showerror("No Annotations", "No annotations loaded.")
            return

        enabled_transforms = []
        for name, config in {**self.multi_image_transforms, **self.single_image_transforms}.items():
            if config['enabled'].get():
                enabled_transforms.append(name)

        if not enabled_transforms:
            messagebox.showwarning("No Transforms", "Please enable at least one transformation.")
            return

        total_split = self.train_split.get() + self.val_split.get() + self.test_split.get()
        if abs(total_split - 100) > 1:
            messagebox.showerror("Invalid Split", "Train + Val + Test must equal 100%")
            return

        original_count = len(self.original_image_files)
        multiplier = self.generation_multiplier.get()
        total_images = int(original_count * multiplier)
        new_images = total_images - original_count

        result = messagebox.askyesno(
            "Generate Dataset",
            f"This will generate an augmented dataset:\n\n"
            f"Original images: {original_count}\n"
            f"New images to generate: {new_images}\n"
            f"Total output: {total_images} images\n\n"
            f"Standard size: {self.standard_size[0].get()}×{self.standard_size[1].get()}\n"
            f"Split: {self.train_split.get()}% train, {self.val_split.get()}% val, {self.test_split.get()}% test\n"
            f"Enabled transforms: {len(enabled_transforms)}\n"
            f"Test mode: {'Yes' if self.test_mode.get() else 'No'}\n\n"
            "Continue?"
        )

        if result:
            try:
                self.perform_augmentation(total_images)

                messagebox.showinfo(
                    "Success",
                    f"Dataset generation completed!\n\n"
                    f"Generated {total_images} total images\n"
                    f"Saved to: {self.output_folder}\n"
                    f"{'Test images saved to test_generation/' if self.test_mode.get() else ''}"
                )

            except Exception as e:
                messagebox.showerror("Generation Failed", f"An error occurred:\n{str(e)}")

    def select_random_image(self, has_ann: bool = False):
        source_idx = random.randint(0, len(self.all_image_paths) - 1)
        source_path, _ = self.all_image_paths[source_idx]
        source_filename = source_path.name

        if has_ann:
            while not self.all_annotations.get(source_filename, None):
                source_idx = random.randint(0, len(self.all_image_paths) - 1)
                source_path, _ = self.all_image_paths[source_idx]
                source_filename = source_path.name

        return Image.open(source_path).copy(), source_filename

    def perform_augmentation(self, total_images):
        """Perform the actual augmentation process"""
        output_path = Path(self.output_folder)

        for split in ['train', 'val', 'test']:
            (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
            (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)

        if self.test_mode.get():
            (output_path / 'test_generation' / 'images').mkdir(parents=True, exist_ok=True)
            (output_path / 'test_generation' / 'labels').mkdir(parents=True, exist_ok=True)

        print(f"Starting augmentation process...")
        print(f"Target: {total_images} total images")

        standard_width = self.standard_size[0].get()
        standard_height = self.standard_size[1].get()

        self.all_annotations = {}
        self.all_image_paths = []

        # Process original images
        for idx, image_file in enumerate(self.original_image_files):
            filename = image_file.name
            resized_image, resized_annotations = self.resize_image_and_annotations(
                image_file, self.original_annotations.get(filename, {}),
                (standard_width, standard_height)
            )

            split = self.determine_split(idx, len(self.original_image_files))

            output_filename = f"orig_{idx:04d}_{filename}"
            image_path = output_path / 'images' / split / output_filename
            resized_image.save(image_path)

            self.all_annotations[output_filename] = resized_annotations
            self.all_image_paths.append((image_path, split))

        original_count = len(self.original_image_files)
        images_to_generate = total_images - original_count

        test_image, test_filename = self.select_random_image(True)
        test_annotations = self.all_annotations[test_filename]

        for i in range(images_to_generate):
            print(f"Generating image {i+1}/{images_to_generate}...")

            source_image, source_filename = self.select_random_image()
            source_annotations = self.all_annotations[source_filename]

            augmented_image, augmented_annotations = self.apply_random_transforms(
                source_image, source_annotations
            )

            split = self.determine_split(original_count + i, total_images)

            aug_filename = f"aug_{i:04d}_{source_filename}"
            aug_path = output_path / 'images' / split / aug_filename
            augmented_image.save(aug_path)

            self.all_annotations[aug_filename] = augmented_annotations

        try:
            for split in ['train', 'val', 'test']:
                split_annotations = {}
                split_image_files = []

                split_dir = output_path / 'images' / split
                for img_file in split_dir.glob('*'):
                    if img_file.name in self.all_annotations:
                        split_annotations[img_file.name] = self.all_annotations[img_file.name]
                        split_image_files.append(img_file)

                if split_annotations:
                    save_path = output_path / 'labels' / split
                    save_annotations_yolo(split_annotations, split_image_files, str(save_path))

        except Exception as e:
            print(f"Error saving annotations: {e}")

        if self.test_mode.get() and test_image is not None:
            self.generate_test_examples(test_image, test_annotations, output_path)


    def resize_image_and_annotations(self, image_file, annotations, target_size):
        """Resize image and scale annotations accordingly"""
        image = Image.open(image_file)

        original_width, original_height = image.size
        target_width, target_height = target_size
        scale_x = target_width / original_width
        scale_y = target_height / original_height

        resized_image = image.resize(target_size, Image.Resampling.LANCZOS)

        resized_annotations = []
        for ann in annotations:
            resized_ann = {
                'class': ann['class'],
                'x1': int(ann['x1'] * scale_x),
                'y1': int(ann['y1'] * scale_y),
                'x2': int(ann['x2'] * scale_x),
                'y2': int(ann['y2'] * scale_y),
                'x3': int(ann['x3'] * scale_x),
                'y3': int(ann['y3'] * scale_y),
                'x4': int(ann['x4'] * scale_x),
                'y4': int(ann['y4'] * scale_y)
            }
            resized_annotations.append(resized_ann)

        return resized_image, resized_annotations

    def determine_split(self, index, total):
        """Determine which split (train/val/test) an image belongs to"""
        train_pct = self.train_split.get() / 100
        val_pct = self.val_split.get() / 100

        ratio = index / total

        if ratio < train_pct:
            return 'train'
        elif ratio < train_pct + val_pct:
            return 'val'
        else:
            return 'test'


    ################################
    ### Apply Transform function ###
    ################################
    def apply_random_transforms(self, image, annotations):
        """Apply random transforms based on enabled options and probabilities"""

        transformed_image = image.copy()
        transformed_annotations = [ann.copy() for ann in annotations]

        all_transforms = {**self.multi_image_transforms, **self.single_image_transforms}

        for transform_name, config in all_transforms.items():
            if not config['enabled'].get() or random.random() > config['prob'].get():
                continue

            transformed_image, transformed_annotations = self.apply_single_transform(
                transformed_image, transformed_annotations, transform_name, config)

        return transformed_image, transformed_annotations


    ##################################
    ### Transform helper functions ###
    ##################################
    def apply_horizontal_flip(self, image, annotations):
        """Apply horizontal flip to image and annotations"""
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        width = image.width

        flipped_annotations = []
        for ann in annotations:
            flipped_ann = ann.copy()
            flipped_ann['x1'] = width - ann['x2']
            flipped_ann['x2'] = width - ann['x1']
            flipped_ann['x3'] = width - ann['x4']
            flipped_ann['x4'] = width - ann['x3']
            flipped_annotations.append(flipped_ann)

        return flipped_image, flipped_annotations

    def apply_vertical_flip(self, image, annotations):
        """Apply vertical flip to image and annotations"""
        flipped_image = image.transpose(Image.FLIP_TOP_BOTTOM)
        height = image.height

        flipped_annotations = []
        for ann in annotations:
            flipped_ann = ann.copy()
            flipped_ann['y1'] = height - ann['y4']
            flipped_ann['y2'] = height - ann['y3']
            flipped_ann['y3'] = height - ann['y2']
            flipped_ann['y4'] = height - ann['y1']
            flipped_annotations.append(flipped_ann)

        return flipped_image, flipped_annotations

    def apply_hsv_transform(self, image, h_gain, s_gain, v_gain):
        """Apply HSV color adjustments"""
        img_array = np.array(image)
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV).astype(np.float32)

        # Hue
        hsv[:, :, 0] = (hsv[:, :, 0] + h_gain * 180) % 180
        # Saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + s_gain), 0, 255)
        # Value
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + v_gain), 0, 255)

        hsv = hsv.astype(np.uint8)
        rgb_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return Image.fromarray(rgb_array)

    def apply_noise(self, image, strength):
        """Add random noise to image"""
        img_array = np.array(image)
        noise = np.random.normal(0, strength * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    def apply_cutout(self, image, size_ratio):
        """Apply cutout (random rectangular mask)"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        cut_h = int(h * size_ratio)
        cut_w = int(w * size_ratio)

        y = random.randint(0, h - cut_h)
        x = random.randint(0, w - cut_w)

        img_array[y:y+cut_h, x:x+cut_w] = 0

        return Image.fromarray(img_array)


    ########################
    ### Affine functions ###
    ########################
    def return_inverse_matrix(self, a, b, c, d, tx, ty):
        det = a * d - b * c
        if abs(det) < 1e-6:
            return None

        inv_a = d / det
        inv_b = -c / det
        inv_c = -b / det
        inv_d = a / det
        inv_tx = (c * ty - d * tx) / det
        inv_ty = (b * tx - a * ty) / det

        return (inv_a, inv_b, inv_tx, inv_c, inv_d, inv_ty)

    def apply_affine_transform(self, image, annotations, scale):
        """Apply affine transformation with scaling and shearing only (no translation)"""
        width, height = image.size
        center_x, center_y = width / 2, height / 2

        scale_x = scale * random.uniform(0.8, 0.95)
        scale_y = scale * random.uniform(0.8, 0.95)
        shear_x = random.uniform(-0.1, 0.1)
        shear_y = random.uniform(-0.1, 0.1)

        a = scale_x
        b = shear_y
        c = shear_x
        d = scale_y
        tx = center_x * (1 - a) - center_y * c
        ty = center_y * (1 - d) - center_x * b

        inverse_tuple = self.return_inverse_matrix(a, b, c, d, tx, ty)
        if inverse_tuple == None:
            return image, annotations

        transformed_image = image.transform(
            (width, height),
            Image.AFFINE,
            inverse_tuple,
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0)
        )

        transformed_annotations = []
        for ann in annotations:
            transformed_ann = ann.copy()

            for corner in [('x1', 'y1'), ('x2', 'y2'), ('x3', 'y3'), ('x4', 'y4')]:
                x_key, y_key = corner
                x, y = ann[x_key], ann[y_key]

                new_x = a * x + c * y + tx
                new_y = b * x + d * y + ty

                new_x = max(0, min(width, new_x))
                new_y = max(0, min(height, new_y))

                transformed_ann[x_key] = round(new_x)
                transformed_ann[y_key] = round(new_y)

            transformed_annotations.append(transformed_ann)

        return transformed_image, transformed_annotations


    ##########################
    ### Rotation functions ###
    ##########################
    def rotate_point(self, original_image, rotated_image, degrees, x, y):
        rad = math.radians(-degrees)

        center_x = original_image.width / 2
        center_y = original_image.height / 2

        translated_x = x - center_x
        translated_y = y - center_y

        rotated_x = translated_x * math.cos(rad) - translated_y * math.sin(rad)
        rotated_y = translated_x * math.sin(rad) + translated_y * math.cos(rad)

        new_center_x = rotated_image.width / 2
        new_center_y = rotated_image.height / 2

        expanded_x = rotated_x + new_center_x
        expanded_y = rotated_y + new_center_y

        scale_x = self.standard_size[0].get() / rotated_image.width
        scale_y = self.standard_size[1].get() / rotated_image.height

        final_x = expanded_x * scale_x
        final_y = expanded_y * scale_y

        return int(final_x), int(final_y)

    def apply_rotation(self, image, annotations, degrees):
        """Apply rotation and transform annotations"""
        rotated_image = image.rotate(degrees, expand=True)

        rotated_annotations = []
        for ann in annotations:
            rotated_ann = ann.copy()
            rotated_ann['x1'], rotated_ann['y1'] = self.rotate_point(
                image, rotated_image, degrees, ann['x1'], ann['y1'])
            rotated_ann['x2'], rotated_ann['y2'] = self.rotate_point(
                image, rotated_image, degrees, ann['x2'], ann['y2'])
            rotated_ann['x3'], rotated_ann['y3'] = self.rotate_point(
                image, rotated_image, degrees, ann['x3'], ann['y3'])
            rotated_ann['x4'], rotated_ann['y4'] = self.rotate_point(
                image, rotated_image, degrees, ann['x4'], ann['y4'])

            rotated_annotations.append(rotated_ann)

        final_image = rotated_image.resize(
            (self.standard_size[0].get(), self.standard_size[1].get()),
            Image.Resampling.LANCZOS)

        return final_image, rotated_annotations


    ################################
    ### Mixup transform function ###
    ################################
    def apply_mixup_transform(self, image, annotations, scale):
        second_image, filename = self.select_random_image(True)
        second_image_annotations = self.all_annotations[filename]

        standard_width = self.standard_size[0].get()
        standard_height = self.standard_size[1].get()

        # Scale
        first_scale_factor = random.uniform(1-scale, 1+scale)
        second_scale_factor = random.uniform(1-scale, 1+scale)

        scaled_width1 = int(standard_width * first_scale_factor)
        scaled_height1 = int(standard_height * first_scale_factor)
        image = image.resize((scaled_width1, scaled_height1), Image.Resampling.LANCZOS)

        scaled_width2 = int(standard_width * second_scale_factor)
        scaled_height2 = int(standard_height * second_scale_factor)
        second_image = second_image.resize((scaled_width2, scaled_height2), Image.Resampling.LANCZOS)

        canvas1 = Image.new('RGB', (standard_width, standard_height), (0, 0, 0))
        canvas2 = Image.new('RGB', (standard_width, standard_height), (0, 0, 0))

        # Offset
        max_offset_x = int(standard_width * 0.15)
        max_offset_y = int(standard_height * 0.15)

        x_offset1 = random.randint(-scaled_width1 // 2, max_offset_x)
        y_offset1 = random.randint(-scaled_height1 // 2, max_offset_y)
        x_offset2 = random.randint(-max_offset_x, max_offset_x)
        y_offset2 = random.randint(-max_offset_y, max_offset_y)

        canvas1.paste(image, (x_offset1, y_offset1))
        canvas2.paste(second_image, (x_offset2, y_offset2))

        alpha = random.uniform(0.2, 0.8)
        img1_array = np.array(canvas1)
        img2_array = np.array(canvas2)

        mixed_array = (alpha * img1_array + (1 - alpha) * img2_array).astype(np.uint8)
        mixed_image = Image.fromarray(mixed_array)

        # Combine annotations from both images
        mixed_annotations = []

        for ann in annotations:
            transformed_ann = ann.copy()

            for corner in [('x1', 'y1'), ('x2', 'y2'), ('x3', 'y3'), ('x4', 'y4')]:
                x_key, y_key = corner

                scaled_x = ann[x_key] * first_scale_factor
                scaled_y = ann[y_key] * first_scale_factor

                final_x = scaled_x + x_offset1
                final_y = scaled_y + y_offset1

                if (final_x >= -50 and final_x <= standard_width + 50 and
                    final_y >= -50 and final_y <= standard_height + 50):
                    transformed_ann[x_key] = round(max(0, min(standard_width, final_x)))
                    transformed_ann[y_key] = round(max(0, min(standard_height, final_y)))
                else:
                    transformed_ann = None
                    break

            if transformed_ann is not None:
                mixed_annotations.append(transformed_ann)

        for ann in second_image_annotations:
            transformed_ann = ann.copy()

            for corner in [('x1', 'y1'), ('x2', 'y2'), ('x3', 'y3'), ('x4', 'y4')]:
                x_key, y_key = corner

                scaled_x = ann[x_key] * second_scale_factor
                scaled_y = ann[y_key] * second_scale_factor

                final_x = scaled_x + x_offset2
                final_y = scaled_y + y_offset2

                if (final_x >= -50 and final_x <= standard_width + 50 and
                    final_y >= -50 and final_y <= standard_height + 50):
                    transformed_ann[x_key] = round(max(0, min(standard_width, final_x)))
                    transformed_ann[y_key] = round(max(0, min(standard_height, final_y)))
                else:
                    transformed_ann = None
                    break

            if transformed_ann is not None:
                mixed_annotations.append(transformed_ann)

        return mixed_image, mixed_annotations


    #################################
    ### Mosaic transform function ###
    #################################
    def apply_mosaic_transform(self, image, annotations):
        second_image, filename = self.select_random_image()
        second_image_annotations = self.all_annotations[filename]
        third_image, filename = self.select_random_image()
        third_image_annotations = self.all_annotations[filename]
        fourth_image, filename = self.select_random_image()
        fourth_image_annotations = self.all_annotations[filename]

        standard_width = self.standard_size[0].get()
        standard_height = self.standard_size[1].get()
        quad_width = standard_width // 2
        quad_height = standard_height // 2

        mosaic_canvas = Image.new('RGB', (standard_width, standard_height), (0, 0, 0))

        images = [image, second_image, third_image, fourth_image]
        annotations_list = [annotations, second_image_annotations, third_image_annotations, fourth_image_annotations]
        quadrant_positions = [
            (0, 0), (quad_width, 0), (0, quad_height), (quad_width, quad_height)
        ]

        mosaic_annotations = []

        for i, (img, anns, (quad_x, quad_y)) in enumerate(zip(images, annotations_list, quadrant_positions)):
            orig_width, orig_height = img.size

            scale = random.uniform(1.1, 2.0)
            scaled_width = int(quad_width * scale)
            scaled_height = int(quad_height * scale)

            scale_factor_x = scaled_width / orig_width
            scale_factor_y = scaled_height / orig_height

            scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)

            max_offset = int(min(scaled_width - quad_width, scaled_height - quad_height) * 0.5)
            x_offset = random.randint(-max_offset, 0)
            y_offset = random.randint(-max_offset, 0)

            crop_box = (-x_offset, -y_offset, -x_offset + quad_width, -y_offset + quad_height)
            cropped_img = scaled_img.crop(crop_box)
            mosaic_canvas.paste(cropped_img, (quad_x, quad_y))

            for ann in anns:
                transformed_ann = ann.copy()

                original_points = []
                for corner in [('x1', 'y1'), ('x2', 'y2'), ('x3', 'y3'), ('x4', 'y4')]:
                    x_key, y_key = corner
                    scaled_x = ann[x_key] * scale_factor_x + x_offset
                    scaled_y = ann[y_key] * scale_factor_y + y_offset
                    final_x = scaled_x + quad_x
                    final_y = scaled_y + quad_y
                    original_points.append((final_x, final_y))

                    clamped_x = max(quad_x, min(quad_x + quad_width, final_x))
                    clamped_y = max(quad_y, min(quad_y + quad_height, final_y))
                    transformed_ann[x_key] = round(clamped_x)
                    transformed_ann[y_key] = round(clamped_y)

                overlap_percentage = self.calculate_polygon_overlap_percentage(
                    original_points, quad_x, quad_y, quad_width, quad_height
                )

                if overlap_percentage >= 0.35:
                    mosaic_annotations.append(transformed_ann)

        return mosaic_canvas, mosaic_annotations


    ##############################
    ### Generate test examples ###
    ##############################
    def generate_test_examples(self, test_image, test_annotations, output_path):
        """Generate test examples showing each transform"""

        test_dir = output_path / 'test_generation'
        test_image.save(test_dir / 'images' / 'original.jpg')
        all_transforms = {**self.multi_image_transforms, **self.single_image_transforms}

        annotations = {}
        names = []

        for transform_name, config in all_transforms.items():
            transformed_image, transformed_annotations = self.apply_single_transform(
                test_image, test_annotations, transform_name, config
            )

            safe_name = transform_name.lower().replace(' ', '_')
            image_path = Path(test_dir / 'images' /  f'{safe_name}.jpg')
            transformed_image.save(image_path)

            names.append(image_path)
            annotations[image_path.name] = transformed_annotations

        save_path = test_dir / 'labels'
        save_annotations_yolo(annotations, names, str(save_path))


    ####################################
    ### List of applyable transforms ###
    ####################################
    def apply_single_transform(self, image, annotations, transform_name, config):
        transformed_image = image.copy()
        transformed_annotations = [ann.copy() for ann in annotations]

        if transform_name == 'Horizontal Flip':
                transformed_image, transformed_annotations = self.apply_horizontal_flip(
                    transformed_image, transformed_annotations
                )

        elif transform_name == 'Vertical Flip':
            transformed_image, transformed_annotations = self.apply_vertical_flip(
                transformed_image, transformed_annotations
            )

        elif transform_name == 'Random Brightness':
            factor = 1.0 + random.uniform(-config['factor'].get(), config['factor'].get())
            enhancer = ImageEnhance.Brightness(transformed_image)
            transformed_image = enhancer.enhance(factor)

        elif transform_name == 'Random Contrast':
            factor = 1.0 + random.uniform(-config['factor'].get(), config['factor'].get())
            enhancer = ImageEnhance.Contrast(transformed_image)
            transformed_image = enhancer.enhance(factor)

        elif transform_name == 'Gaussian Blur':
            kernel_size = config['kernel'].get()
            transformed_image = transformed_image.filter(ImageFilter.GaussianBlur(kernel_size))

        elif transform_name == 'HSV':
            h_gain = random.uniform(-config['h_gain'].get(), config['h_gain'].get()) / 100
            s_gain = random.uniform(-config['s_gain'].get(), config['s_gain'].get()) / 100
            v_gain = random.uniform(-config['v_gain'].get(), config['v_gain'].get()) / 100
            transformed_image = self.apply_hsv_transform(transformed_image, h_gain, s_gain, v_gain)

        elif transform_name == 'Random Affine':
            scale = config['scale'].get()
            transformed_image, transformed_annotations = self.apply_affine_transform(
                transformed_image, transformed_annotations, scale
            )

        elif transform_name == 'Noise':
            strength = config['strength'].get()
            transformed_image = self.apply_noise(transformed_image, strength)

        elif transform_name == 'CutOut':
            size_ratio = config['size'].get()
            transformed_image = self.apply_cutout(transformed_image, size_ratio)

        elif transform_name == 'Random Rotate':
            degrees = random.uniform(-config['degrees'].get(), config['degrees'].get())
            if degrees < 0: degrees += 360
            transformed_image, transformed_annotations = self.apply_rotation(
                transformed_image, transformed_annotations, degrees
            )

        # Multi image transforms
        elif transform_name == "Mixup":
            scale = config['scale'].get()
            transformed_image, transformed_annotations = self.apply_mixup_transform(
                transformed_image, transformed_annotations, scale
            )
        elif transform_name == "Mosaic":
            transformed_image, transformed_annotations = self.apply_mosaic_transform(
                transformed_image, transformed_annotations
            )

        return transformed_image, transformed_annotations

    #################################
    ### Calculate overlap percent ###
    #################################
    def calculate_polygon_overlap_percentage(self, original_points, quad_x, quad_y, quad_width, quad_height):
        """
        Calculate what percentage of the annotation polygon overlaps with the quadrant.
        Takes original (unclamped) points and returns percentage as a float between 0.0 and 1.0
        """

        def polygon_area(points):
            """Calculate area using shoelace formula"""
            if len(points) < 3:
                return 0
            area = 0
            n = len(points)
            for i in range(n):
                j = (i + 1) % n
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            return abs(area) / 2.0

        def clip_polygon_to_rectangle(polygon, rect_min_x, rect_min_y, rect_max_x, rect_max_y):
            """Clip polygon to rectangle using Sutherland-Hodgman algorithm"""
            def is_inside(point, edge):
                x, y = point
                if edge == 'left':
                    return x >= rect_min_x
                elif edge == 'right':
                    return x <= rect_max_x
                elif edge == 'bottom':
                    return y >= rect_min_y
                elif edge == 'top':
                    return y <= rect_max_y

            def get_intersection(p1, p2, edge):
                x1, y1 = p1
                x2, y2 = p2
                if edge == 'left':
                    if x2 != x1:
                        y = y1 + (y2 - y1) * (rect_min_x - x1) / (x2 - x1)
                        return (rect_min_x, y)
                elif edge == 'right':
                    if x2 != x1:
                        y = y1 + (y2 - y1) * (rect_max_x - x1) / (x2 - x1)
                        return (rect_max_x, y)
                elif edge == 'bottom':
                    if y2 != y1:
                        x = x1 + (x2 - x1) * (rect_min_y - y1) / (y2 - y1)
                        return (x, rect_min_y)
                elif edge == 'top':
                    if y2 != y1:
                        x = x1 + (x2 - x1) * (rect_max_y - y1) / (y2 - y1)
                        return (x, rect_max_y)
                return None

            clipped = polygon[:]
            edges = ['left', 'right', 'bottom', 'top']

            for edge in edges:
                if not clipped:
                    break

                input_list = clipped[:]
                clipped = []

                if input_list:
                    s = input_list[-1]  # Last vertex

                    for e in input_list:
                        if is_inside(e, edge):
                            if not is_inside(s, edge):
                                intersection = get_intersection(s, e, edge)
                                if intersection:
                                    clipped.append(intersection)
                            clipped.append(e)
                        elif is_inside(s, edge):
                            intersection = get_intersection(s, e, edge)
                            if intersection:
                                clipped.append(intersection)
                        s = e

            return clipped

        # Calculate original polygon area
        original_area = polygon_area(original_points)
        if original_area == 0:
            return 0.0

        # Clip polygon to quadrant rectangle
        clipped_polygon = clip_polygon_to_rectangle(
            original_points,
            quad_x,
            quad_y,
            quad_x + quad_width,
            quad_y + quad_height
        )

        # Calculate clipped area
        clipped_area = polygon_area(clipped_polygon)

        # Return percentage
        return clipped_area / original_area
