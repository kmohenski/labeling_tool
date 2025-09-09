# Image Labeling and Augmentation tool

A simple GUI application for annotating images and generating augmented datasets for object detection training built with Python and Tkinter. The tool uses 4-corner polygon annotations and includes image transform options.

## Features and Capabilities

### 1. Image Annotation Interface

The labeling interface uses an intuitive way to annotate images with bounding boxes using a 4-corner polygon format. This allows annotations of rotated or skewed objects compared to standard rectangular bounding boxes to be displayed. Currently rotated and skewed bounding boxes can not be drawn.

Key annotation features:

- Click and drag to create bounding boxes
- Multiple object classes, color coded
- Annotation preview and editing
- Keyboard shortcuts for navigation (arrow keys, number keys for class selection)
- Right click to delete annotations

### 2. Data Augmentation and Transformation

The integrated transformer allows us to generate additional images to expand training datasets and improve model generalization. The augmentation pipeline supports single and multi image transformations. For more information read the README in [the src/ dir](src/)

### 3. Export Formats

Supported formats:

- [YOLO/YOLOv8](https://en.wikipedia.org/wiki/You_Only_Look_Once) OBB: Oriented bounding box format with class index, center coordinates, dimensions, and rotation angle
- [DOTA](https://captain-whu.github.io/DOTA/index.html): 4-corner coordinate format commonly used in aerial image datasets
- 4-corner JSON: Custom JSON format with exact coordinates for easier loading

## Installation and Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

or

```bash
pip install tkinter pillow numpy opencv-python pathlib
```

### Running the Application

```bash
python main.py
```

### Basic Workflow

1. Load Images: Select a folder containing your images
2. Add Classes: Define object classes or add custom classes during annotation
3. Annotate: Click and drag to create bounding boxes, select appropriate classes
4. Save: Export annotations in your preferred format
5. Augment (Optional): Use the transformer to generate additional training data

## File Structure

```bash
.
├── main.py                   # Application entry point
└── src/
    ├── labeler.py            # Main GUI labeling interface
    ├── transformer.py        # Data augmentation engine
    └── file_managment.py     # Annotation import/export handlers
```

### Core Components

[labeler.py](src/labeler.py):

- Canvas based image display
- Bounding box draw and delete
- Class management
- Annotation visualization

[transformer.py](src/labeler.py):

- Configurable image transformation
- Automatic annotation coordinate transformation
- Dataset splitting (train/validation/test)
- Test mode for transform visualization

[file_managment.py](src/file_managment.py):

- 3 different export formats
- Creates organized directory structure for test/train split

## Notes

### Custom Annotation Format

The tool uses a 4-corner format stored as:

```json
{
  "class": "object_name",
  "x1": top_left_x, "y1": top_left_y,
  "x2": top_right_x, "y2": top_right_y,
  "x3": bottom_right_x, "y3": bottom_right_y,
  "x4": bottom_left_x, "y4": bottom_left_y
}
```

## Possible improvements

Adding drawing of oriented bounding boxes.

I would like to randomize the order of applied transforms instead of using a preset order. [More info here.](src/README.md)

More transforms, more better.

More save formats, more better.

I don't know how I would save oriented bounding boxes in a rectangle format...
