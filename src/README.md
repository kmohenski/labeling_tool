# Additional Info and examples

## Transform Configuration

Each transform can be independently enabled with a customizable probability value.

## Dataset Generation Methodology

### 1. Standardization

All images, and their annotations, are resized to the selected resolution.

### 2. Generation Multiplier

Selectable number of additional images to be made (0-500%, can be set to more in [transformer.py](transformer.py)) with transform selection to create more diverse training samples.

### 3. Dataset Splitting

Automated train/validation/test splitting, ensuring proper data distribution across splits.

### 4. Quality Control

• Test mode generation for visual verification of all transform effects

## Notes

### Transform Pipeline

Transforms are applied sequentially with early termination on failure. The pipeline prioritizes multi image transforms before single image transforms.

<!-- 'Horizontal Flip':
'Vertical Flip': {'
'Random Rotate': {'
'HSV': {'enabled':
'Random Brightness'
'Random Contrast':
'Gaussian Blur': {'
'Random Affine': {'
'Noise': {'enabled'
'CutOut': {'enabled -->

## Examples

### Original

<p align="center">
  <img src="../test_generation/images/original.jpg" align="middle"/>
</p>

### Horizontal flip

<p align="center">
  <img src="../test_generation/images/horizontal_flip.jpg" align="middle"/>
</p>

### Vertical flip

<p align="center">
  <img src="../test_generation/images/vertical_flip.jpg" align="middle"/>
</p>

### Random rotate

<p align="center">
  <img src="../test_generation/images/random_rotate.jpg" align="middle"/>
</p>

### HSV

<p align="center">
  <img src="../test_generation/images/hsv.jpg" align="middle"/>
</p>

### Random brightness

<p align="center">
  <img src="../test_generation/images/horizontal_flip.jpg" align="middle"/>
</p>

### Random Contrast

<p align="center">
  <img src="../test_generation/images/random_contrast.jpg" align="middle"/>
</p>

### Gaussian Blur

<p align="center">
  <img src="../test_generation/images/gaussian_blur.jpg" align="middle"/>
</p>

### Random Affine

<p align="center">
  <img src="../test_generation/images/random_affine.jpg" align="middle"/>
</p>

### Noise

<p align="center">
  <img src="../test_generation/images/noise.jpg" align="middle"/>
</p>

### CutOut

<p align="center">
  <img src="../test_generation/images/cutout.jpg" align="middle"/>
</p>

### Mixup

<p align="center">
  <img src="../test_generation/images/mixup.jpg" align="middle"/>
</p>

### Mosaic

<p align="center">
  <img src="../test_generation/images/mosaic.jpg" align="middle"/>
</p>
