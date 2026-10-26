# KPK SafeVision — Fire and Smoke Detection Using YOLO11n

**A deep learning-based computer vision system for detecting and localizing fire and smoke in images and video streams.**

KPK SafeVision is an object-detection project built with **YOLO11n and the Ultralytics framework**. It uses the D-Fire dataset to identify fire and smoke, draw bounding boxes around detected objects, and report prediction confidence scores. The project demonstrates a complete workflow from dataset preparation and model training to evaluation and practical inference.

## Key Features

- **Fire and smoke detection:** Identifies two object classes—Fire and Smoke.
- **YOLO11n:** Uses a lightweight, pretrained object-detection model fine-tuned on D-Fire.
- **Dataset validation:** Checks image-label correspondence, annotation statistics, and dataset split overlap.
- **Model evaluation:** Records precision, recall, mAP metrics, training curves, and a confusion matrix.
- **Image inference:** Detects fire and smoke in uploaded images and saves annotated predictions.
- **Video inference:** Processes video files and summarizes detected fire and smoke instances.
- **Webcam experiments:** Includes webcam-based detection examples for real-time experimentation.
- **Model persistence:** Saves the trained checkpoint to Google Drive for subsequent inference.

## Technology Stack

- Python
- PyTorch
- Ultralytics YOLO11n
- OpenCV
- NumPy and Pandas
- Matplotlib
- Google Colab and Google Drive

## Dataset

The project uses the **D-Fire dataset**, containing 21,527 images with bounding-box annotations for fire and smoke.

| Split | Images |
|---|---:|
| Training | 15,498 |
| Validation | 1,723 |
| Testing | 4,306 |
| **Total** | **21,527** |

The original training images were divided into training and validation subsets using a fixed random seed. The original test split was retained separately. Image-label pairs and overlap between splits were checked during preparation.

**Class mapping**

| Class ID | Class name |
|---:|---|
| 0 | Smoke |
| 1 | Fire |

## Model Training

The pretrained `yolo11n.pt` model was fine-tuned using the following configuration:

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Initialization | Pretrained weights |
| Framework | Ultralytics |
| Epochs | 50 |
| Input image size | 640 × 640 |
| Batch size | 16 |
| Optimizer | Auto-selected by Ultralytics |
| Early-stopping patience | 10 epochs |
| Training environment | Google Colab |
| GPU | NVIDIA Tesla T4 |

The best checkpoint was saved as `yolo11n_fire_smoke_baseline.pt`.

## Validation Results

The saved best checkpoint achieved the following results on the validation set:

| Metric | Overall | Smoke | Fire |
|---|---:|---:|---:|
| Precision | 76.6% | 82.7% | 70.6% |
| Recall | 71.2% | 78.4% | 64.0% |
| mAP@50 | 76.8% | 83.8% | 69.8% |
| mAP@50–95 | 44.6% | 52.4% | 36.7% |

These figures reflect validation performance, not independent test-set results. The stronger smoke-detection scores also indicate that fire detection could benefit from further investigation and optimization.

## Getting Started

### 1. Open the notebook

Open `Safe_City_project.ipynb` in Google Colab, or upload the notebook to your own Colab workspace.

### 2. Install the dependencies

```bash
pip install -U ultralytics opencv-python numpy pandas matplotlib pyyaml
```

### 3. Prepare the dataset

The notebook downloads D-Fire through the Kaggle API, extracts the dataset, checks annotations, creates the training/validation/test structure, and generates the YOLO dataset configuration.

Configure your Kaggle API credentials in Colab before running the dataset-download cells.

### 4. Train the model

Run the training cells to fine-tune YOLO11n. Training uses a Tesla T4 GPU when available; adjust the device configuration if you use a different environment.

### 5. Run inference

Load the saved checkpoint and execute the relevant notebook section to test an image, process a video, or experiment with webcam detection.

**Model checkpoint:** `models/yolo11n_fire_smoke_baseline.pt`

The notebook's paths are configured for Google Colab and Google Drive. Update them if running elsewhere.

## Project Workflow

1. Dataset acquisition and extraction.
2. Image and annotation verification.
3. Training, validation, and test split preparation.
4. Pretrained YOLO11n fine-tuning.
5. Validation and metric visualization.
6. Checkpoint saving and reuse.
7. Image, video, and webcam inference experiments.

## Future Improvements

- Evaluate the model on the held-out test set.
- Improve fire detection recall and reduce false positives.
- Test robustness under different lighting, weather, and camera conditions.
- Measure inference latency and performance on target hardware.
- Integrate detections with an alerting interface or monitoring dashboard.

## Limitations

KPK SafeVision is a prototype for computer vision experimentation. Its predictions may contain false positives or miss actual hazards. The notebook demonstrates detection workflows but does not establish a certified fire-alarm system or guarantee real-world safety performance.

## Acknowledgements

This project uses the D-Fire dataset and the Ultralytics YOLO framework. Please consult their respective documentation and licensing terms before redistribution or commercial deployment.
