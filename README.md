# NGC Coin Recognition and Value Counter

A real-time Philippine New Generation Currency (NGC) coin recognition and value counter using OpenCV and MobileNetV2 CNN.

## Approaches

- Option A: Traditional Computer Vision using Hough Circle Transform, coin diameter, and color verification.
- Option B: CNN-based recognition using Hough Circle detection and MobileNetV2 classification.

## Requirements

- Python 3.11
- OpenCV
- NumPy
- TensorFlow
- Matplotlib
- scikit-learn

Install dependencies:

pip install opencv-python numpy tensorflow matplotlib scikit-learn

## Project Files

### capture_dataset.py

Captures coin images using the webcam.

Run:

python capture_dataset.py

Images are saved in:

dataset/raw/

Classes:

1_piso
5_piso
10_piso
20_piso
25_centavo

### split_dataset.py

Splits the dataset into:

- 70% training
- 15% validation
- 15% testing

Run:

python split_dataset.py

### train_classifier.py

Trains the MobileNetV2 CNN classifier.

The trained model is saved as:

models/coin_classifier.keras

### finetune_classifier.py

Fine-tunes selected MobileNetV2 layers.

The fine-tuned model is saved as:

models/coin_classifier_finetuned.keras

### evaluate_classifier.py

Evaluates the CNN models and generates:

- Classification report
- Confusion matrix
- Training/validation accuracy curves
- Training/validation loss curves

Results are saved in:

results/Baseline/
results/FineTuned/

## Option A

### option_a.py

Uses traditional computer vision for coin recognition.

Pipeline:

Webcam
-> Grayscale + Blur
-> Hough Circle Detection
-> Size Calibration
-> Color Verification
-> Tracking
-> Counting
-> Total Value

Run:

python option_a.py

## Option B

### cnn_hybrid_final.py

Uses Hough Circle detection together with MobileNetV2 CNN classification.

Pipeline:

Webcam
-> Hough Circle Detection
-> Size Validation
-> CNN Classification
-> Tracking
-> Temporal Filtering
-> Counting
-> Total Value

Run:

python cnn_hybrid_final.py

Press Q to exit.

## CNN Results

### Baseline

Test Accuracy: 94.67%

10-Piso: Precision 90.62%, Recall 96.67%, F1-score 93.55%
1-Piso: Precision 100.00%, Recall 86.67%, F1-score 92.86%
20-Piso: Precision 100.00%, Recall 100.00%, F1-score 100.00%
25-Centavo: Precision 93.75%, Recall 100.00%, F1-score 96.77%
5-Piso: Precision 90.00%, Recall 90.00%, F1-score 90.00%

Overall Accuracy: 94.67%
Test Samples: 150

### Fine-Tuned

Test Accuracy: 94.00%

The baseline model performed slightly better than the fine-tuned model on the test dataset, so the baseline model is used by cnn_hybrid_final.py.

## Output

The system displays:

- Detected coin denomination
- Number of coins per denomination
- Total monetary value
- Real-time FPS

Generated evaluation files include:

results/Baseline/training_curves.png
results/Baseline/classification_report.txt
results/Baseline/confusion_matrix.png

results/FineTuned/training_curves.png
results/FineTuned/classification_report.txt
results/FineTuned/confusion_matrix.png

## Supported Coins

- 25-Centavo - PHP 0.25
- 1-Piso - PHP 1.00
- 5-Piso - PHP 5.00
- 10-Piso - PHP 10.00
- 20-Piso - PHP 20.00