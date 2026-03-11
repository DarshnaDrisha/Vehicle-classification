Vehicle Classification Project
This repository contains a complete vehicle classification pipeline using a fine-tuned ResNet-18 model on a 12-class imbalanced dataset (auto-rickshaw, bicycle, bus, car, e-rickshaw, mini-bus, mini-truck, motorcycle, rickshaw, tractor, truck, van). The final model achieves ~84-85% validation accuracy and weighted F1 of 0.84, with ONNX export for deployment.
​

Key improvements include data cleaning (duplicates/blur removal), class-weighted loss, dropout, two-phase fine-tuning, and augmentations to handle imbalance and overfitting.
​

Project Structure
text
models/vehicledatasetvehicletest4/
├── bestvalloss.pt          # Best PyTorch model checkpoint
├── vehicletest.onnx        # Exported ONNX model
├── classes.txt             # Class names mapping
├── classificationreport.txt # Final metrics
├── confusionmatrix.png     # Confusion matrix plot
└── learningcurves.png      # Training curves

analysisoutputs/            # Data analysis CSVs (classcounts.csv, etc.)

scripts/                    # All Python scripts
├── dataanalysiscleaning.py
├── isolateleakage.py
├── datapreprocessing.py
├── newdataprocessing.py
├── trainmodel.py          # Baseline
├── trainmodel2.py         # Extended training
├── trainmodel3.py
├── trainmodelfinal.py     # Final run
├── exportonnx.py
└── verifymodel.py

runs/pred/                  # Verification predictions
Requirements
Python 3.8+

PyTorch (with torchvision, torchaudio)

ONNX, ONNXRuntime

OpenCV, Pillow, NumPy, Pandas, Matplotlib, Scikit-learn

Tqdm for progress bars

Install via conda/pip:

text
pip install torch torchvision torchaudio opencv-python pillow numpy pandas matplotlib scikit-learn tqdm onnx onnxruntime
No specific CUDA version noted; runs on CPU/GPU.
​

Dataset
3,364 images (2,719 train, 645 val) across 12 classes.

Heavily imbalanced (e.g., trucks/vans dominant; mini-bus/tractor rare).

Images resized to 180x180; mixed JPG/PNG formats.

Place raw dataset in expected location (e.g., dataset/raw/); scripts handle cleaning/preprocessing.
​

Setup and Data Preparation
Run from project root:

Analyze and clean data:

text
python scripts/dataanalysiscleaning.py
python scripts/isolateleakage.py
Preprocess (choose one based on training script):

text
python scripts/datapreprocessing.py    # For trainmodel.py/trainmodel2.py
# OR
python scripts/newdataprocessing.py    # For trainmodel3.py/trainmodelfinal.py (recommended)
Outputs processed datasets and classes.txt.
​

Training
Reproduce final results (~85% val acc, F1 0.84):

text
python scripts/trainmodelfinal.py
Two-phase: Freeze backbone 5 epochs (LR 1e-4), then unfreeze (LR 1e-5).

AdamW, weight decay 3e-4, ReduceLROnPlateau, early stopping (patience 7).

Batch 32, weighted CE loss (inverse freq, max weight 10.0), dropout 0.5.

Saves bestvalloss.pt, history, report, plots to models/vehicledatasetvehicletest4/.
​

Other variants:

text
python scripts/trainmodel.py     # Baseline 20 epochs
python scripts/trainmodel2.py    # 50 epochs early stop
python scripts/trainmodel3.py    # Weighted loss prototype
Export and Verification
Export to ONNX:

text
python scripts/exportonnx.py -m models/vehicledatasetvehicletest4/bestvalloss.pt -o models/vehicledatasetvehicletest4/vehicletest.onnx
Verify (place autorickshaw.png, car.png, van.png in root):

text
python scripts/verifymodel.py -m models/vehicledatasetvehicletest4/vehicletest.onnx -c models/vehicledatasetvehicletest4/classes.txt -s 180
Expected: 2/3 correct (auto-rickshaw 0.37, car 0.99; van→truck 0.68).
​

Results Summary
Metric	Value
Metric	Value
Val Accuracy	0.84-0.85
Weighted F1	0.84
Top Classes (F1)	Motorcycle 0.96, Bicycle 0.95, Tractor 1.00
Weak Classes	Mini-bus 0.50
Common Confusions	Truck↔Bus/Van/Mini-truck, Auto-rickshaw↔Truck 
​
Confusion matrix and curves in models/.../. Model excels on distinct shapes; struggles with similar large vehicles.
​

Notes
Input size fixed at 180x180 for export.

Run on GPU for speed (e.g., Kaggle/Colab).

Future: More data for rare classes, advanced augs (MixUp), ensemble.
