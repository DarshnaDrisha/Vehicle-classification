# Vehicle Classification Project

This repository contains a complete vehicle classification pipeline using a fine-tuned ResNet-18 model on a 12-class imbalanced dataset (auto-rickshaw, bicycle, bus, car, e-rickshaw, mini-bus, mini-truck, motorcycle, rickshaw, tractor, truck, van). The final model achieves ~84-85% validation accuracy and weighted F1 of 0.84, with ONNX export for deployment.
The pipeline includes:
1. Data analysis and cleaning
2. Data preprocessing and augmentation
3. Two-phase transfer learning (freeze → fine-tune)
4. Model export to ONNX
5. ONNX verification script
## Requirements
Python 3.8+

pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn onnx onnxruntime opencv-python tqdm imagehash

- If using GPU, install PyTorch with CUDA support from
https://pytorch.org

## Dataset
1. 3,364 images (2,719 train, 645 val) across 12 classes
2. Heavily imbalanced: trucks and vans are dominant; mini-bus/tractor rare.
3. Input size: 180x180 (fixed for ONNX export)
4. Place raw dataset in expected location; scripts handle cleaning and preprocessing

## Pipeline
1. Data analysis and Cleaning
Run the analysis script to inspect dataset quality.
- python scripts/data_analysis_cleaning.py
2. Data Preprocessing
Prepare dataloaders and generate class metadata.
- python scripts/data_preprocessing.py
- python scripts/new_data_processing.py

   Expected output:

- Train Batch Images Shape: torch.Size([32, 3, 180, 180])
- Train Batch Labels Shape: torch.Size([32])

   This step also generates: classes.txt, which stores the class order used by the model.

3. Train the Model

Training pipeline:

Phase 1:
  Backbone frozen, 
  Train classifier head

Phase 2:
  Unfreeze backbone,
  Fine-tune entire network.
  
Train the classifier using transfer learning.

-python train_model.py

-python train_model2.py

-python train_model3.py

-python train_model_final.py

  
All outputs are saved in: "/Submission/analysis_outputs" , "Submission/models"

4. Export Model to ONNX:
   
Convert the trained PyTorch model into ONNX format.


python scripts/export_onnx.py -m "models/vehicle_dataset/vehicle_test_weighted_loss1/best_val_loss.pt" -o "models/vehicle_dataset/vehicle_test_weighted_loss1/vehicle_test.onnx"

This produces:

vehicle_test.onnx

5. Verify ONNX Model
   
Run the verification script to test the ONNX model.

python scripts/verify_model.py -m "models/vehicle_dataset/vehicle_test_weighted_loss1/vehicle_test.onnx" -c "models/vehicle_dataset/vehicle_test_weighted_loss1/classes.txt" -s 180

Expected output:

input shape [1, 3, 180, 180]

output shape [1, 12]

auto-rickshaw 0.37,

car 0.99,

truck 0.68

Prediction images are saved in:  runs/pred/

# Results:
| Metric       | Value                                                 |
| ------------ | ----------------------------------------------------- |
| Val Accuracy | 0.84-0.85                                             |
| Weighted F1  | 0.84                                                  |
| Top Classes  | Motorcycle 0.96, Bicycle 0.95, Tractor 1.00           |
| Weak Classes | Mini-bus 0.50                                         |
| Confusions   | Truck↔Bus/Van/Mini-truck, Auto-rickshaw↔Truck         |

Final result saved in : "Submission/models/vehicle_test_weighted_loss1"




 
