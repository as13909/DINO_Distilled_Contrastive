# Self-Supervised Distilled Model (DINO v3)

Single-file contrastive knowledge distillation pipeline with a student–teache 

Setup: dual-view data augmentation feeds a frozen or EMA-updated teacher and a trainable student, each built as a backbone (ViT via timm or tiny CNN) + MLP projection head. Training optimizes an InfoNCE contrastive loss between student and teacher projections (optionally symmetric), with mixed precision, gradient accumulation, EMA momentum teacher updates, and optional supervised CE. Includes 96×96 ViT compatibility, checkpointing, and k-NN evaluation on frozen student features.


## Project Structure

```
Final_Submission/
├── DINO_distillled_contrastive.py          # Main script containing model, training, etc...
├── .gitignore.py                           # The gitignore
├── requirements.txt                        # Python dependencies
|
├── testset_1/                              # Evaluation code for test set 1
│   ├── create_submission_knn6.py           # Creates submission file to submit on kaggle
│   ├── DINO_distillled_contrastive.py      # Same as original DINO makes it easier for Py to find
│   ├── prepare_cub200_for_kaggle.py        # Gets dataset for test set 1
│   ├── submission1.csv                     # Submission csv for test set 1
|   └── kd_out/                             # Sub folder which contains final / best .pth file
|       └── student_final.pth               # This was our best .pth file from running DINO_Distilled_Contrastive
|
├── testset_2/                              # Evaluation code for test set 2
│   ├── create_submission_knn6.py           # Creates submission file to submit on kaggle
│   ├── DINO_distillled_contrastive.py      # Same as original DINO makes it easier for Py to find
│   ├── prepare_cub200_for_kaggle.py        # Gets dataset for test set 2
│   ├── submission2.csv                     # Submission csv for test set 2
|   └── kd_out/                             # Sub folder which contains final / best .pth file
|       └── student_final.pth               # This was our best .pth file from running DINO_Distilled_Contrastive
|
├── testset_3/                              # Evaluation code for test set 3
│   ├── create_submission_knn6.py           # Creates submission file to submit on kaggle
│   ├── DINO_distillled_contrastive.py      # Same as original DINO makes it easier for Py to find
│   ├── prepare_cub200_for_kaggle.py        # Gets dataset for test set 3
|   └── kd_out/                             # Sub folder which contains final / best .pth file
|       └── student_final.pth               # This was our best .pth file from running DINO_Distilled_Contrastive
|
└── README.md                               # This file

```

## Features

* **Contrastive Knowledge Distillation (CKD)**: Student–teacher contrastive training using InfoNCE, with optional **momentum (EMA) teacher** in BYOL/DINO style or frozen teacher (traditional KD)
* **Model Architectures**: Supports **ViT backbones via timm** (e.g., `vit_base_patch16_96`, `vit_small_patch16_96`) and a **TinyConv** fallback; backbone + MLP projection head
* **Training & Evaluation Methods**:
  * **k-NN evaluation** on frozen student features
  * **Linear Probe**
* **Device Support**: CUDA and CPU (automatic selection)
* **Training Utilities**: Mixed precision (AMP), gradient accumulation, gradient clipping, warmup + cosine LR schedule, checkpointing & resume
* **Configuration & Interface**: Fully configurable via **CLI arguments** (single-file, no external config dependency)

## Installation

1. Create a virtual environment (Python 3.10 recommended for `learn2learn`):
```bash
python3.10 -m venv venv_py310
source venv_py310/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: For dataset preparation with `learn2learn`, Python 3.10 is required to avoid compilation errors.

## Data

1. stealthtechnologies/birds-images-dataset:
```bash
kagglehub.dataset_download("stealthtechnologies/birds-images-dataset")
```
2. sasha/birdsnap
```bash
birdsnap = load_dataset("sasha/birdsnap")
```
3. tsbpp/fall2025_deeplearning
```bash
data = load_dataset("tsbpp/fall2025_deeplearning")
```

## Usage

### Training

1. Grab the Data as described in the Data section above. When running the next command specify the folder with your `.jpg`s in it. 
2. Run the `DINO_Distilled_Contrastive.py`:
```bash
python DINO_distilled_contrastive.py \
    --data_dir dataset_cache/unzipped_ims/pretrain \
    --use_timm_teacher  --teacher_model vit_base_patch16_224 --teacher_pretrained  \
    --use_timm_student --student_model vit_small_patch16_224 \
    --contrastive --contrastive_weight 1.0 \ 
    --proj_dim 1024 \
    --epochs 251 \  
    --batch_size 128 \
    --amp \
    --teacher_momentum 0.975 \
    --out_dir kd_out_contrast \
    --lr .0005 \    
```
If you would like to resume from a specific checkpoint specify: `--resume_ckpt PATH_TO_YOUR_pth/.pth`

3. Switch over to any of the testset folders `cd testset_1`
4. Grab the testset dataset. 
For test set 1:
```bash
python prepare_cub200_for_kaggle.py --download_dir ./raw_data --output_dir ./data
```
For test set 2:
```bash
python prepare_miniimagenet_for_kaggle.py --download_dir ./raw_data --output_dir ./data
```
For test set 3:
```bash
python prepare_sun397_for_kaggle.py --download_dir ./raw_data --output_dir ./data
```
5. Put whichever .pth file you're using in the respective `kd_out` folder
6. Run the `create_submission_knn6.py` files in each respective folder to get the `submission.csv` files.
```bash
python create_submission_knn6.py --data_dir ./data --output submission.csv --k 25 --ckpt kd_out/student_final.pth --use_timm_student   
```

## Key Features of Evaluation Script

- **Best Model Selection**: Automatically saves and uses the linear probe model with highest validation accuracy
- **Early Stopping**: Prevents overfitting with configurable patience
- **Device Optimization**: Automatically selects best available device (CUDA > MPS > CPU)
- **macOS Compatibility**: Automatically sets `num_workers=0` on macOS to prevent multiprocessing issues

## Requirements

See `requirements.txt` for full list. Key dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- scikit-learn
- pandas
- numpy
- tqdm
- kaggle

## Notes

- Large data directories and model checkpoints are excluded from git (see `.gitignore`)
- Virtual environments should not be committed
- Submission CSV files are generated outputs and excluded from version control