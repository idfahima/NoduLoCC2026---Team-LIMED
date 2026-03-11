**Reproduction guide**  
   
 **1. Clone and install**  
   
 git clone https://github.com/idfahima/NoduLoCC2026---Team-LIMED.git  
   
  cd Nodulo  
   
  python -m venv .venv  
   
  source .venv/bin/activate  
   
  pip install --upgrade pip  
   
  pip install -r requirements.txt  
   
  pip install -e .  
   
    
 **2. Data layout**  
   
 Place the challenge data at the repository root:  
   
 classification_labels.csv  
   
  localization_labels.csv  
   
  nih_filtered_images/  
   
  lidc_png_16_bit/  
   
    
   
 If the data lives elsewhere, set data.root_dir in configs/default.yaml.  
   
 **3. Download pre-trained weights**  
   
 Download the two soup checkpoints from the GitHub release and place them at the expected paths:  
   
 mkdir -p outputs/soups  
   
  wget https://github.com/idfahima/NoduLoCC2026---Team-LIMED/releases/download/weights/classifier_soup.pt -O outputs/soups/classifier_soup.pt  
   
  wget https://github.com/idfahima/NoduLoCC2026---Team-LIMED/releases/download/weights/localizer_soup.pt  -O outputs/soups/localizer_soup.pt  
   
    
   
 **4. (Optional) Re-train from scratch**  
   
 python -m nodulo.scripts.train --config configs/default.yaml  
   
    
   
 Checkpoints are written to outputs/. The two files needed for inference are:  
   
 outputs/soups/classifier_soup.pt  
   
  outputs/soups/localizer_soup.pt  
   
    
   
 **5. Run inference**  
   
 python -m nodulo.scripts.infer \  
   
    --config configs/default.yaml \  
   
    --input-dir /path/to/test_images \  
   
    --output-dir /path/to/predictions \  
   
    --classifier-checkpoint outputs/soups/classifier_soup.pt \  
   
    --localizer-checkpoint outputs/soups/localizer_soup.pt  
   
    
   
 Output files:  
- classification_test_results.csv — columns file_name, label, confidence  
- localization_test_results.csv — columns file_name, x, y, confidence  
 **6. Docker**  
   
 docker build -t nodulo:latest .  
   
  docker run --rm --gpus all \  
   
    -v $(pwd):/workspace \  
   
    nodulo:latest \  
   
    --config /workspace/configs/default.yaml \  
   
    --input-dir /workspace/test_images \  
   
    --output-dir /workspace/predictions \  
   
    --classifier-checkpoint /workspace/outputs/soups/classifier_soup.pt \  
   
    --localizer-checkpoint /workspace/outputs/soups/localizer_soup.pt  
   
    
