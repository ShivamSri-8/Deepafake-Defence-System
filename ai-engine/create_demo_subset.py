import os
import shutil
from pathlib import Path

def create_demo_subset():
    base = Path("data")
    src_real = base / "images_small" / "real"
    src_fake = base / "images_small" / "fake"
    
    dest = base / "demo_subset"
    train_real = dest / "train" / "real"
    train_fake = dest / "train" / "fake"
    val_real = dest / "val" / "real"
    val_fake = dest / "val" / "fake"
    
    for d in [train_real, train_fake, val_real, val_fake]:
        d.mkdir(parents=True, exist_ok=True)
        
    real_files = sorted(list(src_real.glob("*.jpg")))[:15]
    fake_files = sorted(list(src_fake.glob("*.jpg")))[:15]
    
    for i in range(10):
        shutil.copy(real_files[i], train_real)
        shutil.copy(fake_files[i], train_fake)
        
    for i in range(10, 15):
        shutil.copy(real_files[i], val_real)
        shutil.copy(fake_files[i], val_fake)
        
    print("Demo subset created successfully!")

if __name__ == "__main__":
    create_demo_subset()
