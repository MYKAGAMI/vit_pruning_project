# download_teacher.py
import timm
import torch
import os

model_name = 'beitv2_base_patch16_224'
output_path = './teacher_model_beitv2.pth' 
print(f"Downloading a SOTA teacher model: '{model_name}' from timm...")

teacher_model = timm.create_model(
    model_name,
    pretrained=True,
    num_classes=1000
)

print(f"Saving state_dict to '{output_path}'...")
torch.save(teacher_model.state_dict(), output_path)
print("\nTeacher model checkpoint saved successfully!")
print(f"File path: {os.path.abspath(output_path)}")
print("You can now use this file for the --teacher_checkpoint argument.")