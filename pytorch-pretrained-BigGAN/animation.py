import numpy as np
import os
import torch
from PIL import Image
import imageio
from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample

# Load pre-trained BigGAN model
model = BigGAN.from_pretrained('biggan-deep-256')

# Parameters
COLS = 30
ROWS = 1
truncation = 0.5

# Create class and noise vectors for interpolation
class_vector = np.zeros((ROWS * COLS, 1000), dtype=np.float32)
noise_vector = np.zeros((ROWS * COLS, 128), dtype=np.float32)

# Interpolate between start and end classes
for j in range(ROWS):
    # Randomly select start and end classes for morph
    # CATEGORY LIST HERE: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    class_vector[j * COLS, np.random.randint(151, 281)] = 1  # Start class (dogs)
    class_vector[(j + 1) * COLS - 1, np.random.randint(281, 294)] = 1  # End class (cats)
    
    #class_vector[j * COLS, 76] = 1  
    #class_vector[(j + 1) * COLS - 1, 118] = 1 

    # Interpolate between start and end classes
    step = class_vector[(j + 1) * COLS - 1] - class_vector[j * COLS]
    for k in range(1, COLS - 1):
        class_vector[j * COLS + k] = class_vector[j * COLS] + (k / (COLS - 1)) * step
    
    # Generate noise vectors for smooth transition
    noise_vector[j * COLS] = truncated_noise_sample(truncation=truncation, batch_size=1)
    noise_vector[(j + 1) * COLS - 1] = truncated_noise_sample(truncation=truncation, batch_size=1)
    step = noise_vector[(j + 1) * COLS - 1] - noise_vector[j * COLS]
    for k in range(1, COLS - 1):
        noise_vector[j * COLS + k] = noise_vector[j * COLS] + (k / (COLS - 1)) * step

# Convert to PyTorch tensors and move to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
noise_vector = torch.from_numpy(noise_vector).to(device)
class_vector = torch.from_numpy(class_vector).to(device)
model.to(device)

# Generate images with BigGAN
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)
output = output.to('cpu').numpy().transpose(0, 2, 3, 1)

# Create directory for output if it doesn't exist
os.makedirs('outputs', exist_ok=True)

# Save images as a video
video_path = 'outputs/latent_walk.mp4'
with imageio.get_writer(video_path, mode='I', fps=4) as writer:
    for i in range(ROWS * COLS):
        img = ((output[i] + 1) / 2 * 255).astype(np.uint8)  # Convert to image format
        writer.append_data(img)
        
print(f"Video saved to {video_path}")
