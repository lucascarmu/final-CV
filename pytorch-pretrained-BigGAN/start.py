import os
import torch
from pytorch_pretrained_biggan import BigGAN, one_hot_from_names, truncated_noise_sample, save_as_images


# Cargar el modelo preentrenado
model = BigGAN.from_pretrained('biggan-deep-256')

# Preparar las entradas
truncation = 0.6
list_of_class = ['soap bubble', 'coffee', 'mushroom', 'French bulldog', 'scorpion', 'tarantula', 'pizza']
class_vector = one_hot_from_names(list_of_class, batch_size=len(list_of_class))
noise_vector = truncated_noise_sample(truncation=truncation, batch_size=len(list_of_class))

# Convertir a tensores
noise_vector = torch.from_numpy(noise_vector).to('cuda')
class_vector = torch.from_numpy(class_vector).to('cuda')
model.to('cuda')

# Generar las imágenes
with torch.no_grad():
    output = model(noise_vector, class_vector, truncation)

# Convertir la salida a CPU
output = output.to('cpu')

# Crear el directorio 'outputs' si no existe
os.makedirs('outputs', exist_ok=True)

# Guardar las imágenes en el directorio 'outputs'
save_as_images(output, file_name='outputs/output')

print("Imágenes generadas y guardadas en el directorio 'outputs'")

