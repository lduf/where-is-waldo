#!pip install detecto
#!pip install matplotlib
#!pip install torch
from detecto import core, utils, visualize
from detecto.utils import normalize_transform
from detecto.core import Model, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import torch

torch.cuda.empty_cache()


custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    # Note: all images with a size smaller than 800 will be scaled up in size
    transforms.Resize(200),
    transforms.ColorJitter(saturation=0.2),
    transforms.ToTensor(),  # required
    normalize_transform(),  # required
])
dataset = Dataset('../dataset/train/annotations','dataset/train/images', transform=custom_transforms)

loader = core.DataLoader(dataset, batch_size=2, shuffle=True)
model = Model(['waldo'])
losses = model.fit(loader, dataset, epochs=5, learning_rate=0.01, verbose=True)

# Visualize loss during training

plt.plot(losses)
plt.show()

# Save model

model.save('../models/model_weights.pth')

# Access underlying torchvision model for further control

torch_model = model.get_internal_model()
print(type(torch_model))

image = utils.read_image('dataset/images/waldo141.jpg')
labels, boxes, scores = model.predict(image)
predictions = model.predict_top(image)

print(labels, boxes, scores)
print(predictions)

# Visualize module's helper functions

visualize.show_labeled_image(image, boxes[:1], labels[:1])