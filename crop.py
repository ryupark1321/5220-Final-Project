import torch
import torchvision.transforms as transforms
from PIL import Image
import tqdm 
import os

scratch = os.environ['SCRATCH']
dir1 = os.path.join(scratch,'imagenette/imagenette2/train/')
dir2 = os.path.join(scratch,'imagenette/imagenette2/val/')
directories = [dir1,dir2]
for path_prefix in directories:
  for folder in tqdm.tqdm(os.listdir(path_prefix)):
    current_directory_files = os.listdir(os.path.join(path_prefix,folder))
    for filename in current_directory_files:
    # Read the image, originally 500,375
      img = Image.open(os.path.join(path_prefix,folder,filename))

      resizefn=transforms.Resize(256)
    # define a transform to crop the image at center
      cropfn = transforms.CenterCrop(224)

    # crop the image using above defined transform
      img = resizefn(img)
      img = cropfn(img)

    # save the image
      img.save(os.path.join(path_prefix,folder,filename))
