# %%
import os

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import multiprocessing
from torch.utils.data import DataLoader
from torchinfo import summary
import pathlib
import random
import shutil
from torch import nn

# Setup data paths
# Create target directory path
target_dir_name = f"facemaskDataStore/facemaskwith_facemaskwithout_images"
print(f"Creating directory: '{target_dir_name}'")

# Setup the directory using target_dir_name
target_dir = pathlib.Path(target_dir_name)
# now we can make the directory
target_dir.mkdir(parents=True, exist_ok=True)

def walk_through_dir(dir_path):
    print(f"dir: {dir_path}")
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

walk_through_dir(target_dir)

# Setup train and testing data paths
train_dir = target_dir / "train"
test_dir = target_dir / "test"
print(f"train dir: {train_dir} and test dir: {test_dir}")

from PIL import Image
# Set seed for get random file
random.seed(99)
# 1. Get all image paths (* means "any combination")
image_path_list = list(target_dir.glob("*/*/*.png"))
# 2. Get random image path
random_image_path = random.choice(image_path_list)
# 3. Get image class from path name (the image class is the name of the directory where the image is stored)
image_class = random_image_path.parent.stem
# 4. Open image
img = Image.open(random_image_path)
# 5. Print metadata
print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class}")
print(f"Image height: {img.height}")
print(f"Image width: {img.width}")

import numpy as np
import matplotlib.pyplot as plt

# Turn the image into an array
img_as_array = np.asarray(img)
# Plot the image with matplotlib
plt.figure(figsize=(12, 8))
plt.imshow(img_as_array)
plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
plt.axis(False);

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

#this function for take random images from image path and transform the images
def plot_transformed_images(image_paths, transform, n=4, seed=9):
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")
            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()
# select 6 random images and transform
plot_transformed_images(image_path_list,
                        transform=data_transform,
                        n=6, seed=13)

# Create training transform with TrivialAugment
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=20), # set num_magnitude_bins to 20
    transforms.ToTensor()
])
# Create testing transform (no data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# define TinyVGG for creating a models
class TinyVGGForMask(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape)
        )
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
        # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion


cpuCount = os.cpu_count()
BATCH_SIZE = 16
# 1. turn image folders into Datasets
train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)
train_data_augmented, test_data_simple

# 2. Turn Datasets into DataLoader's
train_dataloader_augmented = DataLoader(train_data_augmented,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    num_workers=1)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=1)

train_dataloader_augmented, test_dataloader_simple

# 3.Create model_augmentation and send it to the target device
# Set random seeds
torch.manual_seed(42)
torch.cuda.manual_seed(42)# create first model using TinyVGG and set hidden units as 10 here
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_augmentation = TinyVGGForMask(
    input_shape=3,
    hidden_units=5, #5/10/20
    output_shape=len(train_data_augmented.classes)).to(device)
model_augmentation

from tqdm.auto import tqdm
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train()
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)
        # 1. Forward pass
        y_pred = model(X)
        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        # 3. Optimizer zero grad
        optimizer.zero_grad()
        # 4. Loss backward
        loss.backward()
        # 5. Optimizer step
        optimizer.step()
        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    # Turn on no_grad context manager
    with torch.no_grad():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
            # 1. Forward pass
            test_pred_logits = model(X)
            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
# 1. Take in various parameters required for training and test steps
def train(model: torch.nn.Module, train_dataloader: torch.utils.data.DataLoader, test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer, loss_fn: torch.nn.Module = nn.CrossEntropyLoss(), epochs: int = 5):
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,dataloader=train_dataloader, loss_fn=loss_fn, optimizer=optimizer)
        test_loss, test_acc = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )
        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    # 6. Return the filled results at the end of the epochs
    return results

# Setup loss function and optimizer for model_augmentation
loss_fn_augmentation = nn.CrossEntropyLoss()
optimizer_augmentation = torch.optim.Adam(params=model_augmentation.parameters(), lr=0.001)

# This function use for draw a graph with compare accuracy and loss in each model
from typing import Dict, List
def plot_loss_curves(results: Dict[str, List[float]]):
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

def main():
    from timeit import default_timer as timer
    import pandas as pd

    # Start the timer
    from timeit import default_timer as timer
    start_time = timer()

    # # Train model_augmentation
    model_augmentation_results = train(model=model_augmentation,
                                       train_dataloader=train_dataloader_augmented,
                                       test_dataloader=test_dataloader_simple,
                                       optimizer=optimizer_augmentation,
                                       loss_fn=loss_fn_augmentation,
                                       epochs=15)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time for augmentation: {end_time - start_time:.3f} seconds")
    plot_loss_curves(model_augmentation_results)
    plt.show()


    # Now we can do the prediction part
    import torchvision.io as tvio
    from pathlib import Path
    # Path to the validation folder
    validation_folder = r'H:\study\AI\project\Face_Mask_Dataset\Face_Mask_Dataset\Validation'
    # Create a list of image file paths in the validation folder
    image_paths_for_validation = list(Path(validation_folder).rglob('*.png'))
    for with_mask_image_for_validation in image_paths_for_validation:
        print(f"####################################################")
        print(f"Custom image: {with_mask_image_for_validation}")

        # Read in custom image
        tvio.read_image(str(with_mask_image_for_validation))
        # Load in custom image and convert the tensor values to float32
        custom_image = torchvision.io.read_image(str(with_mask_image_for_validation)).type(torch.float32)
        # Divide the image pixel values by 255 to get them between [0, 1]
        custom_image = custom_image / 255.
        # Plot custom image
        plt.imshow(custom_image.permute(1, 2,
                                        0))  # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
        plt.title(f"Image shape: {custom_image.shape}")
        plt.axis(False)
        # Create transform pipleine to resize image
        custom_image_transform = transforms.Compose([transforms.Resize((64, 64)), ])  # Transform target image
        custom_image_transformed = custom_image_transform(custom_image)
        # Print out original shape and new shape
        print(f"Original shape: {custom_image.shape}")
        print(f"New shape: {custom_image_transformed.shape}")
        # Plot custom image transform
        plt.imshow(custom_image_transformed.permute(1, 2,
                                                    0))  # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
        plt.title(f"Image shape: {custom_image_transformed.shape}")
        plt.axis(False)
        # plt.show()
        model_augmentation.eval()
        with torch.no_grad():
            # Make a prediction on image with an extra dimension
            custom_image_pred = model_augmentation(custom_image_transformed.unsqueeze(dim=0).to(device))
        custom_image_pred
        print(f"Prediction logits: {custom_image_pred}")

        custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
        print(f"Prediction probabilities: {custom_image_pred_probs}")

        # Convert prediction probabilities -> prediction labels
        custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
        print(f"Prediction label: {custom_image_pred_label}")

        # Find the predicted class
        class_names = ["WithMask", "WithoutMask"]
        custom_image_pred_class = class_names[
            custom_image_pred_label.cpu()]  # put pred label to CPU, otherwise will error
        custom_image_pred_class
        print(f"custom_image_pred_class : {custom_image_pred_class}")
if __name__ == '__main__':
    main()