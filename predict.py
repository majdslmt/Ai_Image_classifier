import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
from get_input_args import get_input_args, check_device
import gc
from train import load_data

def main():
    input_args = get_input_args()
    train_dataloader, valid_dataloader, test_dataloader, image_datasets = load_data(input_args.data_dir)
    model = load_checkpoint(input_args.input, image_datasets)
    device = check_device(input_args.gpu)
    if device == "cuda":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") == "cpu"
        if device == "cpu":
            print("cuda is not available")
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    img = process_image(input_args.image_path)
    imshow(img)
    probs , classes = predict(img, model, device, cat_to_name, topk=5)
    print(probs)
    print(classes)
    show_top_5(input_args.image_path, model,device, cat_to_name)

#load checkpoint
def load_checkpoint(filepath, image_datasets):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = models.densenet161(pretrained=True)
    model.classifier = checkpoint['model.classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = image_datasets[0].class_to_idx
    return model

#process_image
def process_image(image):
    pil_image = Image.open(image, 'r')

    pil_image.thumbnail((256, 256))
    pil_image = pil_image.crop((16, 16, 240, 240))
    np_image = np.array(pil_image)

    transform_image = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])
    np_image = transform_image(np_image).float()
    return np_image

#imgshow
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    return ax


#predict
def predict(img, model,device,cat_to_name, topk=5):
    model.to(device)
    model.eval()
    with torch.no_grad():
        image = img
        image = image.type(torch.FloatTensor).to(device)

        image = image.unsqueeze(0)

        output = model.forward(image)

        ps = F.softmax(output, dim=1)

        top_ps, top_classes = ps.topk(topk, dim=1)
        top_p = top_ps[0]
        idx_to_class = {val: cat_to_name[k] for k, val in model.class_to_idx.items()}

        top_class = [idx_to_class[i] for i in top_classes[0].cpu().numpy()]

    return top_p, top_class

#show top 5
def show_top_5(path, model,device, cat_to_name):
    plt.figure(figsize=(3, 6))
    pl = plt.subplot(2, 1, 1)

    image = process_image(path)
    title = path.split('/')
    name = cat_to_name[title[2]]
    print(name)
    imshow(image, pl, name)

    score, flowers_list = predict(image, model, 'cpu', cat_to_name)
    fig, pl = plt.subplots(figsize=(4, 3))
    sticks = np.arange(len(flowers_list))
    pl.barh(sticks, score, height=0.3, linewidth=2.0, align='center')
    pl.set_yticks(ticks=sticks)
    pl.set_yticklabels(flowers_list)
    plt.show()


if __name__ == "__main__":
    main()
