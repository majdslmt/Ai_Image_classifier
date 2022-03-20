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
from get_input_args import get_input_args
import gc

op = 0

def main():
    input_args = get_input_args()
    train_dataloader, valid_dataloader, test_dataloader, image_datasets = load_data(input_args.data_dir)


    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = init_model(input_args.arch, 512)

    print("start")
    training_network(model, train_dataloader,valid_dataloader,test_dataloader, device, image_datasets)



def load_data(dir):
    data_dir = dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    print(train_transform)

    valid_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print(valid_transform)

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print(test_transform)

    image_datasets = [datasets.ImageFolder(train_dir, transform=train_transform),
                      datasets.ImageFolder(valid_dir, transform=valid_transform),
                      datasets.ImageFolder(test_dir, transform=test_transform)]

    train_dataloader = torch.utils.data.DataLoader(image_datasets[0], batch_size=16, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(image_datasets[1], batch_size=16, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(image_datasets[2], batch_size=16, shuffle=True)
    print("data has been loaded")
    return train_dataloader, valid_dataloader, test_dataloader, image_datasets


def find_classifier(arch, hidden_units):
    if arch == "vgg13":
        classifier = nn.Sequential(
                            nn.Linear(25088, 4096),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(4096, 1000),
                            nn.ReLU(),
                            nn.Dropout(p=0.5),
                            nn.Linear(1000, 102),
                            nn.LogSoftmax(dim=1)
                        )
        return classifier
    else:
        classifier = nn.Sequential(nn.Linear(2208, hidden_units),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(hidden_units, int(hidden_units / 2)),
                                         nn.ReLU(),
                                         nn.Dropout(0.2),
                                         nn.Linear(int(hidden_units / 2), 102),
                                         nn.LogSoftmax(dim=1))
        return classifier


def init_model(arch, hidden_units):
    if arch == "vgg13":
        model = models.vgg13(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = find_classifier(arch, hidden_units)
        return model
    elif arch == "densenet121" :
        model = models.densenet121(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = find_classifier(arch, hidden_units)
        return model
    elif arch == "densenet161":
        model = models.densenet161(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False

        model.classifier = find_classifier(arch, hidden_units)
        return model


def  training_network(model, train_dataloader,valid_dataloader,test_dataloader, device, image_datasets):
    torch.backends.cudnn.allow_tf32 = True
    print(model.classifier)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.002)
    print_every = 5
    epochs = 2
    epoch = 0
    step = 0
    start_time = time.time()
    running_loss = 0
    print("training stared")
    model.to(device)

    for e in range(epochs):
        for (tr_inputs, tr_labels) in train_dataloader:  # for train
            step += 1
            tr_inputs, tr_labels = tr_inputs.to(device), tr_labels.to(device)
            optimizer.zero_grad()

            logps = model.forward(tr_inputs)
            tr_loss = criterion(logps, tr_labels)
            tr_loss.backward()
            optimizer.step()

            running_loss += tr_loss.item()

            if step % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                for (v_inputs, v_labels) in valid_dataloader:  # for validation
                    optimizer.zero_grad()

                    v_inputs, v_labels = v_inputs.to(device), v_labels.to(device)
                    with torch.no_grad():
                        logps = model.forward(v_inputs)
                        batch_loss = criterion(logps, v_labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == v_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                v_loss = test_loss / len(valid_dataloader)
                accuracy = accuracy / len(valid_dataloader)
                print(f"Ep > {e + 1} - {epochs}",
                      f"training loss > {running_loss / print_every:.3f} ..",
                      f"Validation loss > {v_loss}",
                      f"accuracy > {accuracy}")
                running_loss = 0
                epoch = e
                model.train()

    testing_network(model,device, test_dataloader)
    save_model(model, image_datasets, optimizer, epoch)

    print("done")


def testing_network(model, device ,test_dataloader):
    criterion = nn.NLLLoss()
    t_loss = 0
    accuracy = 0
    with torch.no_grad():
        for te_inputs, te_labels in test_dataloader:
            te_inputs, te_labels = te_inputs.to(device), te_labels.to(device)
            logps = model(te_inputs)
            ps = torch.exp(logps)

            b_loss = criterion(logps, te_labels)
            t_loss += b_loss.item()

            top_p, tp_class = ps.topk(1, dim=1)

            eq = tp_class == te_labels.view(*tp_class.shape)

            accuracy += torch.mean(eq.type(torch.FloatTensor)).item()

            print(f"test loss is: {t_loss / len(test_dataloader)}",
                  f"accuracy is: {accuracy / len(test_dataloader)}")


def save_model(model, image_datasets, optimizer, e):
    mode_checkpoint = {'model.classifier': model.classifier,
                       'model.class_to_idx': image_datasets[0].class_to_idx,
                       'state_dict': model.state_dict(),
                       'epoch': e,
                       'optimizer_state_dict': optimizer.state_dict()}

    torch.save(mode_checkpoint, 'save_majed_checkpoint.pth')


if __name__ == "__main__":
    main()
