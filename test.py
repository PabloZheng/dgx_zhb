from torchvision import transforms
import os
os.getcwd()
import time
from torch.utils.data import DataLoader
from P2PSA.models.resnet34 import ResNet34
import torch
# from dataset import ProductDataset
from metric import *
from sklearn import metrics
# from torch.utils.data import DataLoader
from torchvision import datasets, transforms

input_size = 224
mean, std = [0.5], [0.5]
labelnumber = [0, 1, 2, 3, 4]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ResNet34(1, 5)
state_dict = torch.load('./runs/resnet34/best_acc_96.pth')
model.load_state_dict(state_dict)
model.to(device)
model.eval()

test_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(input_size),
                                    transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])

test_valid = datasets.ImageFolder(root='./EUS_bulk/test/',
                                  transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_valid, batch_size=32,
                                          shuffle=False,
                                          num_workers=0)

time_all_start = time.time()
correct = 0
total = 0
class_correct = list(0. for i in range(5))
class_total = list(0. for i in range(5))
allpre = []
alllabel = []
filenames = []
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        t1 = time.time()
        outputs = model(inputs)
        # print("Single_image_time: ", time.time()-t1)
        # filenames.extend(filename)
        _, predicted = torch.max(outputs.data, 1)
        p = predicted.detach().cpu().numpy()
        p = p.tolist()
        allpre.extend(p)
        l = labels.detach().cpu().numpy()
        l = l.tolist()
        alllabel.extend(l)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        c = (predicted == labels).squeeze()
        for i in range(labels.size(0)):
            label = labels[i]
            try:
                class_correct[label] += c[i].item()
            except:
                print('as')
            class_total[label] += 1

time_all_end = time.time()
print("all_image_times: ", time_all_start-time_all_end)
print('Accuracy of the network on the images: %.2f %%' % (
        100 * correct / total))
for i in range(len(labelnumber)):
    print('Accuracy of %5s : %.2f %%' % (
        labelnumber[i], 100 * class_correct[i] / (class_total[i]+1e-9)))

target = ['class 0', 'class 1', 'class 2',  'class 3',  'class 4']
print(metrics.classification_report(alllabel, allpre, labels=labelnumber, digits=4, target_names=target))
make_confusion_matrix(alllabel, allpre, labelnumber, method='ResNet18')

