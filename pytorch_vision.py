from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# model_name = 'vgg11'
# model_name = 'squeezenet1_0'
# model_name = 'resnet18'
# model_name = 'deeplabv3_resnet101'
# model_name = 'mobilenet_v2'
# model_name = 'wide_resnet50_2'
# model_name = 'inception_v3'
model_name = 'googlenet'


model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True)
# or any of these variants

model.eval()


filename = ("dog.jpg")
# sample execution (requires torchvision)
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
# create a mini-batch as expected by the model
input_batch = input_tensor.unsqueeze(0)

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to(device)
    model.to(device)

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)


# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())


# writer = SummaryWriter("runs/"+model_name)
# writer.add_graph(model, input_batch)
# writer.close()

## Action: visualization
# tensorboard --logdir=runs
