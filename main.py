from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import torchvision.transforms as transforms

from vgg19 import vgg19

from PIL import Image
import matplotlib.pyplot as plt

USE_GPU = True

# desire image width and height feeds into CNN model
IMG_WIDTH = 512 if USE_GPU else 128
IMG_HEIGHT = 384 if USE_GPU else 96
IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)

DTYPE = torch.cuda.FloatTensor if USE_GPU else torch.FloatTensor

MSG_DISPLAY_FREQ = 50

def load_image(image_name, transform, size, dtype):
    image = Image.open(image_name)
    image = image.resize(size, Image.BILINEAR)
    image = transform(image)
    # fake batch dimension
    image = image.unsqueeze(0)
    return Variable(image.type(dtype))

def save_image(tensor, filename=None):
    image = tensor.clone().cpu()
    # remove the fake batch dimension
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    image.save(filename)

def imshow(tensor, title=None):
    image = tensor.clone().cpu()
    # remove fake batch dimension
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    if title is not None:
        plt.title(title)
    plt.imshow(image)


CONTENT_LAYERS = ['conv4_2']
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

def get_content_losses(input, target, content_wight):
    criterion = nn.MSELoss()
    losses = []
    for layer in CONTENT_LAYERS:
        loss = criterion(input[layer], target[layer].detach()) * content_wight
        losses.append(loss)
    return losses


def gram(feature_map):
    """ compute gram matrix """
    batch_size, depth, height, width = feature_map.size()
    # vectorize each feature map and reshape into a matrix
    features = feature_map.view(depth, height * width)
    GramMat = torch.mm(features, features.t())
    if USE_GPU:
        GramMat = GramMat.cuda()
    return GramMat


def get_style_losses(inputs, targets, style_weight):
    criterion = nn.MSELoss()
    losses = []
    for layer in STYLE_LAYERS:
        # MSELoss of input and target layers' Gram Matrix
        input = inputs[layer]
        target = targets[layer].detach()
        loss = criterion(gram(input), gram(target)) * style_weight
        losses.append(loss)
    return losses


def transfer_style(content_img, style_img, input_img, model, num_steps=500, style_weight=1000, content_weight=1):
    # extract content representaion
    content_repr = model(content_img, selected_layers=CONTENT_LAYERS)
    # extract style representation
    style_repr = model(style_img, selected_layers=STYLE_LAYERS)
    # set input image as parameter for optimization
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])

    timesteps = [0]
    while timesteps[0] <= num_steps:

        def closure():
            # correct the value of updated image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            input_repr = model(input_param, selected_layers=CONTENT_LAYERS+STYLE_LAYERS)
            content_losses = get_content_losses({layer: input_repr[layer] for layer in CONTENT_LAYERS}, content_repr, style_weight)
            style_losses = get_style_losses({layer: input_repr[layer] for layer in STYLE_LAYERS}, style_repr, content_weight)

            style_score = 0
            content_score = 0

            for sloss in style_losses:
                style_score += sloss / len(STYLE_LAYERS)
            for closs in content_losses:
                content_score += closs / len(CONTENT_LAYERS)

            timesteps[0] += 1
            if timesteps[0] % MSG_DISPLAY_FREQ == 0:
                print("After {} timesteps:".format(timesteps[0]))
                print('Style Loss: {:5f} Content Loss: {:5f}'.format(style_score.data[0]/MSG_DISPLAY_FREQ, content_score.data[0]/MSG_DISPLAY_FREQ))
                # uncomment to save the output every 50 iterations (remember to create an 'output' folder)
                # save_image(input_param.data, './output/' + str(timesteps[0]) + '.png')

            score = style_score + content_score
            score.backward(retain_graph=True)

            return score

        optimizer.step(closure)

    # last correction...
    input_param.data.clamp_(0, 1)

    return input_param


def main():

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    content_img = load_image("images/tuebingen_neckarfront.jpg", transform, IMG_SIZE, DTYPE)
    style_img = load_image("images/picasso.jpg", transform, IMG_SIZE, DTYPE)

    plt.figure()
    imshow(style_img.data, title='Style Image')
    plt.figure()
    imshow(content_img.data, title='Content Image')

    model = vgg19(pretrained=True)

    if USE_GPU:
        model = model.cuda()

    input_img = content_img.clone()
    # run style transfer
    output = transfer_style(content_img, style_img, input_img, model)
    # display result
    plt.figure()
    imshow(output.data, title='Output Image')
    plt.show()

if __name__ == '__main__':
    main()
