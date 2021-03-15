import torch 
from torchvision import transforms, models 

from PIL import Image 

import numpy as np
import matplotlib.pyplot as plt

import os

# Transformation for the input images
transform = transforms.Compose([transforms.Resize(300),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


learning_rate = 0.007
content_weight = 1e4        # Content weight for the loss function     
style_weight = 1e8          # Style weight for the loss function

interval = 500
epochs = 1000

test_path = os.getcwd() + "/tests/4/"   # path to the test folder, which is supposed to contain 'content.jpg' and 'style.jpg'

style_layer_weights = {"conv1_1" : 1.0,         # layers weight
                        "conv2_1" : 0.8,
                        "conv3_1" : 0.4,
                        "conv4_1" : 0.2,
                        "conv5_1" : 0.1}

def main():
    # Loading pretrained VGG19's weights
    model = models.vgg19(pretrained=True).features
    # Setting the model not to calculate the gradients as we already have them pretrained
    for p in model.parameters():
        p.requires_grad = False
    # uploading the model to the gpu
    model = to_gpu(model)

    # Loading input content image and style image and uploading them to the gpu after transforming them to be of the same shape
    content = Image.open(test_path + "content.jpg").convert("RGB")
    style = Image.open(test_path + "style.jpg").convert("RGB")
    #print("Content shape => ", content.shape)
    content = to_gpu(transform(content))
    style = to_gpu(transform(style))

    # Cloning the content image as we do not want to change the original image
    # For this image we do want to learn the weights as we're trying to find an optimal representation of content+style
    target = content.clone().requires_grad_(True)
    target = to_gpu(target)

    # Selecting the outputs of the desired layers for the content and for the style
    content_features = layer_outputs(model, content)
    style_features = layer_outputs(model, style)

    # Calculating the Gram matrix for each layer
    style_grams = { layer : gram_matrix(style_features[layer]) for layer in style_features }

    optimizer = torch.optim.Adam([target], lr=learning_rate)

    for epoch in range(1, epochs + 1):
        # Passing the output image through the model
        target_features = layer_outputs(model, target)
        # Calculating content loss
        content_loss = torch.mean((content_features['conv4_2']-target_features['conv4_2'])**2)

        # Calculation of the style loss as explained in the paper
        style_loss = 0
        for layer in style_layer_weights:
            style_gram = style_grams[layer]
            target_gram = target_features[layer]
            _, d, w, h = target_gram.shape
            target_gram = gram_matrix(target_gram)

            # Calculating style loss
            style_loss += (style_layer_weights[layer] * torch.mean((target_gram-style_gram)**2)) / (d*w*h)
        
        # Calculating weighted total loss
        total_loss = content_weight*content_loss + style_weight*style_loss 
        
        if epoch % 5 == 0:
            print("epoch {} loss: {}".format(epoch, total_loss))
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if epoch % interval == 0:
            plt.imsave(test_path + str(epoch) + '.png', imcnvt(target), format='png')
    


# This function selects the outputs of the desired layers from the given model
# Input: model - the pretrained model that we're using to extract features from the input image
#        input_image - the input image to be passed through the model
# Output: features - dictionary of (layer_name, layer_output)
#           where the layer_name is one of the layer specified in the 'layers' variable
def layer_outputs(model, input_image):
    # Layers to be extracted from the VGG19 model
    layers = {  '0' : 'conv1_1',
                '5' : 'conv2_1',
                '10': 'conv3_1',
                '19': 'conv4_1',
                '21' : 'conv4_2',
                '28': 'conv5_1'}

    features = {}
    x = input_image
    x = x.unsqueeze(0)
    for name, layer in model._modules.items():
        # The input has to be passed through all of the layers anyway
        x = layer(x)
        if name in layers:
            features[layers[name]] = x 
    
    return features

def imcnvt(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1, 2, 0)
    x = x*np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    return x

# This function calculates the Gram matrixex between every two feature maps
def gram_matrix(imgfeature):
    _,d,h,w = imgfeature.size()
    imgfeature = imgfeature.view(d,h*w)     # 'd' feature maps, (h,w) size of each feature map
    gram_mat = torch.mm(imgfeature, imgfeature.t())
    
    return gram_mat

# uploading the variable to the GPU
def to_gpu(x):
    return x.cuda() if torch.cuda.is_available() else x


if __name__ == "__main__":
    main()
