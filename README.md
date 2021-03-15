# Style-Transfer

A work done by Lior Danino and Daniel Goman

A project in the course of Deep Learning Intro

Followed by previous work, this project features implementation of a deep learning model which, given a content image and style image, generates an output image which applies the style (colors, texture, etc) of the style image to the content image, while maintaining the main content of the content imagine relatively intact.

The above is achieved by using various layers of a pretrained VGG-19 model, through which we pass the output image which we're trying to iptimize as the process goes on.

The optimization objective is to minimize the weighted sum of losses over the content and the style images, for specifics - please refer to our paper.
