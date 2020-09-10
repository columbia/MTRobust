import math
import sys

import torch
from torch import nn
import models.drn as drn

# class Decoder(nn.Module):
#
#     def __init__(self,
#                  output_channels = 3,
#                  ):
#
#         self.output_channels = 3
#
#         # DEFINING MODEL AS COMPLEMENT OF BASE CHOSEN FOR DRN
#
#         self.layer_list = model
#
#         self.decoder_model = nn.Sequential(*layer_list)
#
#
#
#     def forward(self, representation):
#
#         x = self.

def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]



class DRNSegDepth(nn.Module):
    def __init__(self,
                 model_name, # tells which DRN architecture has to be loaded.
                 classes=19, # tells how many classes the drn model is used for, though this is for the last layer, may not make sense here.
                 pretrained_model=None,
                 pretrained=True,
                 tasks=[], # So that we can initialise the network for specific tasks
                 use_torch=False, #TODO - may not be needed.
                 old_version=False): #TODO: See all the parameters, are these enough.

        super(DRNSegDepth, self).__init__()

        # Get the DRN model skeleton based on model_name
        model = drn.__dict__.get(model_name)(
            pretrained=pretrained, num_classes=1000
        )
        pmodel = nn.DataParallel(model)

        if pretrained_model is not None:
            pmodel.load_state_dict(pretrained_model)

        # ch = list(model.children())
        # ch1 = (list(model.children())[:-5])
        # ch2 = (list(model.children())[:-2])

        self.branching_layer_number = 5

        # Decide a base from DRN based on which we want to branch into 3 different tasks
        self.encoder = nn.Sequential(*nn.ModuleList(model.children())[:-self.branching_layer_number])

        self.tasks = tasks
        self.softmax = nn.LogSoftmax()
        self.softmax_only = nn.Softmax()

        # Make a decoder for each task - dict so that it is easy to extend for other datasets
        self.task_to_decoder = nn.ModuleDict({})

        if self.tasks is not None:

            for task in self.tasks:
                if task == 'segmentsemantic':
                    output_channels = classes

                if task == 'segment_semantic':
                    output_channels = classes

                if task == 'depth_zbuffer' or task == 'depth':
                    output_channels = 1 # TODO : Confirm if depth is just for one channel

                if task == 'autoencoder' or task == 'reconstruct':
                    output_channels = 3

                # MAKE A SEPARATE DECODER FOR EACH TASK AND PUT IN DICTIONARY
                decoder = nn.ModuleList(model.children())[-self.branching_layer_number:-2]
                decoder.extend([nn.Conv2d(model.out_dim, output_channels,kernel_size=1,bias=True)])
                up = nn.ConvTranspose2d(output_channels, output_channels, 16, stride=8, padding=4, output_padding=0, groups=output_channels, bias=False)
                fill_up_weights(up)
                up.weight.requires_grad = False

                decoder.extend([up])

                decoder = nn.Sequential(*(decoder))

                # Finally, decoder should contain all the layers from end of the model
                # decoder = Decoder(output_channels,num_layers)
                self.task_to_decoder[task] = decoder

        else:
            # Assume segmentation if no tasks are given
            print("\n NO TASKS GIVEN IN CONFIG FILE \n")
            output_channels = 3

        #self.decoders = nn.ModuleDict(self.task_to_decoder)


    def forward(self,x) :

        rep = self.encoder(x)
        outputs = {'rep' : rep}

        for i, (task,decoder) in enumerate(self.task_to_decoder.items()):

            # DEBUG- TEST SIZES - MULTIPLE DECODERS
            # for m in decoder.children():
            #     print("APPLYING ", m)
            #     rep = m(rep)
            #     print(rep.shape)

            decoder_output = decoder(rep)
            if task != 'segmentsemantic' and task != 'segment_semantic':
                outputs[task] = decoder_output
            else:
                outputs[task] = self.softmax(decoder_output)

        #THINK ABOUT THE LAST LINEARITY

        return outputs




