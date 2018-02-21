# -*- coding: utf-8 -*-
#

# Imports
import torch.utils.data
import dataset
from echotorch.transforms import text
import matplotlib.pyplot as plt


# Style change detection dataset
pan18loader = torch.utils.data.DataLoader(
    dataset.StyleChangeDetectionDataset(root='./data/', download=True, transform=text.GloveVector(), train=False, point_form='gauss', sigma=10)
)

# Get training data
for i, data in enumerate(pan18loader):
    # Inputs and labels
    inputs, labels, c = data

    # Print
    print(u"Inputs : {}".format(inputs.size()))
    print(u"Labels : {}".format(labels.size()))
    print(u"Class : {}".format(c))
    plt.plot(labels[0].numpy())
    plt.show()
# end for

print(pan18loader.dataset.changes_char)
