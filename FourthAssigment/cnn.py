import os
import subprocess
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class CNN(nn.Module):
    """
    Convolutional neural network model.

    You may find nn.Conv2d, nn.MaxPool2d and self.add_module useful here.

    :param config: hyper params configuration
    :type config: CNNConfig
    """
    def __init__(self,
                 config):
        super(CNN, self).__init__()
        # YOUR CODE HERE:
        import math

        self.conv = torch.nn.Sequential()

        self.conv.add_module("conv_0", torch.nn.Conv2d(config.channels, config.conv_architecture[0], config.kernel_sizes[0]))

        new_height = config.height - config.kernel_sizes[0]
        new_width = config.width - config.kernel_sizes[0]

        self.conv.add_module("relu_0", torch.nn.ReLU())
        self.conv.add_module("maxpool_0", torch.nn.MaxPool2d(config.pool_kernel[0]))

        new_height = math.ceil(new_height / config.pool_kernel[0])
        new_width = math.ceil(new_width / config.pool_kernel[0])

        for i in range(1, len(config.conv_architecture)) :
            self.conv.add_module("conv_" + str(i), torch.nn.Conv2d(config.conv_architecture[i - 1], config.conv_architecture[i], config.kernel_sizes[i]))

            new_height -= config.kernel_sizes[i]
            new_width -= config.kernel_sizes[i]

            self.conv.add_module("relu_" + str(i), torch.nn.ReLU())
            self.conv.add_module("maxpool_" + str(i), torch.nn.MaxPool2d(config.pool_kernel[i]))

            new_height = math.ceil(new_height / config.pool_kernel[i])
            new_width = math.ceil(new_width / config.pool_kernel[i])


        self.fc = torch.nn.Sequential()

        self.fc.add_module("lin_0", nn.Linear(config.conv_architecture[-1] * new_height * new_width, config.architecture[0]))
        self.fc.add_module("ReLu_0", torch.nn.ReLU())

        for i in range(len(config.architecture) - 1) :
            self.fc.add_module("lin_" + str(i + 1), nn.Linear(config.architecture[i], config.architecture[i + 1]))
            self.fc.add_module("ReLu_" + str(i + 1), torch.nn.ReLU())
        # END YOUR CODE

    def forward(self, x):
        """
        Computes forward pass

        :param x: input tensor
        :type x: torch.FloatTensor(shape=(batch_size, number_of_features))
        :return: logits
        :rtype: torch.FloatTensor(shape=[batch_size, number_of_classes])
        """
        # YOUR CODE HERE:
        logits = self.conv(x)
        logits = logits.view(logits.size()[0], -1)
        logits = self.fc(logits)
        # END YOUR CODE
        return logits

    def predict(self, x):
        """
        Computes model's prediction

        :param x: input tensor
        :type x: torch.FloatTensor(shape=(batch_size, number_of_features))
        :return: model's predictions
        :rtype: torch.LongTensor(shape=[batch_size])
        """
        # YOUR CODE HERE:
        predictions =  torch.argmax(nn.Softmax(dim=0)(self.forward(x)), dim=1)
        # END YOUR CODE
        return predictions


def train_model_img_classification(model,
                                   config,
                                   dataholder,
                                   model_path,
                                   verbose=True):
    """
    Train a model for image classification

    :param model: image classification model
    :type model: LogisticRegression or DFN
    :param config: image classification model
    :type config: LogisticRegression or DFN
    :param dataholder: data
    :type dataholder: DataHolder
    :param model_path: path to save model params
    :type model_path: str
    :param verbose: param to control print
    :type verbose: bool
    """
    train_loader = dataholder.train_loader
    valid_loader = dataholder.valid_loader

    best_valid_loss = float("inf")
    # YOUR CODE HERE:
    # i) define the loss criteria and the optimizer.
    # You may find nn.CrossEntropyLoss and torch.optim.SGD useful here.
    valid = iter(valid_loader)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    # END YOUR CODE

    train_loss = []
    valid_loss = []
    for epoch in range(config.epochs):
        for step, (images, labels) in enumerate(train_loader):
            # YOUR CODE HERE:
            # ii) You should zero the model gradients
            # and define the loss function for the train data.
            images = images / 255
            optimizer.zero_grad()
            loss = criterion((model.forward(images)), labels)
            # END YOUR CODE
            if step % config.save_step == 0:
                # YOUR CODE HERE:
                # iii) You should define the loss function for the valid data.
                batch_vx, batch_vy = next(valid)
                batch_vx = batch_vx / 255
                v_loss = criterion((model.forward(batch_vx)), batch_vy)
                # END YOUR CODE
                valid_loss.append(float(v_loss))
                train_loss.append(float(loss))
                if float(v_loss) < best_valid_loss:
                    msg = "\ntrain_loss = {:.3f} | valid_loss = {:.3f}".format(float(loss),float(v_loss))
                    torch.save(model.state_dict(), model_path)
                    best_valid_loss = float(v_loss)
                    if verbose:
                        print(msg, end="")
            # YOUR CODE HERE:
            # iv) You should do the back propagation
            # and do the optimization step.
            loss.backward()
            optimizer.step()
            # END YOUR CODE
    if verbose:
        x = np.arange(1, len(train_loss) + 1, 1)
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        ax.plot(x, train_loss, label='train loss')
        ax.plot(x, valid_loss, label='valid loss')
        ax.legend()
        plt.xlabel('step')
        plt.ylabel('loss')
        plt.title('Train and valid loss')
        plt.grid(True)
        plt.show()
