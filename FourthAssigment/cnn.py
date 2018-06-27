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
        raise NotImplementedError("falta completar o método __init__ da classe CNN")
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
        raise NotImplementedError("falta completar o método forward da classe CNN")
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
        raise NotImplementedError("falta completar o método predict da classe CNN")
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
    raise NotImplementedError("falta completar a função train_model_img_classification")
    # criterion =  
    # optimizer =
    # END YOUR CODE

    train_loss = []
    valid_loss = []
    for epoch in range(config.epochs):
        for step, (images, labels) in enumerate(train_loader):
            # YOUR CODE HERE:
            # ii) You should zero the model gradients
            # and define the loss function for the train data.
            raise NotImplementedError("falta completar a função train_model_img_classification")
            # loss = 
            # END YOUR CODE
            if step % config.save_step == 0:
                # YOUR CODE HERE:
                # iii) You should define the loss function for the valid data.
                raise NotImplementedError("falta completar a função train_model_img_classification")
                # v_loss = 
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
            raise NotImplementedError("falta completar a função train_model_img_classification")            
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
