import os
import numpy
from torch import Tensor
import torch


def get_iter_file(iteration: int):
    return f'iteration-{iteration:04d}.pkl'



if __name__ == "__main__":
    filename = os.path.join(
                os.path.join("data", "crazyhouse_human"), get_iter_file(0).replace('.pkl', '')
            )
    try:
        data_tensor : Tensor = torch.load(filename + '-data.pkl', weights_only = False)
        policy_tensor = torch.load(filename + '-policy.pkl', weights_only = False)
        value_tensor = torch.load(filename + '-value.pkl', weights_only = False)
        torch.set_printoptions(profile="full")
        print(data_tensor)
        print(policy_tensor)
        print(value_tensor)
    except FileNotFoundError as e:
        print('Warning: could not find tensor data. ' + str(e))
        