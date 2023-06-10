import torch


def get_available_device() -> str:
    """Returns an available device for pytorch

    If CUDA is available prefers any CUDA device
    over CPU.

    Returns
    -------
        A string representation of pytorch device.
    """

    if torch.cuda is None or \
            not torch.cuda.is_available():
        return 'cpu'

    return 'cuda'
