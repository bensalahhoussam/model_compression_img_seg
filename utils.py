import torch
import torch.nn as nn
import torch.nn.init as init
from thop import profile
import os
from logger_config import logger
import time


def initialize_weights(model, initialization_type='kaiming_uniform'):
    """
    Initialize the weights of a model's layers based on the specified initialization type.

    Args:
        model (nn.Module): The model whose weights will be initialized.
        initialization_type (str): The type of weight initialization to use. Supported types:
            - 'kaiming_normal'
            - 'kaiming_uniform'
            - 'xavier_normal'
            - 'xavier_uniform'
            - 'normal'
            - 'zeros'
            - 'ones'
    """
    def _initialize_layer(layer):
        if initialization_type == 'kaiming_normal':
            init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif initialization_type == 'kaiming_uniform':
            init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        elif initialization_type == 'xavier_normal':
            init.xavier_normal_(layer.weight, gain=init.calculate_gain('leaky_relu'))
        elif initialization_type == 'xavier_uniform':
            init.xavier_uniform_(layer.weight, gain=init.calculate_gain('leaky_relu'))
        elif initialization_type == 'normal':
            init.normal_(layer.weight, mean=0.0, std=0.01)
        elif initialization_type == 'zeros':
            init.zeros_(layer.weight)
        elif initialization_type == 'ones':
            init.ones_(layer.weight)
        else:
            raise ValueError(f"Unsupported initialization type: {initialization_type}")

        if hasattr(layer, 'bias') and layer.bias is not None:
            init.zeros_(layer.bias)

    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            _initialize_layer(module)
        elif isinstance(module, nn.BatchNorm2d):
            init.ones_(module.weight)
            init.zeros_(module.bias)

def get_model_size(model):
    size_of_parameters = sum([param.element_size() * param.nelement() for param in model.parameters()])
    size_of_buffers = sum([buf.element_size() * buf.nelement() for buf in model.buffers()])
    model_size = size_of_parameters + size_of_buffers #Bytes
    KILOBYTE_TO_BYTE = 1024
    MEGABYTE_TO_KILOBYTE = 1024
    model_size = model_size / (KILOBYTE_TO_BYTE * MEGABYTE_TO_KILOBYTE) #MegaBytes
    return model_size

def get_metrics(model, input):
    MACs, params = profile(model, inputs=(input,), verbose=False)
    FLOPs = 2*MACs
    return MACs* 1e-6, FLOPs*1e-6, params*1e-6

def get_data(train_results,result_metrics):
    train_results["precision"].append(result_metrics["precision"])
    train_results["recall"].append(result_metrics["recall"])
    train_results["dice"].append(result_metrics["dice"])
    train_results["f1_score"].append(result_metrics["f1_score"])
    train_results["mAP"].append(result_metrics["mAP"])
    return train_results


def get_segment_labels(image, model, device):
    image = image.unsqueeze(0).to(device) # add a batch dimension
    with torch.no_grad():
        start_time = time.time()
        outputs = model(image)
        forward_time = time.time() - start_time
    return outputs,forward_time


checkpoint_dir = 'checkpoints'
def save_checkpoint(epoch, model, optimizer, scheduler, train_loss, train_metrics, valid_loss, valid_metrics, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict':  scheduler.state_dict(),
        'train_loss': train_loss,
        'train_metrics': train_metrics,
        'valid_loss': valid_loss,
        'valid_metrics': valid_metrics
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch.pth')
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved at {checkpoint_path}")

    if is_best:
        best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_checkpoint_path)
        logger.info(f"Best checkpoint saved at {best_checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_path,weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    train_metrics = checkpoint['train_metrics']
    valid_loss = checkpoint['valid_loss']
    valid_metrics = checkpoint['valid_metrics']
    logger.info(f"Loaded checkpoint from epoch {epoch + 1}")
    return epoch, train_loss, train_metrics, valid_loss, valid_metrics