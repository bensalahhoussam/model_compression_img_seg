import torch
from logger_config import logger
from inference import  model_prediction,get_prediction_map
import cv2 as cv


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):

        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score =0.
        self.early_stop = False

    def __call__(self, val_metric, model,epoch):

        if self.best_score==0.:
            self.save_checkpoint(val_metric, model,epoch)
            self.best_score = val_metric
        elif val_metric >   self.best_score + self.delta :  # For metrics where higher is better (e.g., accuracy)
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:

            self.save_checkpoint(val_metric, model,epoch)
            self.counter = 0
            self.best_score = val_metric


    def save_checkpoint(self, val_metric, model,epoch):

        if self.verbose:
            logger.info(f"Validation metric improved ({self.best_score:.3f} --> {val_metric:.3f}). Saving model...")
        torch.save({'model_state_dict':model.state_dict()}, f"best_model/checkpoint_{epoch}.pt")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.eval().to(device)
        outputs, image, forward_time = model_prediction("dataset/idd/val/images/0000000_leftImg8bit.jpg", 256, model, device)
        segmentation_map = get_prediction_map(outputs)
        output_final = cv.cvtColor(segmentation_map, cv.COLOR_RGB2BGR)
        cv.imwrite(f"outputs/inference_results/out_{epoch}.png", output_final)



