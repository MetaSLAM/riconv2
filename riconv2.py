import torch
import time
import logging

from automerge.models.vi_pc_learning_lib.riconv2.models.riconv2_cls import get_model
from automerge.models.vi_pc_learning_lib.vi_backbone import ViBackbone

class RiConv2(ViBackbone):
    """
    RiConv2 is a viewpoint invariant point cloud feature extractor.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.model = get_model(config.num_class, config.n, config.normal_channel)
    
    def infer(self, x, t=False):
        """
        Infer the viewpoint invariant point cloud feature extractor with one frame.
        """
        self.model.eval()
        with torch.no_grad():
            x = None
            if t:
                t1 = time.time()
            with torch.no_grad():
                _, x = self.model(x) # only the local features are saved
            if t:
                t2 = time.time()
                t_delta = t2 - t1
                return x.detach().cpu().numpy(), t_delta
            return x.detach().cpu().numpy()

    def load_weight(self, model_path):
        """
        Load weights
        """
        if model_path is not None:
            try:
                self.model.load_state_dict(torch.load(model_path))
                logging.info(f"Loading weights from {model_path}")
            except FileNotFoundError:
                logging.error(f"Could not load weights from {model_path}")
        else:
            logging.info("Use weight path from the configuration.")
            if self.config.weights.riconv2 is not None:
                self.load_weight(self.config.weights.riconv2)
            else:
                raise ValueError("No weights provided for the viewpoint invariant backbone model.")
        