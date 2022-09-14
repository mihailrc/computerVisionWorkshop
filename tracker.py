import torch

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import numpy as np

class DeepsSortTracker:

    def __init__(self):
        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        self.tracker = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                            nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

    def xyxy2xywh(self, x):
        x = np.array(x)
        y = np.copy(x)
        y[:,0] = (x[:,0] + x[:,2]) / 2  # x center
        y[:,1] = (x[:,1] + x[:,3]) / 2  # y center
        y[:,2] = x[:,2] - x[:,0]  # width
        y[:,3] = x[:,3] - x[:,1]  # height
        return y

    def update(self, xyxy, scores, class_ids, im0):
        xywhsTensor = torch.Tensor(self.xyxy2xywh(xyxy))
        scoresTensor = torch.Tensor(scores)
        outputs = self.tracker.update(xywhsTensor, scoresTensor, class_ids, im0)
        if len(outputs) > 0:
            xxyys = outputs[:, :4]
            class_ids = outputs[:, -1]
            object_ids = outputs[:, -2]
            return xxyys, class_ids,object_ids
        return None,None,None