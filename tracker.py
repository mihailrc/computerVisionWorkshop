import torch
import torch.backends.cudnn as cudnn

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

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

    def update(self, xywhs, scores, oids, im0):
        xywhsTensor = torch.Tensor(xywhs)
        scoresTensor = torch.Tensor(scores)
        return self.tracker.update(xywhsTensor, scoresTensor, oids, im0)