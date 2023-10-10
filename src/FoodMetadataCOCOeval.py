from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import numpy as np

class FoodCOCOEval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
      super().__init__(cocoGt, cocoDt, iouType)
      if self.params.iouType == 'bbox': self.params.score_key = 'box_confidence'
      elif self.params.iouType == 'segm': self.params.score_key = 'mask_confidence'

    def computeIoU(self, imgId, catId):
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d[p.score_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d[p.confidence_key] for d in dt]  # Use 'mask_confidence' if p.iouType == 'segm'
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d[p.confidence_key] for d in dt]  # Use 'box_confidence' if p.iouType == 'bbox'
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        ious = maskUtils.iou(d, g, iscrowd)
        return ious