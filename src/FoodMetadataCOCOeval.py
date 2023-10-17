from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import numpy as np
from collections import defaultdict
from FoodMetadataCOCO import FoodMetadata

class FoodCOCOEval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        super().__init__(cocoGt, cocoDt, iouType)
        if self.params.iouType == 'bbox': self.params.confidence_key = 'box_confidence'
        elif self.params.iouType == 'segm': self.params.confidence_key = 'mask_confidence'

        self.mask_dir = '/tmp/masks'
            
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
        inds = np.argsort([-d[p.confidence_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in inds]
        if len(dt) > p.maxDets[-1]:
            dt = dt[0:p.maxDets[-1]]

        if p.iouType == 'segm':
            g = [g['segmentation'] for g in gt]
            d = [d['segmentation'] for d in dt]
        elif p.iouType == 'bbox':
            g = [g['bbox'] for g in gt]
            d = [d['bbox'] for d in dt]
        else:
            raise Exception('unknown iouType for iou computation')

        # compute iou between each dt and gt region
        iscrowd = [int(o['iscrowd']) for o in gt]
        print(f'd: {d}')
        print(f'g: {g}')
        print(f'iscrowd: {iscrowd}')
        ious = maskUtils.iou(d, g, iscrowd)
        return ious

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _Prune(coco):
            coco.drop_ann(coco.dataset['info']['detection_issues']['failures'])

        def _toBox(anns):
            def xyxy_to_xywh(bbox: list) -> list:
                w = bbox[3] - bbox[1] # y2 - y1
                h = bbox[2] - bbox[0] # x2 - x1
                return [bbox[0], bbox[1], w, h]
            for ann in anns:
                if "bbox" in ann:
                    ann["bbox"] = xyxy_to_xywh(ann["bbox"])

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle

        def _toMaskPrediction(anns, coco, iouType):
            for ann in anns:
                if "mask" in ann: 
                    if ann["mask"].split('.')[1] == '.pt':
                        mask = torch.load(f"{self.mask_dir}/100198_1603.pt")
                        mask = np.asarray(mask.astype(np.uint8), order="F")
                        rle = maskUtils.encode(mask)
                        area = maskUtils.area(rle)
                        if iouType == "segm":
                            ann["segmentation"] = rle
                        ann["area"] = area

        p = self.params
        print(f"before: {len(self.cocoDt.anns.keys())}")
        _Prune(self.cocoDt)

        # test to see if there's masks
        print(f"after: {len(self.cocoDt.anns.keys())}")
        count = 0
        for ann in self.cocoDt.anns.values():
            if "mask" in ann:
                count += 1
        print(f"masks: {count}")
        exit()

        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))
        
        if p.iouType == 'bbox':
            _toBox(dts)
        # change boxes from xyxy to xywh 
        # convert ground truth to mask if iouType == 'segm'
        _toMaskPrediction(dts, self.cocoDt, p.iouType)
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results
    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None

        for g in gt:
            if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d[p.confidence_key] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        ious = self.ious[imgId, catId][:, gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G))
        dtm  = np.zeros((T,D))
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        if not len(ious)==0:
            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t,1-1e-10])
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind,gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately
                        iou=ious[dind,gind]
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[tind,dind] = gtIg[m]
                    dtm[tind,dind]  = gt[m]['id']
                    gtm[tind,m]     = d['id']
        # set unmatched detections outside of area range to ignore
        a = np.array([d['area']<aRng[0] or d['area']>aRng[1] for d in dt]).reshape((1, len(dt)))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.repeat(a,T,0)))
        # store results for given image and category
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'aRng':         aRng,
                'maxDet':       maxDet,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                'dtScores':     [d[p.confidence_key] for d in dt],
                'gtIgnore':     gtIg,
                'dtIgnore':     dtIg,
            }

if __name__ == '__main__':

    gt = FoodMetadata('/me/examples/public_validation_set_release_2.1.json')
    dt = FoodMetadata('/me/examples/public_validation_set_release_2.1_blip2_glove_dino_sam.json')

    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    imgIds = sorted(dt.imgs.keys())

    cocoEval = FoodCOCOEval(gt, dt, annType)
    cocoEval.params.imgIds  = list(dt.imgs.keys())
    cocoEval.params.useCats = 1
    cocoEval.params.catIds = list(dt.cats.keys())
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()