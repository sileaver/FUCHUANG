from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
from speed import Speedcal
import time
from counter import *


threshold=24



palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)


def plot_bboxes(image, bboxes,left_num,right_num,waitcarnum,waitpersonnum,motorize,nonmotorize, line_thickness=None):
    # Plots one bounding box on image img


    cv2.putText(image, 'left-{}'.format(left_num), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, 'right-{}'.format(right_num), (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, 'waitcarnum-{}'.format(waitcarnum), (100, 160), cv2.FONT_HERSHEY_SIMPLEX, 1,
                [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, 'waitpersonnum-{}'.format(waitpersonnum), (100, 190), cv2.FONT_HERSHEY_SIMPLEX, 1,
                [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, 'motorize-{}'.format(motorize), (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1,
                [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
    cv2.putText(image, 'nonmotorize-{}'.format(nonmotorize), (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)
    # cv2.putText(image, 'fps-{}'.format(fps), (1000, 280), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             [0, 255, 255], thickness=3, lineType=cv2.LINE_AA)

    for box in bboxes:
        if len(box)==8:
            tl = line_thickness or round(
                0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness

            x1, y1, x2, y2, cls_id, pos_id,speed,start = box
            # if cls_id=="bus" or cls_id=="person":
            #     continue
            if cls_id in ['person']:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            c1, c2 = (x1, y1), (x2, y2)

            cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled

            cv2.putText(image, '{} ID-{}-{}km/hr'.format(cls_id, pos_id,speed), (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)



        elif len(box)==6:
                tl = line_thickness or round(
                    0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness

                x1, y1, x2, y2, cls_id, pos_id= box
                # if cls_id == "bus" or cls_id == "person":
                #     continue
                if cls_id in ['person']:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)
                c1, c2 = (x1, y1), (x2, y2)

                cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(cls_id, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled

                cv2.putText(image, '{} ID-{}'.format(cls_id, pos_id), (c1[0], c1[1] - 2), 0, tl / 3,
                            [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


    return image

def update_tracker(target_detector, image,before_bboxes2draw):
    waitcarnum = 0
    waitpersonnum = 0
    motorize = 0
    nonmotorize = 0
    left_num = 0
    right_num = 0
    new_faces = []
    _, bboxes = target_detector.detect(image)

    bbox_xywh = []
    confs = []
    clss = []

    for x1, y1, x2, y2, cls_id, conf in bboxes:

        obj = [
            int((x1+x2)/2), int((y1+y2)/2),
            x2-x1, y2-y1
        ]
        bbox_xywh.append(obj)
        confs.append(conf)
        clss.append(cls_id)

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    outputs = deepsort.update(xywhs, confss, clss, image)

    bboxes2draw = []
    face_bboxes = []
    current_ids = []
    for value in list(outputs):
        x1, y1, x2, y2, cls_, track_id = value
        bboxes2draw.append(
            (x1, y1, x2, y2, cls_, track_id)
        )
        current_ids.append(track_id)
        if cls_ == 'face':
            if not track_id in target_detector.faceTracker:
                target_detector.faceTracker[track_id] = 0
                face = image[y1:y2, x1:x2]
                new_faces.append((face, track_id))
            face_bboxes.append(
                (x1, y1, x2, y2)
            )
    if before_bboxes2draw:
        for i,box in enumerate(bboxes2draw):
            x1, y1, x2, y2,cls_,track_id=box
            for before in before_bboxes2draw:
                if len(before)==6:
                    x3, y3, x4, y4,before_cls_,before_track_id=before
                    if before_track_id==track_id and before_cls_==cls_:
                        start = time.time()
                        speed=Speedcal([x3,y3,x4,y4],[x1,y1,x2,y2])
                        bboxes2draw[i]=(x1,y1,x2,y2,cls_,track_id,speed,start)
                    if bboxes2draw and before_bboxes2draw:
                         left_num, right_num = CountCar(bboxes2draw, before_bboxes2draw)

                         motorize,nonmotorize= Countnum(bboxes2draw)
                elif len(before)==8:

                    x3, y3, x4, y4, before_cls_, before_track_id, speed, start = before
                    if before_track_id==track_id and before_cls_==cls_:
                        after = time.time()
                        if abs(y4-y2)<threshold:
                            speed=Speedcal([x3,y3,x4,y4],[x1,y1,x2,y2])
                            if after-start>120:
                                if cls_ in ['car','truck','bus']:
                                    waitcarnum += 1
                                elif cls_ in ['person','rider','motor','bike']:
                                    waitpersonnum += 1
                            bboxes2draw[i]=(x1,y1,x2,y2,cls_,track_id,speed,start)

                        else:
                            speed = Speedcal([x3, y3, x4, y4], [x1, y1, x2, y2])
                            start=after
                            bboxes2draw[i] = (x1, y1, x2, y2, cls_, track_id, speed, start)
                    if bboxes2draw and before_bboxes2draw:
                         left_num, right_num = CountCar(bboxes2draw, before_bboxes2draw)
                         motorize,nonmotorize= Countnum(bboxes2draw)



    ids2delete = []
    for history_id in target_detector.faceTracker:
        if not history_id in current_ids:
            target_detector.faceTracker[history_id] -= 1
        if target_detector.faceTracker[history_id] < -5:
            ids2delete.append(history_id)

    for ids in ids2delete:
        target_detector.faceTracker.pop(ids)
        print('-[INFO] Delete track id:', ids)

    image = plot_bboxes(image, bboxes2draw,left_num,right_num,waitcarnum,waitpersonnum,motorize,nonmotorize)
    return image, new_faces, face_bboxes,bboxes2draw
