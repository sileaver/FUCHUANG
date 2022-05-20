import math



threshold=0

def CountCar(bboxes2draw, before_bboxes2draw):
    left=set()
    right=set()
    left_num=0
    right_num = 0
    if bboxes2draw and before_bboxes2draw:
        for after in bboxes2draw :
            if len(after) == 6:
                x1, y1, x2, y2, cls_, track_id = after
                for before in before_bboxes2draw:
                    if len(before) == 6:
                        x3, y3, x4, y4, before_cls_, before_track_id = before
                        if after[1] - before[1] > threshold and before_track_id == track_id and before_cls_ == cls_:
                            id = cls_ + str(track_id)
                            left.add(id)
                            left_num = len(left)
                        elif after[1] - before[1] < -threshold and before_track_id == track_id and before_cls_ == cls_:
                            id = cls_ + str(track_id)
                            right.add(id)
                            right_num=len(right)

            elif len(after)==8:
                x1, y1, x2, y2, cls_, track_id,_,_ = after
                for before in before_bboxes2draw:
                    if len(before) == 8:
                        x3, y3, x4, y4, before_cls_, before_track_id,_,_ = before
                        if after[1] - before[1] >threshold and before_track_id == track_id and before_cls_ == cls_:
                            id = cls_ + str(track_id)
                            left.add(id)
                            left_num = len(left)
                        elif after[1] - before[1] < -threshold and before_track_id == track_id and before_cls_ == cls_:
                            id = cls_ + str(track_id)
                            right.add(id)
                            right_num=len(right)
    return left_num, right_num

def Countnum(bboxes2draw):
    person = 0
    rider = 0
    car = 0
    truck = 0
    bus = 0
    motor = 0
    bike = 0
    total = set()
    for box in bboxes2draw:
        if len(box) == 6:
            x1, y1, x2, y2, cls_, track_id = box
            id = cls_ + str(track_id)
            total.add(id)
        elif len(box) == 8:
            x1, y1, x2, y2, cls_, track_id, _, _ = box
            id = cls_ + str(track_id)
            total.add(id)
    for i in total:
        if 'person' in i:
            person += 1
        elif 'rider' in i:
            rider += 1
        elif 'car' in i:
            car += 1
        elif 'truck' in i:
            truck += 1
        elif 'bus' in i:
            bus += 1
        elif 'motor' in i:
            motor += 1
        elif 'bike' in i:
            bike += 1
    nonmotorize = person + rider + motor + bike
    motorize = car + truck + bus
    return motorize, nonmotorize






