import time

from AIDetector_pytorch import Detector
import imutils

import cv2
fps=0.0




def main():
    global fps
    name = 'demo'

    det = Detector()
    # cap = cv2.VideoCapture(r'C:\Users\a\Yolov5-Deepsort\data\highway.mp4')
    cap = cv2.VideoCapture(r'D:\Vehicle-Detection-And-Speed-Tracking\Car_Opencv\gta2moify_fps_rate.mp4')
    videoWriter = None

    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        t1=time.time()

        result = det.feedCap(im)
        result = result['frame']
        cv2.putText(result, 'fps-{}'.format(fps), (1000, 280), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [0, 255, 255], thickness=3, lineType=cv2.LINE_AA)
        result = imutils.resize(result, height=500)
        fps = (fps + (1. / (time.time() - t1))) / 2  # 此处的time.time()就是检测完这张图片的结束时间,除以2是为了和之前的fps求一个平均
        print("fps= %.2f"%fps)
        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            videoWriter = cv2.VideoWriter(
                'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))

        videoWriter.write(result)
        cv2.imshow(name, result)
        t = int(1000 / fps)
        cv2.waitKey(t)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
