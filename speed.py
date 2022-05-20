import math

speed = [None] * 1000
threshold=0

def estimateSpeed(location1, location2):
    if abs(location2[1]-location1[1])>threshold:
        x1=(location2[0]+location2[2])/2
        y1 =(location2[1] + location2[3]) / 2
        x2 = (location1[0] + location1[2]) / 2
        y2 = (location1[1] + location1[3]) / 2
        # WIDTH =((location2[2]-location2[0])+(location2[2]-location2[0]))/2
        # HEIGHT =((location2[3]-location2[1])+(location2[3]-location2[1]))/2
        # carWidht = WIDTH/HEIGHT
        d_pixels = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
        # ppm = WIDTH / carWidht
        ppm=8.8
        d_meters = d_pixels / ppm
        # print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
        fps = 20
        speed = d_meters * fps * 3.6
        speed=int(speed)
        if speed>100:
            speed=100
        return speed

def Speedcal(carLocation1,carLocation2):

            [x1, y1, w1, h1] = carLocation1
            [x2, y2, w2, h2] = carLocation2

            # print 'previous location: ' + str(carLocation1[i]) + ', current location: ' + str(carLocation2[i])


            # print 'new previous location: ' + str(carLocation1[i])
            if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                speed = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                return speed

