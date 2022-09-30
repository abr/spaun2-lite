# program to capture single image from webcam in python

# importing OpenCV library
import numpy as np
import cv2
import sys
from abr_control.utils import colors as c

class colSegmentation:
    def __init__(self, debug=False, video=True):
        self.debug = debug
        self.video = video
        self.detected = False
        print(f'{c.green}STARTING COLOR SEGMENTATION{c.endc}')

    def get_mask(self, brightHSV, minHSV, maxHSV, kernel_size=None, get_contours=False, detect_thres=1e6, detection_mask=None):
    # def get_mask(self, brightHSV, minHSV, maxHSV, kernel_size=(7, 7), get_contours=True):
        """
        brightHSV:
        minHSV: len 3 list of floats
            minimum HSV threshold
        maxHSV: len 3 list of floats
            maximum HSV threshold
        kernel_size: 2 tuple of ints
            kernel size for filtering (7x7 works well)
        get_contours: bool, Optional (Default: False)
            returns red outline around segmented mask
        """
        # SIMPLE
        # maskHSV = cv2.inRange(brightHSV, minHSV, maxHSV)
        # resultHSV = cv2.bitwise_and(brightHSV, brightHSV, mask = maskHSV)

        # CLEANUP
        mask = cv2.inRange(brightHSV, minHSV, maxHSV)

        if kernel_size is not None:
            #define kernel size
            kernel = np.ones(kernel_size,np.uint8)
            # Remove unnecessary noise from mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # detection_mask[:, :320] = int(1)
        # # for ii in range(0, detection_mask.shape[0]):
        # for jj in range(0, detection_mask.shape[1]):
        #     if jj > 100 and jj < 540:
        #         detection_mask[:, jj] = int(1)
        #     # cv2.imshow('', detection_mask)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(detection_mask)
        # plt.show()

        # select an area of the image to check for detections
        if detection_mask is not None:
            mask = cv2.bitwise_and(mask, detection_mask, mask=mask)

        if np.sum(mask) > detect_thres:
            self.detected = True
        else:
            self.detected = False
        # print('DETECTED: ', np.sum(mask))

        resultHSV = cv2.bitwise_and(brightHSV, brightHSV, mask=mask)

        if get_contours:
            # Find contours from the mask
            contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # resultHSV = cv2.drawContours(resultHSV, contours, -1, (0, 0, 255), 3)
        else:
            contours = None



        return resultHSV, contours

    # mouse callback function
    def showPixelValue(self, event,x,y,flags,param):
        global img, combinedResult, placeholder

        if event == cv2.EVENT_MOUSEMOVE:
            # get the value of pixel from the location of mouse in (x,y)
            bgr = img[y,x]

            # Convert the BGR pixel into other colro formats
            ycb = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2YCrCb)[0][0]
            lab = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2Lab)[0][0]
            hsv = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2HSV)[0][0]

            # Create an empty placeholder for displaying the values
            placeholder = np.zeros((img.shape[0],400,3),dtype=np.uint8)

            # fill the placeholder with the values of color spaces
            cv2.putText(placeholder, "BGR {}".format(bgr), (20, 70), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(placeholder, "HSV {}".format(hsv), (20, 140), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(placeholder, "YCrCb {}".format(ycb), (20, 210), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
            cv2.putText(placeholder, "LAB {}".format(lab), (20, 280), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)

        # if placeholder is not None:
        #     # Combine the two results to show side by side in a single image
        #     combinedResult = np.hstack([img,placeholder])

            # cv2.imshow('PRESS P for Previous, N for Next Image',combinedResult)


    def run(self, kernel_size=None, get_contours=False, detect_thres=1e6, detection_mask=None):
        self.running = True
        # initialize the camera
        # If you have multiple camera connected with
        # current device, assign a value in cam_port
        # variable according to that
        cam_port = 0
        cam = cv2.VideoCapture(cam_port)

        #python
        # bgr = [40, 158, 16]
        # minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
        # maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])
        #
        # maskBGR = cv2.inRange(bright,minBGR,maxBGR)
        # resultBGR = cv2.bitwise_and(bright, bright, mask = maskBGR)

        #convert 1D array to 3D, then convert it to HSV and take the first element
        # this will be same as shown in the above figure [65, 229, 158]
        # hsv = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2HSV)[0][0]

        # #convert 1D array to 3D, then convert it to YCrCb and take the first element
        # ycb = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2YCrCb)[0][0]
        #
        # minYCB = np.array([ycb[0] - thresh, ycb[1] - thresh, ycb[2] - thresh])
        # maxYCB = np.array([ycb[0] + thresh, ycb[1] + thresh, ycb[2] + thresh])
        #
        # maskYCB = cv2.inRange(brightYCB, minYCB, maxYCB)
        # resultYCB = cv2.bitwise_and(brightYCB, brightYCB, mask = maskYCB)
        #
        # #convert 1D array to 3D, then convert it to LAB and take the first element
        # lab = cv2.cvtColor( np.uint8([[bgr]] ), cv2.COLOR_BGR2LAB)[0][0]
        #
        # minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
        # maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])
        #
        # maskLAB = cv2.inRange(brightLAB, minLAB, maxLAB)
        # resultLAB = cv2.bitwise_and(brightLAB, brightLAB, mask = maskLAB)
        # reading the input using the camera

        # test with pink sticky notes
        # hsv = [165, 95, 245]
        # hsv of red balls in plastic jar in lab, lamp and no overhead lights
        # hsv = [3, 67, 87]
        hsv = [4, 154, 111]
        thresh = 60

        minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
        maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])


        if not self.video:
            result, bright = cam.read()
            brightHSV = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)

            resultHSV = self.get_mask(brightHSV, minHSV, maxHSV)
            while self.running:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                # cv2.imshow("Result BGR", resultBGR)
                cv2.imshow("Result HSV", resultHSV)

            cam.release()
            cv2.destroyAllWindows()
        else:
            global img
            # global combinedResult
            global placeholder
            placeholder = None
            combinedResult = None
            cv2.namedWindow('live_stream')
            # cv2.setWindowProperty(
            #     "live_stream",
            #     cv2.WND_PROP_FULLSCREEN,
            #     cv2.WINDOW_FULLSCREEN
            # )
            while self.running:
                ret, frame = cam.read()
                brightHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                resultHSV, contours = self.get_mask(
                    brightHSV,
                    minHSV,
                    maxHSV,
                    kernel_size,
                    get_contours,
                    detect_thres,
                    detection_mask
                )
                # resultHSV, contours = self.get_mask(brightHSV, minHSV, maxHSV)
                # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                if contours is not None:
                    frame = cv2.drawContours(frame, contours, -1, (0, 0, 255), 3)

                img = np.concatenate((frame, resultHSV), axis=1)
                # cv2.imshow('Input', np.concatenate((img, resultHSV), axis=1))
                # if combinedResult is None:
                if placeholder is None:
                    cv2.imshow('live_stream', img)
                else:
                    combinedResult = np.hstack([img,placeholder])
                    cv2.imshow('live_stream', combinedResult)
                if self.debug:
                    cv2.setMouseCallback('live_stream',self.showPixelValue)

                c = cv2.waitKey(1)
                if c == 27:
                    break

            cam.release()
            cv2.destroyAllWindows()

        # cv2.imshow("Result YCB", resultYCB)
        # cv2.imshow("Output LAB", resultLAB)
        # If image will detected without any error,
        # show result
        # if result:
        #
        #     # showing result, it take frame name and image
        #     # output
        #     cv.imshow("GeeksForGeeks", image)
        #
        #     # saving image in local storage
        #     cv.imwrite("GeeksForGeeks.png", image)
        #
        #     # If keyboard interrupt occurs, destroy image
        #     # window
        #     cv.waitKey(0)
        #     cv.destroyWindow("GeeksForGeeks")
        #
        # # If captured image is corrupted, moving to else part
        # else:
        #     print("No image detected. Please! try again")
if __name__ == '__main__':
    debug = False
    video = True
    if 'debug' in sys.argv:
        debug = True

    detection_mask = np.zeros((480, 640), dtype='uint8')
    detection_mask[:, 140:500] = int(1)
    col_seg = colSegmentation(debug=debug, video=video)
    col_seg.run(
        kernel_size=(7, 7),
        get_contours=True,
        detect_thres=5e-4,
        detection_mask=detection_mask)
