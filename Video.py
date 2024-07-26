#!/usr/bin/python 
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import socket
import io
import os
import sys
import struct
import yolov5
from PIL import Image
from multiprocessing import Process
from Command import COMMAND as cmd
import time
import logging

class VideoStreaming:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
        self.video_Flag = True
        self.connect_Flag = False
        self.face_x = 0
        self.face_y = 0
        self.endChar = '\n'
        self.intervalChar = '#'
        self.client_socket = None
        self.client_socket1 = None
        self.connection = None
        self.intervalChar='#'
        self.endChar='\n'

    def StartTcpClient(self, IP):
        try:
            if self.client_socket1 is None:
                self.client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if self.client_socket is None:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket.connect((IP, 8000))
            self.connection = self.client_socket.makefile('rb')
        except Exception as e:
            logging.error(f"Error starting TCP client: {e}")
    
    def StopTcpcClient(self):
        try:
            if self.client_socket:
                self.client_socket.shutdown(socket.SHUT_RDWR)
                self.client_socket.close()
                self.client_socket = None
            if self.client_socket1:
                self.client_socket1.shutdown(socket.SHUT_RDWR)
                self.client_socket1.close()
                self.client_socket1 = None
        except Exception as e:
            logging.error(f"Error stopping TCP client: {e}")

    def IsValidImage4Bytes(self,buf): 
        bValid = True
        if buf[6:10] in (b'JFIF', b'Exif'):     
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                bValid = False
        else:        
            try:  
                Image.open(io.BytesIO(buf)).verify() 
            except:  
                bValid = False
        return bValid
    
    def Led_Off(self):
        led_Off = self.intervalChar + str(0) + self.intervalChar + str(0) + self.intervalChar + str(0) + self.endChar
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x01) + led_Off)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x02) + led_Off)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x04) + led_Off)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x08) + led_Off)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x10) + led_Off)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x20) + led_Off)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x40) + led_Off)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x80) + led_Off)
    
    def LedChange(self,R,G,B):
        led_Off = self.intervalChar + str(0) + self.intervalChar + str(0) + self.intervalChar + str(0) + self.endChar
        color = self.intervalChar + str(R) + self.intervalChar + str(G) + self.intervalChar + str(B) + self.endChar

        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x01) + color)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x02) + color)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x04) + color)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x08) + color)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x10) + color)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x20) + color)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x40) + color)
        self.sendData(cmd.CMD_LED + self.intervalChar + str(0x80) + color)

        #self.sendData(cmd.CMD_LED + self.intervalChar + str(0x01) + led_Off)
        #self.sendData(cmd.CMD_LED + self.intervalChar + str(0x02) + led_Off)
        #self.sendData(cmd.CMD_LED + self.intervalChar + str(0x04) + led_Off)
        #self.sendData(cmd.CMD_LED + self.intervalChar + str(0x08) + led_Off)
        #self.sendData(cmd.CMD_LED + self.intervalChar + str(0x10) + led_Off)
        #self.sendData(cmd.CMD_LED + self.intervalChar + str(0x20) + led_Off)
        #self.sendData(cmd.CMD_LED + self.intervalChar + str(0x40) + led_Off)
        #self.sendData(cmd.CMD_LED + self.intervalChar + str(0x80) + led_Off)

    def pos_adjus (self,middlepixel):

            #self.sendData(cmd.CMD_MODE+self.intervalChar+'zero'+self.endChar) # Turn on Nothing?
            #time.sleep(0.1)

            Stop=self.intervalChar+str(0)+self.intervalChar+str(0)+self.intervalChar+str(0)+self.intervalChar+str(0)+self.endChar
            Turn_Left=self.intervalChar+str(-800)+self.intervalChar+str(-800)+self.intervalChar+str(800)+self.intervalChar+str(800)+self.endChar
            Turn_Right=self.intervalChar+str(800)+self.intervalChar+str(800)+self.intervalChar+str(-800)+self.intervalChar+str(-800)+self.endChar
            Go_Forward=self.intervalChar+str(800)+self.intervalChar+str(800)+self.intervalChar+str(800)+self.intervalChar+str(800)+self.endChar


            if not ((180 < middlepixel) and (middlepixel < 230)):
            

                if (180 > middlepixel):
                    # Turn Left
                    print ("turn left")
                    self.sendData(cmd.CMD_MOTOR+Turn_Left)

                    '''time.sleep(0.5)

                    self.sendData(cmd.CMD_MOTOR+Stop)'''


                else:
                    # Turn Right
                    print ("Turn Right")
                    self.sendData(cmd.CMD_MOTOR+Turn_Right)

                    '''time.sleep(0.5)

                    self.sendData(cmd.CMD_MOTOR+Stop)'''

            elif ((180 < middlepixel) and (middlepixel < 230)):
                self.sendData(cmd.CMD_MOTOR+Go_Forward)
                #time.sleep(0.7)

            else:
                self.sendData(cmd.CMD_MOTOR+Go_Forward)
                

                    

    def face_detect(self,img):
        if sys.platform.startswith('win') or sys.platform.startswith('darwin') or sys.platform.startswith('linux'):
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            
            model_name = 'Yolov5_models'
            yolov5_model = 'balls5n.pt'
            model_labels = 'balls5n.txt'

            CWD_PATH = os.getcwd()
            PATH_TO_LABELS = os.path.join(CWD_PATH,model_name,model_labels)
            PATH_TO_YOLOV5_GRAPH = os.path.join(CWD_PATH,model_name,yolov5_model)

            # Import Labels File
            with open(PATH_TO_LABELS, 'r') as f:
                labels = [line.strip() for line in f.readlines()]

            # Initialize Yolov5
            model = yolov5.load(PATH_TO_YOLOV5_GRAPH)

            stride, names, pt = model.stride, model.names, model.pt
            print('stride = ',stride, 'names = ', names)
            #model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup

            min_conf_threshold = 0.7
            # set model parameters
            model.conf = 0.7  # NMS confidence threshold
            model.iou = 0.45  # NMS IoU threshold
            model.agnostic = False  # NMS class-agnostic
            model.multi_label = True # NMS multiple labels per box
            model.max_det = 1000  # maximum number of detections per image

            frame = img.copy()
            results = model(frame)
            predictions = results.pred[0]

            boxes = predictions[:, :4]
            scores = predictions[:, 4]
            classes = predictions[:, 5]
            # Draws Bounding Box onto image
            results.render() 

            # Initialize frame rate calculation
            frame_rate_calc = 30
            freq = cv2.getTickFrequency()

            frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #imW, imH = int(400), int(300)
            imW, imH = int(640), int(640)
            frame_resized = cv2.resize(frame_rgb, (imW, imH))
            input_data = np.expand_dims(frame_resized, axis=0)

            max_score = 0
            max_index = 0
            print('flag')

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            lower_red = np.array([0, 100, 100])
            upper_red = np.array([10, 255, 255])

            lower_green = np.array([40, 20, 50])
            upper_green = np.array([90, 255, 255])

            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])

            lower_yellow = np.array([15, 50, 50])
            upper_yellow = np.array([35, 255, 255])

            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Loop over all detections and draw detection box if confidence is above minimum threshold
            for i in range(len(scores)):
                curr_score = scores.numpy()
                # Found desired object with decent confidence
                if ((curr_score[i] > min_conf_threshold) and (curr_score[i] <= 1.0)):
                    print('Class: ',labels[int(classes[i])],' Conf: ', curr_score[i])

                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                    xmin = int(max(1,(boxes[i][0])))
                    ymin = int(max(1,(boxes[i][1])))
                    xmax = int(min(imW,(boxes[i][2])))
                    ymax = int(min(imH,(boxes[i][3])))
                            
                    # Draw label
                    object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                    label = '%s: %d%%' % (object_name, int(curr_score[i]*100)) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    #cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
                    #cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    #cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                    #if cType.getType() == "ball":

                    croppedImage = frame[ymin:ymax,xmin:xmax]
                    cv2.imwrite('video.jpg', frame)

                    # Find Center
                    ccx = int((xmax - xmin)/2)
                    ccy = int((ymax - ymin)/2)

                    middlepixel = (xmin + xmax)/2
                    self.pos_adjus(middlepixel)

                    #pixel = croppedImage[ccx,ccy]
                    hsv_pixel = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
                    pixel_center = hsv_pixel[ccy,ccx]
                    hue_value = pixel_center[0]
                    
                    ball_color = "None"
                    if hue_value in range(160,180):
                        ball_color = 'RED'
                        self.LedChange(255,0,0)
                        '''time.sleep(2)
                        self.Led_Off()'''
                    elif hue_value in range(70,94):
                        ball_color = 'GREEN'
                        self.LedChange(0,255,0)
                        '''time.sleep(2)
                        self.Led_Off()'''
                    elif hue_value in range(95,105):
                        ball_color = "BLUE"
                        self.LedChange(0,0,255)
                        '''time.sleep(2)
                        self.Led_Off()'''
                    elif hue_value in range(24,34):
                        ball_color = "YELLOW"
                        self.LedChange(255,255,0)
                        '''time.sleep(2)
                        self.Led_Off()'''
                    else:
                        self.Led_Off()

                    print('Hue: ',hue_value, ball_color)

                    # Record current max
                    max_score = curr_score[i]
                    max_index = i

                    #for cnt in contours_red:
                        #contour_area = cv2.contourArea(cnt)
                        #if contour_area > 1000:
                            #x, y, w, h = cv2.boundingRect(cnt)
                            #cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
                            #cv2.putText(img, 'Red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

                            #cv2.imwrite('video.jpg',frame)
                            #self.LedChange(255,0,0)

                    #for cnt in contours_green:
                        #contour_area = cv2.contourArea(cnt)
                        #if contour_area > 1000:
                            #x, y, w, h = cv2.boundingRect(cnt)
                            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            #cv2.putText(img, 'Green', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            #cv2.imwrite('video.jpg',frame)
                            #self.LedChange(0,255,0)

                    #for cnt in contours_blue:
                        #contour_area = cv2.contourArea(cnt)
                        #if contour_area > 1000:
                            #x, y, w, h = cv2.boundingRect(cnt)
                            #cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
                            #cv2.putText(img, 'Blue', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                            #cv2.imwrite('video.jpg',frame)
                            #self.LedChange(0,0,255)

                    #for cnt in contours_yellow:
                       # contour_area = cv2.contourArea(cnt)
                        #if contour_area > 1000:
                           # x, y, w, h = cv2.boundingRect(cnt)
                            #cv2.rectangle(img, (x, y), (x + w, y + h), (255,255,0), 2)
                            #cv2.putText(img, 'Yellow', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
                            #cv2.imwrite('video.jpg',frame)
                            #self.LedChange(255,255,0)

            cv2.imwrite('video.jpg',frame)

    def streaming(self,ip):
        stream_bytes = b' '
        try:
            self.client_socket.connect((ip, 8000))
            self.connection = self.client_socket.makefile('rb')
        except:
            #print "command port connect failed"
            pass
        while True:
            try:
                stream_bytes= self.connection.read(4) 
                leng=struct.unpack('<L', stream_bytes[:4])
                jpg=self.connection.read(leng[0])
                if self.IsValidImage4Bytes(jpg):
                            image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if self.video_Flag:
                                self.face_detect(image)
                                self.video_Flag=False
            except Exception as e:
                print (e)
                break
                  
    def sendData(self,s):
        if self.connect_Flag:
            self.client_socket1.send(s.encode('utf-8'))

    def recvData(self):
        data=""
        try:
            data=self.client_socket1.recv(1024).decode('utf-8')
        except:
            pass
        return data

    def socket1_connect(self,ip):
        try:
            self.client_socket1.connect((ip, 5000))
            self.connect_Flag=True
            print ("Connection Successful !")
        except Exception as e:
            print ("Connect to server Failed!: Server IP is right? Server is opened?")
            self.connect_Flag=False
            

        time.sleep(5)
if __name__ == '__main__':
    pass
