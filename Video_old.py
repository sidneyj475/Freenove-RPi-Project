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

    def IsValidImage4Bytes(self, buf):
        bValid = True
        if buf[6:10] in (b'JFIF', b'Exif'):
            if not buf.rstrip(b'\0\r\n').endswith(b'\xff\xd9'):
                bValid = False
        else:
            try:
                Image.open(io.BytesIO(buf)).verify()
            except Exception as e:
                logging.error(f"Invalid image bytes: {e}")
                bValid = False
        return bValid

    def LedChange(self, R, G, B):
        try:
            color = f"{self.intervalChar}{R}{self.intervalChar}{G}{self.intervalChar}{B}{self.endChar}"
            for i in range(1, 9):
                hex_val = hex(1 << (i-1))
                self.sendData(f"{cmd.CMD_LED}{self.intervalChar}{int(hex_val, 16)}{color}")
            cv2.waitKey(5)
        except Exception as e:
            logging.error(f"Error changing LED: {e}")

    def face_detect(self, img):
        try:
            if sys.platform.startswith(('win', 'darwin', 'linux')):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                model_name = 'Yolov5_models'
                yolov5_model = 'balls5n.pt'
                model_labels = 'balls5n.txt'

                CWD_PATH = os.getcwd()
                PATH_TO_LABELS = os.path.join(CWD_PATH, model_name, model_labels)
                PATH_TO_YOLOV5_GRAPH = os.path.join(CWD_PATH, model_name, yolov5_model)

                # Import Labels File
                with open(PATH_TO_LABELS, 'r') as f:
                    labels = [line.strip() for line in f.readlines()]

                # Initialize Yolov5
                model = yolov5.load(PATH_TO_YOLOV5_GRAPH)

                stride, names, pt = model.stride, model.names, model.pt
                logging.info(f'Stride: {stride}, Names: {names}')

                min_conf_threshold = 0.7
                model.conf = min_conf_threshold  # NMS confidence threshold
                model.iou = 0.45  # NMS IoU threshold
                model.agnostic = False  # NMS class-agnostic
                model.multi_label = True  # NMS multiple labels per box
                model.max_det = 1000  # maximum number of detections per image

                frame = img.copy()
                results = model(frame)
                predictions = results.pred[0]

                boxes = predictions[:, :4]
                scores = predictions[:, 4]
                classes = predictions[:, 5]
                results.render()

                imW, imH = int(640), int(640)
                frame_resized = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (imW, imH))
                input_data = np.expand_dims(frame_resized, axis=0)

                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

                lower_red = np.array([160, 20, 20])
                upper_red = np.array([185, 255, 255])
                lower_green = np.array([70, 50, 50])
                upper_green = np.array([90, 255, 255])
                lower_blue = np.array([90, 10, 10])
                upper_blue = np.array([110, 255, 255])
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

                for i, score in enumerate(scores):
                    if min_conf_threshold < score <= 1.0:
                        object_name = labels[int(classes[i])]
                        logging.info(f'Class: {object_name}, Conf: {score}')
                        xmin = int(max(1, boxes[i][0]))
                        ymin = int(max(1, boxes[i][1]))
                        xmax = int(min(imW, boxes[i][2]))
                        ymax = int(min(imH, boxes[i][3]))

                        croppedImage = frame[ymin:ymax, xmin:xmax]
                        ccx = int((xmax - xmin) / 2)
                        ccy = int((ymax - ymin) / 2)
                        hsv_pixel = cv2.cvtColor(croppedImage, cv2.COLOR_BGR2HSV)
                        pixel_center = hsv_pixel[ccy, ccx]
                        hue_value = pixel_center[0]
                        logging.info(f'Hue: {hue_value}')

                        for cnt in contours_red:
                            self.process_contour(cnt, img, 'Red', (0, 0, 255), [0, 0, 255])

                        for cnt in contours_green:
                            self.process_contour(cnt, img, 'Green', (0, 255, 0), [0, 255, 0])

                        for cnt in contours_blue:
                            self.process_contour(cnt, img, 'Blue', (255, 0, 0), [255, 0, 0])

                        for cnt in contours_yellow:
                            self.process_contour(cnt, img, 'Yellow', (255, 255, 0), [255, 255, 0])


                cv2.imwrite('video.jpg', frame)
        except Exception as e:
            logging.error(f"Error in face_detect: {e}")

    def process_contour(self, cnt, img, color_name, color_bgr, led_color):
        try:
            contour_area = cv2.contourArea(cnt)
            if contour_area > 1000:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, 2)
                cv2.putText(img, color_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
                self.LedChange(*led_color)
        except Exception as e:
            logging.error(f"Error processing contour: {e}")

    def streaming(self, ip):
        try:
            if self.client_socket is None:
                self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.client_socket.connect((ip, 8000))
                self.connection = self.client_socket.makefile('rb')
        except Exception as e:
            logging.error(f"Error connecting to stream: {e}")
            return
        stream_bytes = b' '
        while True:
            try:
                stream_bytes = self.connection.read(4)
                leng = struct.unpack('<L', stream_bytes[:4])
                jpg = self.connection.read(leng[0])
                if self.IsValidImage4Bytes(jpg):
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if self.video_Flag:
                        self.face_detect(image)
                        self.video_Flag = False
            except Exception as e:
                logging.error(f"Error during streaming: {e}")
                break

    def sendData(self, s):
        if self.connect_Flag:
            try:
                self.client_socket1.send(s.encode('utf-8'))
            except Exception as e:
                logging.error(f"Error sending data: {e}")

    def recvData(self):
        data = ""
        try:
            data = self.client_socket1.recv(1024).decode('utf-8')
        except Exception as e:
            logging.error(f"Error receiving data: {e}")
        return data

    def socket1_connect(self, ip):
        try:
            if self.client_socket1 is None:
                self.client_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.client_socket1.connect((ip, 5000))
            self.connect_Flag = True
            logging.info("Connection Successful!")
        except Exception as e:
            logging.error(f"Connect to server failed: {e}")
            self.connect_Flag = False
        time.sleep(5)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
