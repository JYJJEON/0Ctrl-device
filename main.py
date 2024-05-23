import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np

from IPython.display import display
import ipywidgets
import ipywidgets.widgets as widgets
import traitlets
from jetbot import Robot, ObjectDetector, Camera, bgr8_to_jpeg

import threading
import time
from SCSCtrl import TTLServo

from pymongo import MongoClient
from datetime import datetime

# MongoDB에 연결
client = MongoClient("mongodb://3.36.60.248:27017")

# 필요한 데이터베이스와 컬렉션 선택
db = client['zero_ctrl']  # 데이터베이스 이름으로 대체

table_control = db['control']
table_log = db['log']
robot = Robot()
camera = Camera.instance(width=300, height=300)

# 모델 로드
model = torchvision.models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('../best_steering_model_xy_test.pth'))

device = torch.device('cuda')
model = model.to(device)
model = model.eval().half()

mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()

normalize = torchvision.transforms.Normalize(mean, std)

collisionavoid = ObjectDetector('./ssd_mobilenet_v2_coco.engine')

print('모델 로드 성공')

xPos = 100
yPos = 0
servoPos_4 = 0

servoPos_1 = 0
servoPos_5 = 0

def stop():
    robot.stop()

def step_forward():
    robot.forward(0.4)

def step_backward():
    robot.backward(0.4)

def step_left():
    robot.left(0.3)
    time.sleep(0.5)
    robot.stop()

def step_right():
    robot.right(0.3)
    time.sleep(0.5)
    robot.stop()

def xIn():
    global xPos
    xPos -= 5
    TTLServo.xyInput(xPos, yPos)

def xDe():
    global xPos
    xPos += 5
    if xPos < 85:
        xPos = 85
    TTLServo.xyInput(xPos, yPos)

def yIn():
    global yPos
    yPos -= 5
    TTLServo.xyInput(xPos, yPos)

def yDe():
    global yPos
    yPos += 5
    TTLServo.xyInput(xPos, yPos)

def grab():
    global servoPos_4
    servoPos_4 -= 15
    if servoPos_4 < -90:
        servoPos_4 = -90
    TTLServo.servoAngleCtrl(4, servoPos_4, 1, 150)

def loose():
    global servoPos_4
    servoPos_4 += 15
    if servoPos_4 > -10:
        servoPos_4 = -10
    TTLServo.servoAngleCtrl(4, servoPos_4, 1, 150)

def limitCtl(maxInput, minInput, rawInput):
    if rawInput > maxInput:
        limitBuffer = maxInput
    elif rawInput < minInput:
        limitBuffer = minInput
    else:
        limitBuffer = rawInput
    return limitBuffer

def cameraUp():
    global servoPos_5
    servoPos_5 = limitCtl(25, -40, servoPos_5 - 15)
    TTLServo.servoAngleCtrl(5, servoPos_5, -1, 150)

def cameraDown():
    global servoPos_5
    servoPos_5 = limitCtl(25, -40, servoPos_5 + 15)
    TTLServo.servoAngleCtrl(5, servoPos_5, -1, 150)

def ptRight():
    global servoPos_1
    servoPos_1 = limitCtl(80, -80, servoPos_1 + 15)
    TTLServo.servoAngleCtrl(1, servoPos_1, 1, 150)

def ptLeft():
    global servoPos_1
    servoPos_1 = limitCtl(80, -80, servoPos_1 - 15)
    TTLServo.servoAngleCtrl(1, servoPos_1, 1, 150)

def resetRobot():
    TTLServo.servoAngleCtrl(1, 0, -1, 100)
    TTLServo.servoAngleCtrl(5, 50, -1, 100)
    TTLServo.servoAngleCtrl(4, 0, -1, 100)

from datetime import datetime

init_document = {
    "time": datetime.now(),
    "toggle_move": {
        "mode": "stop"
    },
    "manual_move": {
        "direction": "",
        "speed": 0
    },
    "auto_move": {
        "speed_gain": 0.3,
        "steering_gain": 0.26,
        "steering_kd": 0.0,
        "steering_bias": 0.0
    },
    "ready_replace": {
        "mode": ""
    },
    "gripper": {
        "action": ""
    },
    "move_xy": {
        "direction": ""
    },
    "camera_move": {
        "direction": ""
    },
    "start_area": {
        "color": "purple"
    },
    "end_area": {
        "color": "blue"
    }
}

result = table_control.insert_one(init_document)
print(result)

areaA = init_document['start_area']['color']
areaB = init_document['end_area']['color']
speedGain = init_document['auto_move']['speed_gain']
steeringGain = init_document['auto_move']['steering_gain']
steeringKd = init_document['auto_move']['steering_kd']
steeringBias = init_document['auto_move']['steering_bias']

xReturn = 0.0
yReturn = 0.0
speedReturn = 0.0
steeringReturn = 0.0

colors = [
        {'name': 'red', 'lower': np.array([0, 140, 170]), 'upper': np.array([11, 180, 255])},
        {'name': 'green', 'lower': np.array([64, 90, 160]), 'upper': np.array([71, 130, 180])},
        {'name': 'blue', 'lower': np.array([98, 150, 150]), 'upper': np.array([113, 250, 200])},
        {'name': 'purple', 'lower': np.array([123, 102, 131]), 'upper': np.array([132, 110, 154])},
        {'name': 'yellow', 'lower': np.array([25, 150, 167]), 'upper': np.array([33, 255, 230])},
        {'name': 'orange', 'lower': np.array([12, 148, 170]), 'upper': np.array([18, 205, 255])}
        # Add more colors as needed
    ]

areaA_color = next((color for color in colors if color['name'] == areaA), None)
areaB_color = next((color for color in colors if color['name'] == areaB), None)

findArea = areaA

frame_width = 300
frame_height = 300
camera_center_X = int(frame_width / 2)
camera_center_Y = int(frame_height / 2)

colorHSVvalueList = []
max_len = 20

roadFinding = None
goalFinding = None
collisionAvoiding = None

class WorkingAreaFind(threading.Thread):
    global areaA,areaB,areaA_color,areaB_color
    flag = 1

    def __init__(self):
        super().__init__()
        self.th_flag = True
        self.imageInput = 0

    def run(self):
        while self.th_flag:
            self.imageInput = camera.value
            # BGR을 HSV로 변환
            hsv = cv2.cvtColor(self.imageInput, cv2.COLOR_BGR2HSV)
            # 블러 처리
            hsv = cv2.blur(hsv, (15, 15))

            # areaA, areaB 색상 탐색
            areaA_mask = cv2.inRange(hsv, areaA_color['lower'], areaA_color['upper'])
            areaA_mask = cv2.erode(areaA_mask, None, iterations=2)
            areaA_mask = cv2.dilate(areaA_mask, None, iterations=2)

            areaB_mask = cv2.inRange(hsv, areaB_color['lower'], areaB_color['upper'])
            areaB_mask = cv2.erode(areaB_mask, None, iterations=2)
            areaB_mask = cv2.dilate(areaB_mask, None, iterations=2)

            AContours, _ = cv2.findContours(areaA_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            BContours, _ = cv2.findContours(areaB_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if AContours and WorkingAreaFind.flag == 1:
                self.findCenter(areaA, AContours)

            elif BContours and WorkingAreaFind.flag == 2:
                self.findCenter(areaB, BContours)

            time.sleep(0.1)

    def findCenter(self, name, Contours):
        global findArea
        c = max(Contours, key=cv2.contourArea)
        ((box_x, box_y), radius) = cv2.minEnclosingCircle(c)

        X = int(box_x)
        Y = int(box_y)

        error_Y = abs(camera_center_Y - Y)
        error_X = abs(camera_center_X - X)

        if error_Y < 30 and error_X < 30:
            if name == areaA and self.flag == 1:
                robot.stop()
                WorkingAreaFind.flag = 2
                print(WorkingAreaFind.flag)
                findArea = areaB

                roadFinding.stop()
                roadFinding.join()
                print("도로 추적 종료")

            elif name == areaB and WorkingAreaFind.flag == 2:
                WorkingAreaFind.flag = 1

                roadFinding.stop()
                roadFinding.join()
                print("도로 추적 종료")

    def stop(self):
        self.th_flag = False
        robot.stop()

class RobotMoving(threading.Thread):
    def __init__(self):
        super().__init__()
        self.th_flag = True
        self.angle = 0.0
        self.angle_last = 0.0

    def run(self):
        global xReturn, yReturn, speedReturn, steeringReturn
        while self.th_flag:
            image = camera.value
            xy = model(self.preprocess(image)).detach().float().cpu().numpy().flatten()
            # 객체 탐지
            detections = collisionavoid(image)
            matching_detections = [d for d in detections[0] if d['label'] == 1]
            flag1 = 0

            for det in matching_detections:
                bbox = det['bbox']
                size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # 바운딩 박스의 면적 계산
                if size > 0.2:
                    flag1 = 1
                    robot.stop()
                    time.sleep(1)

            if flag1 == 0:
                x = xy[0]
                y = (0.5 - xy[1]) / 2.0

                xReturn = x
                yReturn = y

                speedReturn = speedGain

                self.angle = np.arctan2(x, y)

                if not self.th_flag:
                    break
                pid = self.angle * steeringGain + (
                            self.angle - self.angle_last) * steeringKd
                self.angle_last = self.angle

                steeringReturn = pid + steeringBias

                robot.left_motor.value = max(min(speedReturn + steeringReturn, 1.0), 0.0)
                robot.right_motor.value = max(min(speedReturn - steeringReturn, 1.0), 0.0)
                if not self.th_flag:
                    break

            time.sleep(0.1)

        robot.stop()
        goalFinding.stop()
        goalFinding.join()
        print("영역 찾기 종료")

    def preprocess(self, image):
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device).half()
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def stop(self):
        self.th_flag = False
        robot.stop()

def autoStart():
    global roadFinding, goalFinding
    goalFinding = WorkingAreaFind()
    roadFinding = RobotMoving()
    goalFinding.start()
    roadFinding.start()

def autoStop():
    if goalFinding is not None:
        goalFinding.stop()
        goalFinding.join()
    if roadFinding is not None:
        roadFinding.stop()
        roadFinding.join()

runStatus = 0

# left_cnt = 서버에서 받아오기
# if left_cnt == 0:
#     데이터를 init값으로 변경

resetRobot()

while True:
    pre_document = table_control.find_one(sort=[("time", -1)])
    if pre_document is None:
        # pre_document가 None이면 기본값 설정
        pre_document = init_document

    areaA = pre_document['start_area']['color']
    areaB = pre_document['end_area']['color']
    speedGain = pre_document['auto_move']['speed_gain']
    steeringGain = pre_document['auto_move']['steering_gain']
    steeringKd = pre_document['auto_move']['steering_kd']
    steeringBias = pre_document['auto_move']['steering_bias']

    flagArea = areaA

    if pre_document['toggle_move']['mode'] == 'auto' and runStatus == 0:
        autoStart()
        runStatus = 1
    elif pre_document['toggle_move']['mode'] == 'stop' and runStatus == 1:
        autoStop()
        runStatus = 0


    post_log = {
        "time": datetime.now(),
        "run_status": {
            "run": runStatus,
            "speed": float(speedReturn),
            "x": float(xReturn),
            "y": float(yReturn),
            "steering": float(steeringReturn),
            "find_area": findArea
        }
    }

    # 로그 컬렉션에 문서 삽입
    result = table_log.insert_one(post_log)


