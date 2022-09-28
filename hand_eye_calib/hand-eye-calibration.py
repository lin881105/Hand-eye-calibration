import time
import rospy
import numpy as np
import quaternion # pip install numpy-quaternion
from quaternion import from_euler_angles
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Float32
from sensor_msgs.msg import CameraInfo, Image
from pynput.keyboard import Listener, KeyCode
from scipy.spatial.transform import Rotation as R
from scipy.linalg import solve_sylvester
from apriltag_ros.msg import AprilTagDetectionArray
from apriltag_ros.msg import AprilTagDetection
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
from PIL import Image as Img
import scipy.io as sio

#export ID1=745212070452

class Calibration():
  def __init__(self):
    self.curr_pos = np.array([0,0.5,0])
    self.curr_ori = np.array([1,0,0,0])
    self.br = CvBridge()
    self.R_gripper2base=[]
    self.t_gripper2base=[]
    self.R_target2cam=[]
    self.t_target2cam=[]
    self.gripper_flag = False
    self.cam_flag = False
    self.cam_side_rgb_flag = False
    self.cam_side_depth_flag = False
    self.frame = None
    self.side_rgb = None
    self.side_depth = None
    self.R = None
    self.t = None
    self.cnt = 0
    rospy.Subscriber("/745212070452/color/image_raw", Image, self.cameraFrame_callback)
    rospy.Subscriber("/141722071222/color/image_raw", Image, self.cameraFrameSideRGB_callback)
    rospy.Subscriber("/141722071222/aligned_depth_to_color/image_raw", Image, self.cameraFrameSideDepth_callback)
    rospy.Subscriber("/141722071222/aligned_depth_to_color/camera_info",CameraInfo,self.cameraInfo_callback)
    rospy.Subscriber("/cartesian_pose", PoseStamped, self.ee_pos_callback)
    rospy.Subscriber("/tag_detections",AprilTagDetectionArray,self.aprilTag_pos_callback)
    # self.frame = Img.open('test.png')

  def cameraInfo_callback(self, data):
    self.K = np.array(data.K).reshape([3, 3])
    self.D = np.array(data.D)
  
  def cameraFrame_callback(self, msg):
    self.frame = np.reshape(self.br.imgmsg_to_cv2(msg, msg.encoding), (480, 640, 3))

  def cameraFrameSideRGB_callback(self, msg):
    if self.cam_side_rgb_flag == True:
      try:
          self.side_rgb = self.br.imgmsg_to_cv2(msg, msg.encoding)
          print(f'side rgb {self.cnt}')
      except CvBridgeError as e:
          print(e)
      self.cam_side_rgb_flag = False

  def cameraFrameSideDepth_callback(self, msg):
    if self.cam_side_depth_flag == True:
      try:
          self.side_depth = self.br.imgmsg_to_cv2(msg, msg.encoding)
          print(f'side depth {self.cnt}')
      except CvBridgeError as e:
          print(e)
      self.cam_side_depth_flag = False

  def ee_pos_callback(self, data):
    if self.gripper_flag == True:
      self.curr_pos = np.array([[data.pose.position.x], [data.pose.position.y], [data.pose.position.z]])
      self.curr_ori = np.array([data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z, data.pose.orientation.w])

      r = R.from_quat(self.curr_ori).as_matrix()
      T_gripper2base = np.identity(4)
      T_gripper2base[:3, :3] = r 
      T_gripper2base[:3, 3] = np.reshape(self.curr_pos, (3))
      # T_gripper2base = np.linalg.inv(T_gripper2base)

      # r = R.from_quat(self.curr_ori)
      # r = r.as_matrix()
      self.R_gripper2base.append(T_gripper2base[:3, :3])
      self.t_gripper2base.append(T_gripper2base[:3, 3])
      # print(f'end effector pose : {T_gripper2base[:3, 3]}, {T_gripper2base[:3, :3]}')

      self.gripper_flag = False

  def debug(self):
    print(self.A)
  
  def aprilTag_pos_callback(self,data):
    if self.cam_flag == True:
      if len(data.detections)>0: 
        x = data.detections[0].pose.pose.pose.position
        q = data.detections[0].pose.pose.pose.orientation
        pos = np.array([[x.x],[x.y],[x.z]])
        ori = np.array([q.x,q.y,q.z,q.w])
        r = R.from_quat(ori)
        r = r.as_matrix()
        r = np.asarray(r)
        self.R_target2cam.append(r)
        self.t_target2cam.append(pos)
        # print(f'tag2cam pose : {pos}, {r}')
      self.cam_flag = False

  def run(self):
    key = None
    while True:
      if self.frame is not None:
        cv2.imshow('test', self.frame)
        key = cv2.waitKey(10)
      if key == ord('s'):
        self.cnt += 1
        print(f'process {self.cnt}')
        self.gripper_flag = True
        self.cam_flag = True
      if key == ord('q'):
        self.cam_side_rgb_flag = True
        self.cam_side_depth_flag = True
        time.sleep(1)
        cv2.destroyAllWindows()
        break


  def solve_B(self):

    print(len(self.R_gripper2base),len(self.t_gripper2base),len(self.R_target2cam),len(self.t_target2cam))
    self.R, self.T = cv2.calibrateHandEye(self.R_gripper2base,self.t_gripper2base,self.R_target2cam,self.t_target2cam)
    data = {
      'R_cam2ee' : self.R.tolist(),
      't_cam2ee' : self.T.tolist(),
      'R_gripper2base' : self.R_gripper2base,
      't_gripper2base' : self.t_gripper2base,
      'R_target2cam' : self.R_target2cam,
      't_target2cam' : self.t_target2cam,
    }

    sio.savemat('data.mat', data)   

    return self.T,self.R
  
  def last_frame(self):
    self.cam_side_rgb_flag = True
    self.cam_side_depth_flag = True


  def debug(self):
    T = np.identity(4)
    T[:3, :3] = self.R
    T[:3, 3] = np.reshape(self.T, (3))
    for i in range(len(self.R_gripper2base)-1):
        T_tc1 = np.identity(4)
        T_tc1[:3, :3] = self.R_target2cam[i]
        T_tc1[:3, 3] = np.reshape(self.t_target2cam[i], (3))

        T_tc2 = np.identity(4)
        T_tc2[:3, :3] = self.R_target2cam[i+1]
        T_tc2[:3, 3] = np.reshape(self.t_target2cam[i+1], (3))

        T_eb1 = np.identity(4)
        T_eb1[:3, :3] = self.R_gripper2base[i]
        T_eb1[:3, 3] = np.reshape(self.t_gripper2base[i], (3))

        T_eb2 = np.identity(4)
        T_eb2[:3, :3] = self.R_gripper2base[i+1]
        T_eb2[:3, 3] = np.reshape(self.t_gripper2base[i+1], (3))

        left = T @ T_tc2 @ np.linalg.inv(T_tc1)
        right = np.linalg.inv(T_eb2) @ T_eb1 @ T
        print(f'{left-right}')

  

if __name__ == '__main__':
  rospy.init_node('Calibration', anonymous=True)

  calibration=Calibration()
  calibration.run()
  t, r=calibration.solve_B()
  print(t,r)
  R_gripper2base = calibration.R_gripper2base[-1]
  t_gripper2base = calibration.t_gripper2base[-1]
  R_target2cam = calibration.R_target2cam[-1]
  t_target2cam = calibration.t_target2cam[-1]
  side_rgb = calibration.side_rgb
  side_depth = calibration.side_depth

  T_cam_ee = np.identity(4)
  T_cam_ee[:3, :3] = r
  T_cam_ee[:3, 3] = np.reshape(t, (3))

  T_gripper2base = np.identity(4)
  T_gripper2base[:3, :3] = R_gripper2base
  T_gripper2base[:3, 3] = np.reshape(t_gripper2base, (3))

  T_target2cam = np.identity(4)
  T_target2cam[:3, :3] = R_target2cam
  T_target2cam[:3, 3] = np.reshape(t_target2cam, (3))

  data = {
    'cam2ee' : T_cam_ee.tolist(),
    'base2gripper' : np.linalg.inv(T_gripper2base).tolist(),
    'target2cam' : T_target2cam.tolist(),
    'side_rgb' : side_rgb.tolist(),
    'side_depth' : side_depth.tolist(),
    'side_intr' : calibration.K.tolist(),
  }

  sio.savemat('tf.mat', data)
  calibration.debug()
  print('process completed.')

  
  # print(t, r)

  # rospy.sleep(2)
  # calibration.save()

