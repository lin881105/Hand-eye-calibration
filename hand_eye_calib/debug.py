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

  def dataloader(self):
    mat_file = 'data.mat'
    mat_dict = sio.loadmat(mat_file)
    print(mat_dict.keys())
    
    self.R_gripper2base = mat_dict['R_gripper2base']
    self.t_gripper2base = mat_dict['t_gripper2base']

    self.R_target2cam = mat_dict['R_target2cam']
    self.t_target2cam = mat_dict['t_target2cam']

  

  def solve(self):
    print(len(self.R_gripper2base),len(self.t_gripper2base),len(self.R_target2cam),len(self.t_target2cam))
    self.R, self.T = cv2.calibrateHandEye(self.R_gripper2base,self.t_gripper2base,self.R_target2cam,self.t_target2cam)

    return self.T,self.R

  

if __name__ == '__main__':
  calibration = Calibration()
  calibration.dataloader()
  r,t = calibration.solve()
  print(r,t)


  
  # print(t, r)

  # rospy.sleep(2)
  # calibration.save()

