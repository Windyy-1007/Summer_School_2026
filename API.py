#-------------------------
from robot import robot

robot.forward(50, 1, True)
robot.backward(50, 1, True)
robot.turn_left(50, 1)
robot.turn_right(50, 1)
robot.forward(50)
robot.backward(50)
robot.turn_left(50)
robot.turn_right(50)
robot.turn_left_angle(90)
robot.turn_right_angle(90)
robot.set_wheel_speed(50, 50)
robot.stop()

#-------------------------
from ultrasonic import ultrasonic

ultrasonic.distance_cm(1)
ultrasonic.distance_cm(1) < 10
ultrasonic.distance_cm(1) > 10
ultrasonic.distance_cm(1) == 10

#-------------------------
from line_array import line_array

line_array.read(0) == (0, 1, 1, 0)
line_array.read(0, 0)

#-------------------------
from button import btn_onboard

btn_onboard.is_pressed()


{selected_value}

#-------------------------
from utility import timer

timer.reset()
timer.get()

#-------------------------
from motion import motion

motion.get_accel('x')
motion.get_accel('y')
motion.get_accel('z')
motion.get_gyro_roll()
motion.get_gyro_pitch()
motion.get_gyro_yaw()
motion.is_shaked()

#-------------------------
import time

time.sleep(10)

#-------------------------
from utility import wait_for

wait_for(lambda:motion.is_shaked() )


while True:
  pass

for count in range(10):
  pass

while False:
  pass

for i in range(1, 11):
  pass

for j in []:
  break

#-------------------------
from robocon_xbot import *

turn_until_line_detected(-30, 30, 0, 5000, BRAKE)
turn_until_line_detected(-30, 30, 0, 5000, STOP)
turn_until_line_detected(-30, 30, 0, 5000, None)
follow_line_until(30, lambda: (False), 0, 5000)
turn_until_line_detected(30, 0, 0, 5000)
turn_until_condition(30, 0, lambda: (), 5000)
