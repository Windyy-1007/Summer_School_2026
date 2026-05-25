import machine
from machine import Pin, SoftI2C
from robocon_xbot import *
from setting import *
from utility import *
from micropython import const
import time
import gamepad

MODE_DPAD = const(1)
MODE_LEFT_JOYSTICK = const(2)
MODE_RIGHT_JOYSTICK = const(3)
MODE_BOTH_JOYSTICK = const(4)


class GamepadHandler:

    def __init__(self, port):

        self.port = port
        # Grove port: GND VCC SCL SDA
        scl_pin = Pin(PORTS_DIGITAL[port][0])
        sda_pin = Pin(PORTS_DIGITAL[port][1])

        try:
            self.i2c_gp = SoftI2C(scl=scl_pin, sda=sda_pin, freq=100000)

            self.gamepad = gamepad.GamePadReceiver(self.i2c_gp)

            self.gamepad._verbose = False
            self.filter_btn_data = self.gamepad.data

            self._speed = 50
            self._speed_turbo = 100

            self.drive_mode = MODE_DPAD
            self.btnChangeMode = 'm2'

            self.btnIncr = 'b'
            self.btnDecr = 'a'

            self.btnTurboMode = None

            self.btnLineFlwMode = None
            self.speedLineFlwMode = 30
            self.portLineFlwMode = 0

            self.colorVal = '#0000ff'
            self.servoVal = {}
            self.servo_last_angle = [0, 0, 0, 0, 0, 0, 0, 0]

            self.btnBallLauncherLoad = None
            self.btnBallLauncherShoot = None
            self.ballLauncherServo1 = 0
            self.ballLauncherServo2 = 1

        except:
            self.gamepad = None
            self.port = None
            say('Gamepad Receiver not found')

    def is_connected(self):
        return self.gamepad._isconnected

    def set_drive_mode(self, drive_mode):
        self.drive_mode = drive_mode

    def set_speed_btn(self, btnIncr='b', btnDecr='a'):
        self.btnIncr = btnIncr
        self.btnDecr = btnDecr

    def set_servo_btn(self, index=0, btn1='x', btn2='y', angle1=0, angle2=90, speed=100):
        if index < 0 or index > 7:
            return

        if angle1 < 0 or angle2 > 180:
            return

        if angle1 < 0 or angle2 > 180:
            return

        if speed < 0 or speed > 100:
            return

        self.servoVal[index] = [btn1, btn2, angle1, angle2, speed]

    def set_servo_fork_btn(self, index=0, btn1='x', btn2='y', angle1=0, angle2=90, speed=100):
        if index < 0 or index > 7:
            return

        if angle1 < 0 or angle2 > 180:
            return

        if angle1 < 0 or angle2 > 180:
            return

        if speed < 0 or speed > 100:
            return

        self.servoVal[index] = [btn1, btn2, angle1, angle2, speed]

    def set_ball_launcher_btn(self, index1=0, index2=1, btn1='x', btn2='y'):
        self.ballLauncherServo1 = index1
        self.ballLauncherServo2 = index2
        self.btnBallLauncherLoad = btn1
        self.btnBallLauncherShoot = btn2

    def set_change_mode_btn(self, btn='m2'):
        self.btnChangeMode = btn

    def set_turbo_btn(self, btn='r1'):
        self.btnTurboMode = btn

    def set_follow_line_btn(self, port=0, speed=30, btn=None):
        self.btnLineFlwMode = btn
        self.speedLineFlwMode = speed
        self.portLineFlwMode = port

    def set_led_color(self, color):
        if self.gamepad != None:
            self.colorVal = color
            self.gamepad.set_led_color(hex_to_rgb(self.colorVal))

    def set_rumble(self, force, duration):
        if self.gamepad != None:
            new_force = translate(force, 0, 100, 0, 255)
            new_duration = translate(duration, 0, 2000, 0, 255)
            self.gamepad.set_rumble(new_force, new_duration)

    def drive_mode_dpad(self):
        if self.gamepad.data['dpad_up'] and self.gamepad.data['dpad_left']:
            robot.move(4)
        elif self.gamepad.data['dpad_up'] and self.gamepad.data['dpad_right']:
            robot.move(2)
        elif self.gamepad.data['dpad_down'] and self.gamepad.data['dpad_left']:
            robot.move(6)
        elif self.gamepad.data['dpad_down'] and self.gamepad.data['dpad_right']:
            robot.move(8)
        elif self.gamepad.data['dpad_up']:
            robot.move(3)
        elif self.gamepad.data['dpad_down']:
            robot.move(7)
        elif self.gamepad.data['dpad_left']:
            robot.move(5)
        elif self.gamepad.data['dpad_right']:
            robot.move(1)
        else:
            robot.stop()

    # 0=left joystick, 1=right joystick
    def drive_mode_single_joystick(self, index=0):

        x, y, angle, dir, distance = self.gamepad.read_joystick(index)

        # speed = distance * self._speed  # adjust speed based on joystick drag distance

        if dir == 2:
            robot.move(4)
        elif dir == 4:
            robot.move(2)
        elif dir == 8:
            robot.move(6)
        elif dir == 6:
            robot.move(8)
        elif dir == 3:
            robot.move(3)
        elif dir == 7:
            robot.move(7)
        elif dir == 1:
            robot.move(5)
        elif dir == 5:
            robot.move(1)
        else:
            robot.stop()

    def drive_mode_both_joystick(self):
        # to be implemented

        j_left = self.gamepad.read_joystick(0)
        j_right = self.gamepad.read_joystick(1)

        new_max_speed_left = translate(j_left[4], 0, 100, 0, robot._speed)
        new_max_speed_right = translate(j_right[4], 0, 100, 0, robot._speed)
        new_max_speed = max(new_max_speed_left, new_max_speed_right)

        if j_left[3] == 3:
            if j_right[3] == 3:
                robot.forward(new_max_speed)
            elif j_right[3] == 7:
                robot.turn_right(new_max_speed/2)
            else:
                robot.set_wheel_speed(new_max_speed_left, 0)

        elif j_left[3] == 7:
            if j_right[3] == 3:
                robot.turn_left(new_max_speed/2)
            elif j_right[3] == 7:
                robot.backward(new_max_speed)
            else:
                robot.set_wheel_speed(-(new_max_speed_left), 0)

        elif j_right[3] == 3:
            if j_left[3] == 3:
                robot.forward(new_max_speed)
            elif j_left[3] == 7:
                robot.turn_left(new_max_speed/2)
            else:
                robot.set_wheel_speed(0, new_max_speed_right)

        elif j_right[3] == 7:
            if j_left[3] == 7:
                robot.backward(new_max_speed)
            elif j_left[3] == 3:
                robot.turn_right(new_max_speed/2)
            else:
                robot.set_wheel_speed(0, -(new_max_speed_right))

        if j_left[3] == 0 and j_right[3] == 0:
            robot.stop()

    def filter_btn(self, data=None):
        self.filter_btn_data[data] = (self.filter_btn_data[data] if isinstance(
            self.filter_btn_data[data], (int, float)) else 0) + 1

        if self.filter_btn_data[data] > 1:
            self.filter_btn_data[data] = 0
            #print(data, 'is filtered')
            return False
        return True

    def driving_servo(self, index, next_angle, speed=100):
        last_angle = self.servo_last_angle[index]
        # speed of movement
        sleep = translate(speed, 0, 100, 40, 0)

        if speed == 0:
            return

        # limit min/max values
        if next_angle < 0:
            next_angle = 0
        if next_angle > 180:
            next_angle = 180

        if next_angle > last_angle:
            for k in range(last_angle, next_angle):
                servo.position(index, k)
                last_angle = next_angle
                time.sleep_ms(int(sleep))
        else:
            for k in range(last_angle, next_angle, -1):
                servo.position(index, k)
                last_angle = next_angle
                time.sleep_ms(int(sleep))

        self.servo_last_angle[index] = next_angle
        # print(self.servo_last_angle)

    def process(self):
        if self.gamepad != None:

            self.gamepad.update()

            if self.is_connected():

                if self.gamepad.data[self.btnChangeMode]:
                    if self.filter_btn(self.btnChangeMode):
                        self.drive_mode = (self.drive_mode if isinstance(
                            self.drive_mode, (int, float)) else 0) + 1
                        if self.drive_mode > 4:
                            self.drive_mode = 1

                if self.drive_mode == MODE_DPAD:
                    self.drive_mode_dpad()
                    self.set_led_color('#ffa500')
                elif self.drive_mode == MODE_LEFT_JOYSTICK:
                    self.drive_mode_single_joystick(0)
                    self.set_led_color('#0000ff')
                elif self.drive_mode == MODE_RIGHT_JOYSTICK:
                    self.drive_mode_single_joystick(1)
                    self.set_led_color('#00ff00')
                elif self.drive_mode == MODE_BOTH_JOYSTICK:
                    self.drive_mode_both_joystick()
                    self.set_led_color('#800080')

                if self.gamepad.data[self.btnIncr]:
                    if self.filter_btn(self.btnIncr):
                        self._speed = (self._speed if isinstance(
                            self._speed, (int, float)) else 0) + 5
                        if self._speed > 100:
                            self._speed = 100

                if self.gamepad.data[self.btnDecr]:
                    if self.filter_btn(self.btnDecr):
                        self._speed = (self._speed if isinstance(
                            self._speed, (int, float)) else 0) - 5
                        if self._speed < 0:
                            self._speed = 0

                if self.btnTurboMode != None:
                    if self.gamepad.data[self.btnTurboMode]:
                        robot._speed = self._speed_turbo
                    else:
                        robot._speed = self._speed
                else:
                    robot._speed = self._speed

                if self.btnLineFlwMode != None:
                    if self.gamepad.data[self.btnLineFlwMode]:
                        follow_line(self.speedLineFlwMode,
                                    self.portLineFlwMode)

                if (self.btnBallLauncherLoad != None) or (self.btnBallLauncherShoot != None):
                    if self.gamepad.data[self.btnBallLauncherLoad]:
                        if self.filter_btn(self.btnBallLauncherLoad):
                            ball_launcher(self.ballLauncherServo1,
                                          self.ballLauncherServo2, mode=0)

                    if self.gamepad.data[self.btnBallLauncherShoot]:
                        if self.filter_btn(self.btnBallLauncherShoot):
                            ball_launcher(self.ballLauncherServo1,
                                          self.ballLauncherServo2, mode=1)

                for i in range(len(self.servoVal)):
                    if self.gamepad.data[self.servoVal[i][0]]:
                        self.driving_servo(
                            i, self.servoVal[i][2], self.servoVal[i][4])
                        #servo.position(i, self.servoVal[i][2])
                    if self.gamepad.data[self.servoVal[i][1]]:
                        self.driving_servo(
                            i, self.servoVal[i][3], self.servoVal[i][4])
                        #servo.position(i, self.servoVal[i][3])

            # print('Mode: ',self.drive_mode ,'Speed: ', robot._speed)
            time.sleep_ms(10)


#gamepad_handler = GamepadHandler(3)
