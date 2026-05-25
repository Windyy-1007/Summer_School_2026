import time
from micropython import const
from machine import Pin, SoftI2C
from utility import *
from setting import *
from ble import *
from constants import *
import gamepad

GAMEPAD_RECEIVER_ADDR = const(0x55)

# gamepad buttons
# BTN_UP = const(0)
# BTN_DOWN = const(1)
# BTN_LEFT = const(2)
# BTN_RIGHT = const(3)
# 
# BTN_A = 'A'
# BTN_B = 'B'
# BTN_C = 'C'
# BTN_D = 'D'
# 
# BTN_SQUARE = const(4)
# BTN_TRIANGLE = const(5)
# BTN_CROSS = const(6)
# BTN_CIRCLE = const(7)
# 
# BTN_L1 = const(8)
# BTN_R1 = const(9)
# BTN_L2 = const(10)
# BTN_R2 = const(11)
# 
# BTN_M1 = ''
# BTN_M2 = ''
# BTN_THUMBL = ''
# BTN_THUMBR = ''
# 
# AL = ''
# ALX = ''
# ALY = ''
# AL_DIR = ''
# AL_DISTANCE = ''
# AR = ''
# ARX = ''
# ARY = ''
# AR_DIR = ''
# AR_DISTANCE = ''

BTN_RELEASED = ''

MOVE1 = const(0)
MOVE2 = const(1)
MOVE3 = const(2)
MOVE4 = const(3)
MOVE5 = const(4)
MOVE6 = const(5)
MOVE7 = const(6)
MOVE8 = const(7)


class RemoteControlMode():

    def __init__(self, port):
        self.data = {
            BTN_UP: 0,
            BTN_DOWN: 0,
            BTN_LEFT: 0,
            BTN_RIGHT: 0,
            BTN_SQUARE: 0,
            BTN_TRIANGLE: 0,
            BTN_CROSS: 0,
            BTN_CIRCLE: 0,
            BTN_L1: 0,
            BTN_R1: 0,
            BTN_L2: 0,
            BTN_R2: 0,
            BTN_THUMBL: 0,
            BTN_THUMBR: 0,
            BTN_M1: 0,
            BTN_M2: 0,
            AL: 0,
            ALX: 0,
            ALY: 0,
            AL_DIR: -1,
            AL_DISTANCE: 0,
            AR: 0,
            ARX: 0,
            ARY: 0,
            AR_DIR: -1,
            AR_DISTANCE: 0,
        }

        # remote control
        self._cmd = ''
        self._last_cmd = ''
        self._speed = 30
        self._cmd_handlers = {
            BTN_UP: None,
            BTN_DOWN: None,
            BTN_LEFT: None,
            BTN_RIGHT: None,

            BTN_SQUARE: None,
            BTN_TRIANGLE: None,
            BTN_CROSS: None,
            BTN_CIRCLE: None,

            BTN_L1: None,
            BTN_R1: None,
            BTN_L2: None,
            BTN_R2: None,

            BTN_M1: None,
            BTN_M2: None,
            BTN_THUMBL: None,
            BTN_THUMBR: None,
            }

        self.port = port
        # Grove port: GND VCC SCL SDA
        scl_pin = Pin(PORTS_DIGITAL[port][0])
        sda_pin = Pin(PORTS_DIGITAL[port][1])
        
        self._i2c_gp = SoftI2C(scl=scl_pin, sda=sda_pin, freq=100000)
        if self._i2c_gp.scan().count(GAMEPAD_RECEIVER_ADDR) == 0:
            self._gamepad_v2 = None
            print('Gamepad V2 Receiver not found {:#x}'.format(GAMEPAD_RECEIVER_ADDR))
        else:
            self._gamepad_v2 = gamepad.GamePadReceiver(self._i2c_gp)
        
        ble.on_receive_msg('name_value', self.on_ble_cmd)

    def on_ble_cmd(self, name, value):
#         print(name + '=' + str(value))
        if name not in list(self.data.keys()):
            print('Invalid remote control command')
            return
        self.data[name] = value
        self._cmd = ''
        if name == AL:
            ax, ay, angle, dir, distance = self._gamepad_v2.read_joystick(0)
            self.data[ALX] = ax
            self.data[ALY] = ay
            self.data[AL_DIR] = dir
            self.data[AL_DISTANCE] = distance
        elif name == AR:
            ax, ay, angle, dir, distance = self._gamepad_v2.read_joystick(1)
            self.data[ARX] = ax
            self.data[ARY] = ay
            self.data[AR_DIR] = dir
            self.data[AR_DISTANCE] = distance
            
            
        if value == 1:
            if name == BTN_UP:
                self._cmd = BTN_UP
            elif name == BTN_DOWN:
                self._cmd = BTN_DOWN
            elif name == BTN_LEFT:
                self._cmd = BTN_LEFT
            elif name == BTN_RIGHT:
                self._cmd = BTN_RIGHT
            elif name == BTN_SQUARE:
                self._cmd = BTN_SQUARE
            elif name == BTN_TRIANGLE:
                self._cmd = BTN_TRIANGLE
            elif name == BTN_CIRCLE:
                self._cmd = BTN_CIRCLE
            elif name == BTN_CROSS:
                self._cmd = BTN_CROSS
            elif name == BTN_L1:
                self._cmd = BTN_L1
            elif name == BTN_R1:
                self._cmd = BTN_R1
            else:
                self._cmd = BTN_RELEASED
        
    def set_command(self, cmd, handler):
        if cmd not in self._cmd_handlers:
            print('Invalid remote control command')
            return

        self._cmd_handlers[cmd] = handler
        
        
    def run(self):
        # read command from gamepad v2 receiver if connected
        if self._gamepad_v2 != None:
            # read status
            x, y, angle, dir, distance = self._gamepad_v2.read_joystick(0)
            self._gamepad_v2.update()
            
            if self._gamepad_v2._isconnected == True:
                if self._gamepad_v2.data['dpad_up']:
                    self._cmd = BTN_UP
                elif self._gamepad_v2.data['dpad_down']:
                    self._cmd = BTN_DOWN
                elif self._gamepad_v2.data['dpad_left']:
                    self._cmd = BTN_LEFT
                elif self._gamepad_v2.data['dpad_right']:
                    self._cmd = BTN_RIGHT
                elif self._gamepad_v2.data['a']:
                    self._cmd = BTN_CROSS
                elif self._gamepad_v2.data['b']:
                    self._cmd = BTN_CIRCLE
                elif self._gamepad_v2.data['x']:
                    self._cmd = BTN_SQUARE
                elif self._gamepad_v2.data['y']:
                    self._cmd = BTN_TRIANGLE
                elif self._gamepad_v2.data['l1']:
                    self._cmd = BTN_L1
                elif self._gamepad_v2.data['r1']:
                    self._cmd = BTN_R1
                elif self._gamepad_v2.data['l2']:
                    self._cmd = BTN_L2
                elif self._gamepad_v2.data['r2']:
                    self._cmd = BTN_R2
                elif self._gamepad_v2.data['m1']:
                    self._cmd = BTN_M1
                elif self._gamepad_v2.data['m2']:
                    self._cmd = BTN_M2
                elif self._gamepad_v2.data['thumbl']:
                    self._cmd = BTN_THUMBL
                elif self._gamepad_v2.data['thumbr']:
                    self._cmd = BTN_THUMBR
                elif dir == 5:
                    self._cmd = MOVE5
                elif dir == 4:
                    self._cmd = MOVE4
                elif dir == 3:
                    self._cmd = MOVE1
                elif dir == 2:
                    self._cmd = MOVE2
                elif dir == 1:
                    self._cmd = MOVE3
                elif dir == 8:
                    self._cmd = MOVE8
                elif dir == 7:
                    self._cmd = MOVE7
                elif dir == 6:
                    self._cmd = MOVE6
                else:
                    self._cmd = BTN_RELEASED

        
        if self._cmd != self._last_cmd: # got new command
            self._speed = 30 # reset speed
        else:            
            if self._speed < 50:
                self._speed = self._speed + 1
                
            else:
                self._speed = 50

        if self._cmd_handlers.get(self._cmd) != None:
            self._cmd_handlers[self._cmd]()
            
        elif self._cmd == BTN_UP:
            s = self._speed * 2
            if s > 80:
                s = 80
            robot.forward(s)

        elif self._cmd == BTN_DOWN:
            s = self._speed * 2
            if s > 80:
                s = 80
            robot.backward(s)

        elif self._cmd == BTN_LEFT:
            robot.turn_left(self._speed)

        elif self._cmd == BTN_RIGHT:
            robot.turn_right(self._speed)
        
        elif self._cmd in self._cmd_handlers:
            if self._cmd_handlers[self._cmd] != None:
                self._cmd_handlers[self._cmd]()
        elif self._cmd == MOVE1:
            robot.turn_left(self._speed)
            
        elif self._cmd == MOVE5:
            robot.turn_right(self._speed)
            
        elif self._cmd == MOVE3:
            robot.forward(self._speed)
            
        elif self._cmd == MOVE7:
            robot.backward(self._speed)
            
        elif self._cmd == MOVE2:
            robot.set_wheel_speed(self._speed/2, self._speed)
            
        elif self._cmd == MOVE4:
            robot.set_wheel_speed(self._speed, self._speed/2)
            
        elif self._cmd == MOVE6:
            robot.set_wheel_speed(-(self._speed), - self._speed/2)
            
        elif self._cmd == MOVE8:
            robot.set_wheel_speed(-self._speed/2, -self._speed)

        else:
            robot.stop()
        
        self._last_cmd = self._cmd

    def read_gamepad(self, data):
        if self._gamepad_v2 != None and self._gamepad_v2._isconnected == True:
            return self._gamepad_v2.data[data]
        else:
            return 0

''' 

# Example code

def on_gamepad_button_A():
    # button A: lift down and release gripper
    robot.servo_write(2, 0)
    time.sleep_ms(500)
    robot.servo_write(1, 0)

def on_gamepad_button_D():
    # button D: collect and lift up gripper
    robot.servo_write(1, 90)
    time.sleep_ms(500)
    robot.servo_write(2, 90)

# allow user to config what to do when a gamepad button pressed
rc_mode.set_command(BTN_A, on_gamepad_button_A)
rc_mode.set_command(BTN_D, on_gamepad_button_D)

while True:
    rc_mode.run()
    time.sleep_ms(50)

'''
