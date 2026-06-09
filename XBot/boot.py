# This file is executed on every boot (including wake-boot from deepsleep)
import os, time, gc, esp, sys
from machine import Pin, reset
from neopixel import NeoPixel

from setting import *
from utility import *

esp.osdebug(None)

if Pin(BTN_A_PIN, Pin.IN, Pin.PULL_UP).value() == 0:
  print('Button A is pressed during boot')
  press_begin = time.ticks_ms()
  press_duration = 0

  while Pin(BTN_A_PIN, Pin.IN, Pin.PULL_UP).value() == 0:
    press_end = time.ticks_ms()
    press_duration = time.ticks_diff(press_end, press_begin)

    if press_duration > 2000:
      print('Button A pressed longer than 2 seconds during boot')
      print('Remove main.py...')
      try:
          os.remove('main.py')
      except OSError:  # remove failed
          pass

      try:
          os.stat('main.py')
          print('...failed.')
      except OSError:  # stat failed
          print('...OK.')
      
      print('Reset now...')

      np = NeoPixel(Pin(RGB_LED_PIN), 25)

      for i in range(3):
        np.fill((50,0,0))
        np.write()
        time.sleep_ms(200)
        np.fill((0,0,0))
        np.write()
        time.sleep_ms(200)
      reset()

    time.sleep_ms(100)

def stop_all():
  pass

def run(mod):
  if mod in sys.modules:
    del sys.modules[mod]
  __import__(mod)

while True:
  try:

    if DEV_VERSION >= 4:
      try:
        run('bleuart')
        run('blerepl')
        run('ble')
        from ble import ble_o, ble  
        ble.start()
      except Exception as err:
        print('Failed to start Bluetooth: ')
        print(err)
    else:
      print('This device does not support Bluetooth: ')

    gc.collect()
    print('Yolo:Bit firmware version',  VERSION)
    break
  except KeyboardInterrupt as err:
    print('Device is booting')


__main_exists = False

try:
    os.stat('main.py')
    print('User program exists. No need to run default program')
    __main_exists = True
except OSError:  # stat failed
    pass

if not __main_exists:
  # boot animation
  np = NeoPixel(Pin(RGB_LED_PIN), 25)


  def screen_show(image):
    global np
    it = iter(image)
    for r in range(25):
        col = next(it)
        np[r] = (55, 0, 0) if col else (0, 0, 0)
    np.write()

  full = [0] * 25
  np.fill((0,0,0))
  np.write()
  time.sleep_ms(150)

  d = 5
  d2 = 2
  x = 2
  y = 2
  for i in range((d2 + 1) * 2 - d, d + 1, 2):
      xx = x - (i-1)
      yy = y - (i-1)
      x1 = x
      y1 = y
      xx1 = xx
      yy1 = yy

      for j in range(i):
          y1 = y - j
          yy1 = yy + j
          full[x1 * 5 + y1] = 1
          full[xx1 * 5 + yy1] = 1
          screen_show(full)
          time.sleep_ms(150)
      for j in range(1, i-1, 1):                
          x1 = x - j
          xx1 = xx + j
          full[x1 * 5 + y1] = 1
          full[xx1 * 5 + yy1] = 1
          screen_show(full)
          time.sleep_ms(150)
      x += 1
      y += 1
  
  np.fill((0,0,0))
  np.write()
  del np, x, y, d, d2, screen_show, full
