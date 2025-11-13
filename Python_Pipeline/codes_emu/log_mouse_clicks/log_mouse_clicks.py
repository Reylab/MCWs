# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 12:48:40 2021

@author: Fernando J. Chaure


to package with pyinstaller, set the version of pynput:
    pip install pynput==1.6.8

in one file with:
    pyinstaller log_mouse_clicks.py -F
"""
from pynput import mouse
import time 
from winsound import Beep


button2track = mouse.Button.left
data = []
init_time = time.time()
def on_click(x, y, button, pressed):
    if (button2track == button) and  pressed:
        t = time.time()
        Beep(500, 100)
        data.append([t-init_time, x, y])
        

listener = mouse.Listener(on_click=on_click)
listener.start()
file_name = time.asctime(time.localtime(init_time)).replace(':','-')+'.csv'

print('filename : {}'.format(file_name))

key = ''
while key != 'q':
    key = input('Press \'q\' to end script: ')
listener.stop()
print('saving....')

with open(file_name, 'wt') as f:
    f.write('time(sec), x, y\n')
    for t,x,y in data:
        f.write('{}, {}, {}\n'.format(t,x,y))
    