# -*- encoding: utf-8 -*-
"""
    @Project: KRQA - 副本.py
    @File   : pdf_.py
    @Author : ZHul
    @E-mail : zl2870@qq.com
    @Data   : 2023/5/27  19:12
"""
# from PIL import ImageGrab
# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter
# from reportlab.lib.units import inch
# from PIL import Image
#
# im = ImageGrab.grab()
# im = ImageGrab.grab(bbox=(100, 100, 500, 500))
# c = canvas.Canvas("screenshot.pdf", pagesize=letter)
# # im = Image.open("screenshot.png")
# # width, height = im.size
# # c.drawImage("screenshot.png", 0, 0, width=width/inch, height=height/inch)
# c.save()

import pyautogui
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
# 获取屏幕的大小
screen_width, screen_height = pyautogui.size()
console_x=150
console_y=625
console_width=2000
console_height=805
time.sleep(5)
# 截取整个屏幕
screenshot = pyautogui.screenshot(region=(console_x, console_y, console_width, console_height))

# 保存截图为临时文件
temp_file = 'screenshot.png'
screenshot.save(temp_file)

# 打开截图文件
image = Image.open(temp_file)

# 创建PDF文件
pdf_file = 'screenshot.pdf'
c = canvas.Canvas(pdf_file, pagesize=letter)

# 获取图像的尺寸
img_width, img_height = image.size

# 设置PDF页面尺寸为图像尺寸
c.setPageSize((img_width, img_height))

# 将图像绘制到PDF页面上
c.drawImage(temp_file, 0, 0, width=img_width, height=img_height)

# 保存PDF文件
c.showPage()
c.save()
