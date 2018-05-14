import serial
import time
import cv2


# Эта часть кода делает фоточки
def make_photo():
    camera = cv2.VideoCapture(0)
    for i in range(10):
        return_value, image = camera.read()
        cv2.imwrite('opencv'+str(i)+'.png', image)
    del camera


# ----------------------------------
def serial_input(ser):
    state = b'0'
    trash = ser.readline()
    print("Message: ", trash)
    state = ser.read()
    print(state)
    if state == b'1':
        #тут делаем фотографию
        make_photo()
        print("We got a photo!")
    # В конце закрываем порт


if __name__ == "__main__":
    # открываем порт
    ser = serial.Serial('COM3', 9600)
    print("Port is open")
    while True:
        serial_input(ser)
