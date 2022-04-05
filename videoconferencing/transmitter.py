from http import client
import socket
import cv2
import pickle
import struct


server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_name  = socket.gethostname()
host_ip = socket.gethostbyname(host_name)
print('HOST IP:', host_ip)
port = 1234
socket_address = ('127.0.0.1',port)
print("Socket Created Successfully")
server_socket.bind(socket_address)
print("Socket Bind Successfully")
server_socket.listen(5)
print("LISTENING AT:",socket_address)

client_socket,addr = server_socket.accept()
print('GOT CONNECTION FROM:',addr)

vid = cv2.VideoCapture(0)

while client_socket and vid.isOpened():
    try:
        img,frame = vid.read()
        a = pickle.dumps(frame)
        message = struct.pack("Q",len(a))+a
        client_socket.sendall(message)
        

        cv2.imshow('TRANSMITTING VIDEO',frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key ==ord('q'):
            break
    except:
        print("error has occured in transmitter")
client_socket.close()