#inicialización de complementos
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pygame
import time
import torch
import torch.nn as nn
import datetime
import os
from torch.utils.data import DataLoader, TensorDataset

# Filtro de Kalman
class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimation_error, initial_value=0):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimation_error = estimation_error
        self.estimate = initial_value
        self.kalman_gain = 0

    def update(self, measurement):
        self.kalman_gain = self.estimation_error / (self.estimation_error + self.measurement_variance)
        self.estimate += self.kalman_gain * (measurement - self.estimate)
        self.estimation_error = (1 - self.kalman_gain) * self.estimation_error + abs(self.estimate) * self.process_variance
        return self.estimate
#Definición para guardado de datos
def guardar_dat(mensaje, duracion):
    fecha_hora_actual = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('registro_incidentes.txt', 'a') as archivo:
        archivo.write(f"{fecha_hora_actual} - {mensaje} - Duración: {duracion} seg\n")
#definición para dibujado de contornos, contadores como parpadeo, bostezo y cabeceo.
def drawing_output(frame, coord_left_eye, coord_right_eye, blink_counter, cont_sue, boste, theta):
    aux_image = np.zeros(frame.shape, np.uint8)
    contours1 = np.array([coord_left_eye])
    contours2 = np.array([coord_right_eye])
    cv2.fillPoly(aux_image, pts=[contours1], color = (255, 0 , 0))
    cv2.fillPoly(aux_image, pts=[contours2], color = (255, 0 , 0))
    output = cv2.addWeighted(frame, 1, aux_image, 0.7, 1)
    cv2.putText(output, "Blinking:", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(output, "{}".format(blink_counter), (220,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128,0,250), 2)
    cv2.putText(output, "Count drowsiness:",(10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(output, "{}".format(cont_sue), (240,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128,0,250), 2)
    cv2.putText(output, "Yawn:", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(output, "{}".format(boste), (220,85), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128,0,250), 2)
    cv2.putText(output, "Pitching:", (10,105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.putText(output, "{}".format(theta), (220, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (128,0,255),2)
    return output
#definición de el ratio de los labios    
def lips_ratio_ang(coordinates):
    d_a=np.linalg.norm(np.array(coordinates[1]) - np.array(coordinates[2]))
    d_b=np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[3]))
    return np.arctan(d_a/d_b)
#definición de EAR
def eye_aspect_ratio(coordinates):
    d_a = np.linalg.norm(np.array(coordinates[2]) - np.array(coordinates[10])) #D_A
    d_b = np.linalg.norm(np.array(coordinates[4]) - np.array(coordinates[8])) #D_B
    d_c = np.linalg.norm(np.array(coordinates[0]) - np.array(coordinates[6])) #D_C
    return (d_a + d_b) / (2 * d_c)
#Calculo del angulo de inclinacion (Cabeceo)
def calc_angulo(coordinates):
    d_a = coordinates[1][0] - coordinates[0][0] #x
    d_b = coordinates[1][1] - coordinates[0][1] #y
    angulo = np.arctan2(d_b,d_a) * 180.0 / np.pi
    return angulo
#Comprobar que tipo de camara si es a color o IR
def type_camera(frame):
    camera_type = None
        # Determinar si la imagen es en blanco y negro o a color
    if len(frame.shape) == 2:  # Si la imagen tiene un canal (grayscale)
        camera_type = "Cámara IR"
    elif len(frame.shape) == 3:
        b, g, r = cv2.split(frame)
        if np.array_equal(b, g) and np.array_equal(g, r):
            camera_type = "Cámara IR"
        else:
            camera_type = "Cámara Normal"
    return camera_type, frame
#Procesado de imagen segun el tipo de camara
def process_image(camera_type, image):
    if camera_type == "Cámara Normal":
        # Convertir de BGR a RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_image
    elif camera_type == "Cámara IR":
        # Convertir de blanco y negro (grayscale) a RGB
        if len(image.shape) == 2:  # Ya es escala de grises
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            # Si tiene 3 canales pero es blanco y negro (IR)
            b, g, r = cv2.split(image)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        return rgb_image  

# Red LSTM en PyTorch
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Usar la última salida de LSTM
        return out

# Crear el modelo LSTM
input_size = 1  # EAR y ángulo de cabeceo
hidden_size = 64
output_size = 1
model = LSTMNetwork(input_size, hidden_size, output_size)

# Comprobar si el archivo del modelo existe, si no, se creará uno nuevo
model_file = "ear_lstm_model.pth"
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file))
    model.eval()  # Cambiar a modo evaluación si solo vas a predecir
    print("Modelo cargado correctamente.")
else:
    print("No se encontró un modelo previo. Se entrenará un nuevo modelo.")
    
# Optimizador y criterio de pérdida
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Captura de video o carga de video
Video= 'E:\prob/Pruebacabos.mp4'
cap = cv2.VideoCapture(Video) #inicializamos la captura de video
mp_mesh = mp.solutions.face_mesh #inicialilzamos face mesh (Enmallado)
pygame.mixer.init() #inicializamos el modulo para el sonidog

#inicialización del sonido y carga del archivo
sound = pygame.mixer.Sound("E:\prob/alarm.wav") #Se carga el archivo de sonido

# Ajustes para el filtro de Kalman
kalman_filter = KalmanFilter(process_variance=1e-20, measurement_variance=1e-18, estimation_error=1)

#Control de somnolencia: Parametro de los ojos
index_face_left_eye = [33,161,160,159,158,157,133,154,153,145,144,163]
index_face_right_eye = [362,384,385,386,387,388,263,390,373,374,380,381]
ear_thresh = 0.22 #Limite de apertura en los ojos

#Inicio de contadores para detección de somnolencia ojos
num_frame = 12
aux_counter = 0
blink_counter = 0
parpadeo = False   
pts_ear = deque(maxlen=64)
tiempo_in = 0
tiempo_fin = 0
cont_sue = 0
duracion_ms=0
last_update_time = time.time()
update_interval = 5  # Tiempo en segundos para actualizar el umbral
ear_list = []

#Control de somnolencia: Parametro de Bostezo
index_lips=[62,0,17,292] #Parametro para hallar el angulo con arcotangente
aper_lips=0.76 #Limite de apertura de los labios

#Inicio de contadores para detección de somnolencia bostezo
pts_lips = deque(maxlen=64)
time_in2=0
time_fin2=0
duracion_ms2=0
boste=0
bostezo = False

#Control de somnolencia: Parametro del cabeceo 
#(Encontrar el punto de la Nariz y pomulo)
index_head= [1, 366]
cabeza = False
umbral_cab = -15
time_in3=0
time_fin3=0
duracion_ms3=0
num_frame2 = 90
aux_counter2 = 0

# Función para entrenar el modelo
def train_model(model, data, targets, epochs=10):
    model.train()
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for inputs, target in dataloader:
            optimizer.zero_grad()
            output = model(inputs.view(-1, 1, input_size))
            loss = criterion(output, target.view(-1, 1))
            loss.backward()
            optimizer.step()

with mp_mesh.FaceMesh(static_image_mode = False,
                      max_num_faces=1) as face_mesh:
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame,1)
        height, widht, _= frame.shape
        #inicialización para detectar el tipo de camara y procesar el video
        camera_type, frame = type_camera(frame)
        frame_rgb = process_image(camera_type, frame)
        #Aplicación del uso de Face Mesh
        results = face_mesh.process(frame_rgb) #Aplicación de la Malla
        #Almacenaje de coordenadas
        coord_left_eye = []
        coord_right_eye = []
        coord_lips = []
        coord_head = []    
            
        #Inicio para la detección de rostro
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                #Puntos para encontrar los ojos
                for index in index_face_left_eye:
                    x = int(face_landmarks.landmark[index].x * widht)
                    y = int(face_landmarks.landmark[index].y * height)
                    coord_left_eye.append([x, y])
                    cv2.circle(frame,( x, y),1,(0,255,255),1)
                    cv2.circle(frame,( x, y),1,(128,0,250),1)
                for index in index_face_right_eye:
                    x = int(face_landmarks.landmark[index].x * widht)
                    y = int(face_landmarks.landmark[index].y * height)
                    coord_right_eye.append([x, y])
                    cv2.circle(frame,( x, y),1,(128,0,250),1)
                    cv2.circle(frame,( x, y),1,(0,255,255),1)
                #Puntos para encontrar los labios.
                for index in index_lips:
                    x = int(face_landmarks.landmark[index].x * widht)
                    y = int(face_landmarks.landmark[index].y * height)
                    coord_lips.append([x, y])
                    cv2.circle(frame,( x, y),1,(0,255,255),1)
                    cv2.circle(frame,( x, y),1,(128,0,250),1)
                for index in index_head:
                    x = int(face_landmarks.landmark[index].x * widht)
                    y = int(face_landmarks.landmark[index].y * height)
                    coord_head.append([x, y])
                    cv2.circle(frame,( x, y),1,(0,120,255),1)
                    cv2.circle(frame,( x, y),1,(0,120,255),1)
            
            #VALORES DE APERTURA LABIOS, OJOS Y CABECEO        
            lips_ang = lips_ratio_ang(coord_lips)
            ear_left = eye_aspect_ratio(coord_left_eye)
            ear_right = eye_aspect_ratio(coord_right_eye)
            theta = calc_angulo(coord_head)
            #VALOR MEDIO DE APERTURA DE OJOS
            ear = (ear_left + ear_right)/2
            smoothed_ear = kalman_filter.update(ear)
            ear_list.append(smoothed_ear)
            
            # Preparar datos para el modelo
            if len(ear_list) >= 10:  # Entrenar el modelo cada 10 muestras
                inputs = torch.tensor(ear_list[-10:], dtype=torch.float32).view(-1, 1, input_size)  # [batch, seq_len, input_size]
                targets = torch.tensor([np.mean(ear_list[-10:])] * len(inputs), dtype=torch.float32).view(-1, 1)  # Usar el promedio como target
                train_model(model, inputs.numpy(), targets.numpy(), epochs=1)  # Entrenar el modelo
                # Limpiar la lista de EAR después de entrenar
                ear_list = []

            # Ajustar umbral cada 5 segundos
            current_time = time.time()
            if current_time - last_update_time >= update_interval:
                EAR_THRESHOLD = 0.80 * np.mean(ear_list) if ear_list else smoothed_ear  # Promedio de EAR de las predicciones
                last_update_time = current_time  # Reiniciar el temporizador
                print(f'EAR_THRESHOLD actualizado: {EAR_THRESHOLD:.2f}')
            
            #Condición de apertura de labios para la detección de bostezo
            if lips_ang > aper_lips and bostezo == False: 
                bostezo = True
                time_in2=time.time()
                
            else:
                if lips_ang < aper_lips and bostezo == True:
                    time_fin2 = time.time()
                    duracion_ms2 = (time_fin2 - time_in2)
                    bostezo = False
                    print("Duración Bostezo:", duracion_ms2)    
                    if duracion_ms2 >= 5:
                        boste += 1
                        sound.play()
                        #time.sleep(1)
                        guardar_dat("Bostezo", duracion_ms2)
                        print("PELIGRO DE BOSTEZO")
                        tiempo_fin2=0
                        time_in2=0
                        cont_sue += 1
                    duracion_ms2 = 0
                    
            #Condición de apertura de ojos para detección de pestañeo    
            if ear < ear_thresh:
                aux_counter +=1
                print("aux", aux_counter)
            
            if aux_counter > num_frame:
                sound.play()
                #time.sleep(1)
                print("PELIGRO SOMNOLENCIA")
                aux_counter=0  
            
            sound.stop()
            
            if ear < ear_thresh and parpadeo==False:
                parpadeo = True
                tiempo_in = time.time()
                blink_counter +=1
                
            else:
                
                if ear > ear_thresh and parpadeo == True:
                    parpadeo = False
                    aux_counter=0        
                    tiempo_fin = time.time()                    
                    duracion_ms = (tiempo_fin - tiempo_in)
                    print("OJOS CERRADOS:", duracion_ms)
                
                if duracion_ms >= 3:
                    cont_sue += 1
                    guardar_dat("Microsueño", duracion_ms)
                    tiempo_in = 0
                    tiempo_fin = 0
                    
                duracion_ms=0
            
            #Condición para la detección del cabeceo, si el angulo es menor a 0.    
            if theta < umbral_cab and cabeza == False: 
                cabeza = True
                time_in3=time.time()
                
            else:
                if theta > umbral_cab and cabeza == True:
                    time_fin3 = time.time()
                    duracion_ms3 = (time_fin3 - time_in3)
                    aux_counter2=0 
                    cabeza = False
                    print("Duración Cabeceo:", duracion_ms3)
                    if duracion_ms3 >= 3:
                        cont_sue += 1
                        #sound.play()
                        #time.sleep(1)
                        guardar_dat("Cabeceo", duracion_ms3)
                        print("PELIGRO CABECEO")
                        tiempo_fin3=0
                        time_in3=0
                    duracion_ms3 = 0
            #Condición para detección del cabeceo, para la activación de alarma.        
            if theta < umbral_cab:
                aux_counter2 +=1
                print("aux2", aux_counter2)
            
            if aux_counter2 > num_frame2:
                sound.play()
                #time.sleep(1)
                print("PELIGRO DE SOMNOLENCIA")
                aux_counter2=0  
            
            sound.stop()
            
            #Dibujado de contornos, conteo y muestra de valores en el video mostrado.
            frame = drawing_output(frame, coord_left_eye, coord_right_eye, blink_counter, cont_sue, boste, theta)
            pts_lips.append(lips_ang)
            pts_ear.append(ear)
        
        #Muestrar Video    
        cv2.imshow("ROSTRO CONDUCTOR", frame)
        
        torch.save(model.state_dict(), "ear_lstm_model.pth")
        
        k = cv2.waitKey(1) & 0XFF
        if k == 27:
            break
        
cap.release()
cv2.destroyAllWindows()

# Prepara los datos para entrenar la red neuronal LSTM
ear_tensor = torch.tensor(ear_list, dtype=torch.float32).unsqueeze(-1)
train_data = torch.cat((ear_tensor,), dim=1).unsqueeze(0)  # Forma [1, secuencia, características]
