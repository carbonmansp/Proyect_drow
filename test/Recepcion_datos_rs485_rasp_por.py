import RPi.GPIO as GPIO
import serial
import time
import tkinter as tk
from tkinter import ttk
import pygame

pygame.mixer.init()

#se definen los puertos
EN_485 = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(EN_485, GPIO.OUT)
GPIO.output(EN_485, GPIO.HIGH)

# Inicializar el puerto serial
ser = serial.Serial("/dev/ttyS0", 9600)
ser.timeout = 1

# Funcion para enviar solicitud y recibir respuesta
def send_request(arduino_id):
    message = f"{arduino_id}%%REQUEST"
    GPIO.output(EN_485, GPIO.HIGH)
    ser.write(message.encode())
    time.sleep(0.1)
    GPIO.output(EN_485, GPIO.LOW)
    
    response = ser.readline().decode().strip()
    return response

class SeatMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Seat Occupancy Monitor")
        self.correct_count = 0
        self.incorrect_count = 0
        self.create_widgets()
        self.update_data()

    def create_widgets(self):
        
        self.percent_label = tk.Label(self.root, text="Percentage of correct use of seat belt: 0%")
        self.percent_label.pack(pady=10)
        
        #creacion de la ventana para mostrar el estado de los asientos
        self.tree = ttk.Treeview(self.root, columns=("Seat", "Sensor1", "Sensor2", "Sensor3", "Sensor4"), show="headings")
        self.tree.heading("Seat", text="Seat")
        self.tree.heading("Sensor1", text="Seat")
        self.tree.heading("Sensor2", text="Brooch")
        self.tree.heading("Sensor3", text="Belt")
        self.tree.heading("Sensor4", text="LED")
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.refresh_button = tk.Button(self.root, text="Refresh", command=self.update_data)
        self.refresh_button.pack(pady=10)

        # Crear las filas iniciales
        for i in range(40):
            self.tree.insert("", "end", iid=i, values=(f"Seat {i+1}", "No signal", "No signal", "No signal", "No signal"))

    def update_data(self):
        self.correct_count = 0
        self.incorrect_count = 0
        
        for i in range(1, 6):  # Iterar sobre los Arduinos del 1 al 5
            arduino_id = f"ARD{i}"
            response = send_request(arduino_id)
            
            if response:
                if response.startswith(arduino_id):
                    parts = response.split('%%')
                    if len(parts) == 2:
                        data_segments = parts[1].split(';')
                        base_seat_number = (i - 1) * 8
                        
                        for segment in data_segments:
                            if segment:
                                segment_data = segment.split(',')
                                seat_number = base_seat_number + int(segment_data[0])
                                data_asiento = "OK" if segment_data[1] == '1' else "Incorrect"
                                data_broche = "OK" if segment_data[2] == '1' else "Incorrect"
                                data_cinturon = "OK" if segment_data[3] == '0' else "Incorrect"
                                data_LED = "OFF" if segment_data[4] == '0' else "ON"
                                
                                print(f"Seat: {seat_number}")
                                print(f"Seat Data: {data_asiento}")
                                print(f"Brooch Data: {data_broche}")
                                print(f"Belt Data: {data_cinturon}")
                                print(f"Led Data: {data_LED}")

                                self.tree.item(seat_number - 1, values=(f"Seat {seat_number}", data_asiento, data_broche, data_cinturon, data_LED))

                                if data_asiento == "OK" and data_broche == "OK" and data_cinturon == "OK":
                                    self.tree.item(seat_number - 1, tags=("OK",))
                                    self.correct_count += 1
                                elif data_asiento == "OK" and (data_broche == "Incorrect" or data_cinturon == "Incorrect"):
                                    self.tree.item(seat_number - 1, tags=("Incorrect",))
                                    self.incorrect_count += 1
                                else:
                                    self.tree.item(seat_number - 1, tags=("Free",))
                    else:
                        print(f"Answer Format incorrect from {arduino_id}: {response}")
                else:
                    print(f"Unrecognized response from {arduino_id}: {response}")
            else:
                print(f"No signal from {arduino_id}")
                for j in range(8):
                    seat_number = (i - 1) * 8 + j + 1
                    self.tree.item(seat_number - 1, values=(f"Seat {seat_number}", "No signal", "No signal", "No signal", "No signal"))
                    self.tree.item(seat_number - 1, tags=("Free",))

        # Aplicar estilos a las filas
        self.tree.tag_configure("OK", background="green")
        self.tree.tag_configure("Incorrect", background="red")
        self.tree.tag_configure("Free", background="White")
        
        # Calculo de porcentaje que hacen uso correcto del cinturon
        total_asientos = self.correct_count + self.incorrect_count
        if total_asientos > 0:
            porcentaje_correcto = round((self.correct_count/total_asientos)*100)
        else:
            porcentaje_correcto = 0
        
        self.percent_label.config(text=f"Percentage of correct use of seat belt: {porcentaje_correcto}%")
        
        # Comprobacion en caso el uso correcto sea menor al 50%
        if porcentaje_correcto < 50:
            self.play_alert_message()
        else:
            print("Belts greater than: 50%")

        # Guardar los datos en un archivo de texto
        self.save_data_to_file(porcentaje_correcto)

        # Siguiente actualizacion automatica en 5 minutos
        self.root.after(5 * 60 * 1000, self.update_data)
    
    def play_alert_message(self):
        pygame.mixer.music.load("alerta_cinturon.mp3")
        pygame.mixer.music.play()
        print("Alert: Belts less than 50%")

    def save_data_to_file(self, porcentaje_correcto):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open("sensor_data.txt", "a") as file:
            file.write(f"Refresh: {timestamp}\n")
            file.write(f"{'Seat':<10} {'Sensor 1':<10} {'Sensor 2':<10} {'Sensor 3':<10} {'Sensor 4':<10}\n")
            for i in range(40):
                values = self.tree.item(i, "values")
                file.write(f"{values[0]:<10} {values[1]:<10} {values[2]:<10} {values[3]:<10} {values[4]:<10}\n")
            file.write(f"\nPercentage of correct use of seat belt: {porcentaje_correcto}%\n")
            file.write("\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = SeatMonitor(root)
    root.mainloop()

    # Cerramos el puerto serial antes de salir
    ser.close()
    GPIO.cleanup()
