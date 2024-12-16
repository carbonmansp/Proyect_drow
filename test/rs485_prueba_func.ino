#include <SoftwareSerial.h>

// MAX485 control pin (La comunicación solo puede ser SemiDuplex)
#define RS485EN 2
#define pinTX 9
#define pinRX 8
#define LED_PIN 13

SoftwareSerial RS485Serial(pinRX, pinTX); // RX, TX

String ARD_NAME="ARD1"; // Cambiar a "ARD2", "ARD3", etc., según sea necesario
boolean Data_Asiento[8] = {1, 0, 1, 1, 1, 1, 1, 1};
boolean Data_Broche[8] = {1, 0, 1, 1, 0, 0, 1, 1};
boolean Data_Cinturon[8] = {0, 1, 0, 0, 0, 1, 1, 1};
boolean Data_LED[8] = {0, 0, 0, 0, 1, 1, 0, 0};
//Los datos estan en el siguiente orden sensor[8] = {asiento 1, asiento 2, asiento 3, asiento 4, asiento 5, asiento 6, asiento 7, asiento 8}
//Para hacer la prueba con la raspberry pi 4 es necesario modificar los datos booleanos de broche, asiento, cinturon y led.

void setup() {
  Serial.begin(9600);
  RS485Serial.begin(9600);
  pinMode(LED_PIN, OUTPUT);
  pinMode(RS485EN, OUTPUT);

  digitalWrite(RS485EN, LOW); // Configura RS485 como Rx
}

void loop() {
  if(RS485Serial.available() > 0) {
    String mensaje = RS485Serial.readStringUntil('\n'); // Lee el mensaje recibido
    mensaje.trim();   // Elimina los espacios en blanco al inicio y al final del String
    
    if(mensaje == ARD_NAME + "%%REQUEST") {
      digitalWrite(RS485EN, HIGH);  // Coloca el arduino en modo Transmisión
      
      // Enviar todos los datos de los arrays en el formato requerido
      String text_data = ARD_NAME + "%%";
      for (byte i = 0; i < 8; i++) {
        text_data += String(i + 1, DEC) + "," + 
                     String(Data_Asiento[i], BIN) + "," + 
                     String(Data_Broche[i], BIN) + "," + 
                     String(Data_Cinturon[i], BIN) + "," + 
                     String(Data_LED[i], BIN) + ";";
      }
      RS485Serial.println(text_data);
      digitalWrite(RS485EN, LOW);  // Coloca el arduino en modo Recepción
    }
  }
}
