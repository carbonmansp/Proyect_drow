#include <Arduino.h>
#include "Mux.h"
#include "SoftwareSerial.h"
String ARD_NAME = "ARD1";    //Nombre del Arduino esclavo
using namespace admux;

unsigned int Data_Asiento[8] = {0 ,0 ,0 ,0 ,0 ,0 ,0 ,0};    //Variable de almacenamiento de valores de entrada de sensor de Asiento
boolean Data_Broche[8] = {0 ,0 ,0 ,0 ,0 ,0 ,0 ,0};         //Variable de almacenamiento de valores de entrada de sensor de Broche
boolean Data_Cinturon[8] = {0 ,0 ,0 ,0 ,0 ,0 ,0 ,0};       //Variable de almacenamiento de valores de entrada de sensor de Cintuón
boolean Data_LED[8] = {0 ,0 ,0 ,0 ,0 ,0 ,0 ,0};            //Variable de almacenamiento de resultado de procesado de datos de entrada
String text_data;   //Variable para enviar datos a Raspberry
String text_data_temp;

//Definición de pines para comunicación RS485 mediante Modulo MAX485
#define RS485EN 2
#define PinTx 0
#define PinRx 1
SoftwareSerial RS485Serial(PinTx, PinRx);

//Contrucción de Multiplexores de entrada
Mux Mux_Asiento(Pin(A0, INPUT, PinType::Analog), Pinset(9, 10, 11, 12)); //Mux sensores asiento
Mux Mux_Broche(Pin(7, INPUT_PULLUP, PinType::Digital), Pinset(9, 10, 11, 12)); //Mux sensores broche
Mux Mux_Cinturon(Pin(6, INPUT_PULLUP, PinType::Digital), Pinset(9, 10, 11, 12)); //Mux cinturón 
//Contrucción de Multiplexores de salida
Mux Mux_LED(Pin(8, OUTPUT, PinType::Digital), Pinset(9, 10, 11, 12)); //Mux salida 

void setup() {
 Serial.begin(9600);    //Inicialización de puerto serial
 RS485Serial.begin(9600);
 pinMode(2, OUTPUT);
 digitalWrite(RS485EN, LOW); //Arduino en modo RX
}
void loop() {
  for (byte i = 0; i < 8; i++) {
    Data_Asiento[i] = Mux_Asiento.read(i); //Lectura y guardado de valor de entrada de Sensor de Asiento
    //Conversión de valor análogo de sensor de Asiento a Boleano
    //Se determina el rango del valor de la resistencia(MIN 200Ω y max 500Ω) para cuando el Asiento se encuentra ocupado.
    //Los rangos MIN y MAX dependerán del tipo de sensor que se esté utilizando.
    if(Data_Asiento[i] > 200 && Data_Asiento[i] < 500){
      Data_Asiento[i] = HIGH;
      }
    else{
    Data_Asiento[i] = LOW;
    }
    Data_Broche[i] = Mux_Broche.read(i); //Lectura y guardado de valor de entrada de sensor de Broche
    Data_Cinturon[i] = Mux_Cinturon.read(i); //Lectura y guardado de valor de entrada de sensor de Cinturón
    //Condicional para determinar si el asiento esta ocupado, si está con el conutón y, si la persona se coloca correctamente el cinturón
    if (Data_Asiento[i] && !Data_Broche[i] && !Data_Cinturon[i] || Data_Asiento[i] && !Data_Broche[i] && Data_Cinturon[i] || Data_Asiento[i] && Data_Broche[i] && Data_Cinturon[i]){
      Data_LED[i] = 1;
      Mux_LED.write(HIGH,i);  //envía al Demultiplexor Mux_LED un 1 logico encendiendo el LED correspondiente
    }
    else{
      Data_LED[i] = 0;
      Mux_LED.write(LOW,i);//envía al Demultiplexor Mux_LED un 0 logico apagando el LED correspondiente
    }
   // Serial.print("Asiento "); Serial.print(i+1); Serial.print(": "); Serial.println(Mux_Asiento.read(i)); //uso de monitor serial para fines de simulación
    delay(1000); //Ciclo de parpadeo de LED
  }
  if(RS485Serial.available() > 0) {//Lee el puerto Serial en busca de algún mensaje
    String mensaje = RS485Serial.readStringUntil('\n'); //Guarda el mensaje recibido en una variable tipo String
    mensaje.trim();   //Elimina los espacios en blanco al inicio y al final del String
    if(mensaje == ARD_NAME + "%%REQUEST") {//Analiza los ultimos caracteres de la variable mesage.ARD1 es la dirección/nombre para ESTE Arduino
      digitalWrite(RS485EN, HIGH);  //Coloca el arduino en modo Transmisión
      String text_data = ARD_NAME + "%%";
      //Se envía por el puerto serial los valores de los sensores de entrada del asiento(sensor Asiento, sensor Broche y sensor Cinturon)
      //de ese instante hacia la Raspberry mediante el Modulo MAX485.
      //Formato de envío:
      // [ARD1%%][# ASIENTO],[Data_Asiento],[Data_Broche],[Data_Cinturon],[Data_LED]
      // ejm:  ARD1%%1%%1,1,0,0
      //El formato será interpretado por la Raspberry y almacenado en un archivo de texto.
      for (byte i = 0; i < 8; i++){
        text_data += String(i + 1, DEC) + "," + 
                     String(Data_Asiento[i], BIN) + "," + 
                     String(Data_Broche[i], BIN) + "," + 
                     String(Data_Cinturon[i], BIN) + "," + 
                     String(Data_LED[i], BIN) + ";";
      }
      RS485Serial.println(text_data);
      digitalWrite(RS485EN, LOW);
    }
  }
}