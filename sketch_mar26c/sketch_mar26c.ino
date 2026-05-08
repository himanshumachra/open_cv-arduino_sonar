#include <Servo.h>

// Updated for A0 and A1
const int trigPin = A0; 
const int echoPin = A1;
const int servoPin = 11;
Servo myServo;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
  myServo.attach(servoPin);
}

void loop() {
  // Sweep from 0 to 180 degrees
  for (int i = 0; i <= 180; i++) {
    moveAndMeasure(i);
  }
  // Sweep back from 180 to 0 degrees
  for (int i = 180; i >= 0; i--) {
    moveAndMeasure(i);
  }
}

void moveAndMeasure(int angle) {
  myServo.write(angle);
  delay(30); // Allow servo to reach position
  
  long duration;
  int distance;
  
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2;

  // Crucial: Format must be "angle,distance"
  Serial.print(angle);
  Serial.print(",");
  Serial.println(distance);
}