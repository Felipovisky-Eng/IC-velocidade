const int TOTAL_AMOSTRAS = 52100;
unsigned long t;
uint16_t leitura;

void setup() {
  Serial.begin(1500000);
  analogReadResolution(10);

  // Aquece o ADC
  for (int i = 0; i < 10000; i++) {
    analogRead(34);
    delayMicroseconds(100);
  }

  delay(500);
  Serial.println("READY");
}

void loop() {
  static int count = 0;

  while (count < TOTAL_AMOSTRAS) {
    t = micros();
    leitura = analogRead(34);
    Serial.write((uint8_t*)&t, sizeof(t));           // 4 bytes
    Serial.write((uint8_t*)&leitura, sizeof(leitura)); // 2 bytes
    delayMicroseconds(19);  // 10 kHz
    count++;
  }

  Serial.println("FIM");
  while (1);
}
