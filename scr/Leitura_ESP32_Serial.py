import serial
import struct
import time
import csv

porta = 'COM3'  # ajuste conforme necessário
baud = 1500000
total_amostras = 52100

ser = serial.Serial(porta, baud, timeout=1)
print("Aguardando READY...")

# Aguarda ESP32
while ser.readline().strip() != b'READY':
    pass

print("Iniciando leitura...")

dados = []
inicio = time.time()

for _ in range(total_amostras):
    raw = ser.read(6)
    if len(raw) < 6:
        continue
    t, leitura = struct.unpack('<IH', raw)  # < = little endian, I = uint32, H = uint16
    dados.append((t, leitura))

fim = time.time()
print(f"Leituras recebidas: {len(dados)}  Taxa: {len(dados)/(fim - inicio):.1f} amostras/s")

# Salva CSV simples
with open('dados_adc.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(dados)

print("Salvo como dados_adc.csv")
