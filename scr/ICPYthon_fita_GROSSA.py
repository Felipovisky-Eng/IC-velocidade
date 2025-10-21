
import pandas  as pd  #Análise dos dados
import tkinter as tk  #Interface Gráfica
import numpy   as np  #Calculos 
#import scipy   as sp  #Calculos complexos
import os             # Para manipulação de caminhos
import matplotlib.pyplot as plt   # Plota os gráficos
from tkinter import filedialog            # Busca arquivos
from scipy.signal import butter, filtfilt, firwin, freqz, find_peaks # Filtragem do sinal
from scipy.signal import hilbert, savgol_filter
from scipy.interpolate import interp1d


DistanciaFaixa = 10e-3  # Distancia entre as faixas pretas em Metros
raio = 320e-3             # Raio do disco em Metros

# Informações para subtitulo do gráfico

LarguraFaixa = DistanciaFaixa * 1000  # Largura das faixas em milímetros
Diametro = 2 * raio * 100 # Diâmetro do disco em cm
NumePassos = 9600 # Número de passos do motor NEMA 23 com o driver A4988 e 16 micro passos

# Informações para o subtítulo do gráfico
info_subtitulo1 = f"NEMA 23; {NumePassos} Passos/s; Distância entre faixas: {LarguraFaixa:.2f} mm; Diâmetro: {Diametro:.2f} cm; ADC 10 bits." # Subtítulo com informações do motor, passos por segundo, distância entre faixas, diâmetro do disco, ADC e tensão de alimentação
info_subtitulo2 = f"Filtro PB Butterworth 600 Hz de 2ª Ordem; {NumePassos} Passos/s; Distância entre faixas: {LarguraFaixa:.2f} mm." # Subtítulo com informações do motor, passos por segundo, distância entre faixas, diâmetro do disco, ADC e tensão de alimentação

#Dados esperados:

RMP_motor = ((NumePassos/16)/200)*60 # RMP do motor NEMA 23 (4800 passos/s dividido por 16 micro passos e 200 passos por revolução)
RPM_disco = RMP_motor*(20/75) # RMP do disco (RMP do motor vezes a relação de engrenagem 47/18)
Velocidade_disco = (RPM_disco * 2 * np.pi * raio) / 60 # Velocidade do disco em m/s (RMP do disco vezes 2*pi*raio dividido por 60)

print("RMP do motor NEMA 23: ",f'{RMP_motor:.2f}', "RPM") # Imprime o RMP do motor
print("RMP do disco: ", f'{RPM_disco:.2f}', "RPM") # Imprime o RMP do disco
print("Velocidade do disco: ", f'{Velocidade_disco:.2f}', "m/s") # Imprime a velocidade do disco

#
# Importaçao do arquivo .txt
#
#
#

def selecionar_arquivo():     #Função vai permitir abrir a janela para selecionar o arquivo                      
    
    root = tk.Tk()                                   # Base para parte gráfica
    root.withdraw()                                  # Oculta a janela principal do Tkinter
    arquivo = filedialog.askopenfilename(            # Abre a janela de seleção
        title="Selecione o arquivo .txt",            # Titulo do arquivo
        filetypes=[("Arquivos de texto", "*.txt")]   # Tipo do arquivo
    )
    return arquivo # Retorna o caminho do arquivo com o titulo na variavel "arquivo"

#
#
#
# Separação do arquivo .txt usando a pandas
#
#
#

def carregar_dados(caminho_arquivo): # Lê o arquivo 
    
    dados = pd.read_csv(caminho_arquivo, delimiter=",", header=None) # Separa o arquivo em duas colunas usando o ";"
    tempo = dados[0]      # Primeira coluna (tempo)
    valores = dados[1]    # Segunda coluna  (valores do sensor)
    return tempo, valores # Retorna primeiro a coluna de tempo depois a de valores

if __name__ == "__main__":
    print("Selecione o arquivo .txt no explorador de arquivos...")
    caminho = selecionar_arquivo() # Define o caminho para o arquivo com base na função anterior
    
    if caminho:
        print(f"Arquivo selecionado: {caminho}") # Imprime o caminho do arquivo
        tempo, valores = carregar_dados(caminho) # Carrega as variaveis 
        nome_arquivo = os.path.basename(caminho) # Carrega em uma string o nome do arquivo
        print("\nDados carregados com sucesso!")
        #print("Tempo:", tempo.values)           # Imprime os dados do tempo no terminal
        #print("Valores:", valores.values)       # Imprime os dados do valor no terminal
        #print(f"Nome do arquivo: {nome_arquivo}")
    else:
        print("Nenhum arquivo foi selecionado.")

#
#
#
# Manipulaçao de dados
#
#
#

Nome = nome_arquivo.replace(".txt", "") # Tira a extensão do arquivo (.txt) do nome dele
Nome = Nome.replace("_", " ") # Tira o "_" e subtitui por um espaço

tempo   = tempo.values    # Converte pandas.Series para numpy.ndarray
valores = valores.values  # Converte pandas.Series para numpy.ndarray

# Tratamento da variavel tempo
tempo = tempo - tempo[0]        # Faz com que a array comece em zero
tempo = tempo * 1e-6            # Converte microsegnudos para segundos
Diferenca = np.diff(tempo)      # Calcula a diferença do tempo
Mdiferenca = np.mean(Diferenca) # Calcula a média das diferenças
FS = 1/Mdiferenca               # Calcula a frequencia de amostragem




# ================================================================
# 🔧 Função para refinar o número de amostras (interpolação)
# ================================================================

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import numpy as np

def refinar_amostragem(tempo, valores, fator=2, FS=FS, ativar=False):
    """
    Aumenta o número de amostras por interpolação.

    Parâmetros:
    ------------
    tempo : ndarray
        Vetor original de tempo (em segundos)
    valores : ndarray
        Vetor original de valores correspondentes
    fator : int
        Fator de aumento da amostragem (ex: 2 = dobra, 4 = quadruplica)
    ativar : bool
        Se True, aplica interpolação e substitui variáveis

    Retorna:
    --------
    tempo, valores : ndarrays (interpolados ou originais)
    """

    if not ativar:
        print("Interpolação desativada — usando dados originais.")
        return tempo, valores

    # Nova frequência de amostragem
    FS_nova = FS * fator
    n_novo = int(len(tempo) * fator)

    # Cria vetor de tempo interpolado com o mesmo intervalo total
    tempo_interp = np.linspace(tempo[0], tempo[-1], n_novo)

    # Função de interpolação (linear)
    interp_func = interp1d(tempo, valores, kind='cubic', fill_value="extrapolate")
    valores_interp = interp_func(tempo_interp)

    # Gráfico de comparação
    plt.figure(figsize=(10, 4))
    plt.plot(tempo, valores, 'o-', label='Original', alpha=0.7)
    plt.plot(tempo_interp, valores_interp, '-', label=f'Interpolado ×{fator}', linewidth=2)
    plt.title(f'Comparação entre sinal original e interpolado (fator {fator}x)')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Atualiza frequência de amostragem global
    
    FS = FS_nova

    print(f"Interpolação ativada — {fator}x mais amostras.")
    print(f"Nova frequência de amostragem: {FS:.2f} Hz")

    return tempo_interp, valores_interp, FS


# ================================================================
# 🧩 Aplicação da função (cole logo após o cálculo de FS)
# ================================================================

# Exemplo: dobra a densidade de amostras
tempo, valores, FS = refinar_amostragem(tempo, valores, fator=20, FS=FS, ativar=True)



def sinc_interpol(tempo, valores, fator_interp=4):
    """
    Interpolação por sinc (banda limitada)
    - tempo: vetor original de tempo (s)
    - valores: vetor original de sinal
    - fator_interp: fator de aumento de amostras (ex: 4 → quadruplica Fs)
    """
    # Calcula nova frequência de amostragem
    Ts = np.mean(np.diff(tempo))
    Fs = 1 / Ts
    novo_Fs = Fs * fator_interp
    novo_Ts = 1 / novo_Fs

    # Novo eixo temporal
    tempo_interp = np.arange(tempo[0], tempo[-1], novo_Ts)

    # Interpolação por sinc
    t_diff = tempo_interp[:, None] - tempo[None, :]
    sinc_matrix = np.sinc(t_diff / Ts)
    valores_interp = np.dot(sinc_matrix, valores)


    plt.figure(figsize=(8,4))
    plt.plot(tempo, valores, 'o-', label='Original', alpha=0.7)
    plt.plot(tempo_interp, valores_interp, '-', label='Sinc interpolado', linewidth=1.5)
    plt.legend(); plt.grid()
    plt.title("Interpolação por sinc")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.show()


    return tempo_interp, valores_interp
# Exemplo de uso

#tempo, valores = sinc_interpol(tempo, valores, fator_interp=10)




Diferenca = np.diff(tempo)      # Calcula a diferença do tempo
Mdiferenca = np.mean(Diferenca) # Calcula a média das diferenças
FS = 1/Mdiferenca               # Calcula a frequencia de amostragem



print("\n")
print("Dados do tempo de coleta:") 
print("Intervalo médio entre coletas: ",f'{Mdiferenca:.3e}', "s")
if FS < 1000:
    print("Frequência de amostragem: ",f'{FS:.3f}', "Hz")
else:
    print("Frequência de amostragem: ",f'{FS/1000:.3f}', "kHz")
print('\n')



#
#
#
# Gráficos
#
#
#

# Configurações globais de fontes e DPI
plt.rcParams['font.family'] = 'Palatino Linotype'  # Ou 'Palatino'
plt.rcParams['axes.titleweight'] = "bold" # Títulos dos eixos e gráficos em negrito
plt.rcParams['figure.dpi'] = 170         # Define o DPI para todas as figuras
plt.rcParams['axes.labelweight'] = "bold" # Rótulos dos eixos
plt.rcParams['lines.linewidth'] = 1.5     # Espessura padrão das linhas
plt.rcParams['figure.autolayout'] = True  # Ajusta automaticamente o layout das figuras
plt.rcParams['axes.titlesize'] = 'large'  # Tamanho do título
plt.rcParams['axes.edgecolor'] = 'gray'   # Cor das bordas dos eixos
plt.rcParams['grid.color'] = 'lightgray'  # Cor da grade



# Plotar o gráfico dos dados dos Dados brutos
fig1, ax = plt.subplots()  # Cria figura e eixos
fig1.set_size_inches(7, 3.5)  # Largura x Altura em polegadas
ax.plot(tempo, valores, color=plt.cm.viridis(0.6), label="Dados do Sensor")  # Adiciona label para legenda
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)  # Personaliza a grade
ax.set_title("Leitura do ADC", fontsize=14, pad=26)     # Define o titúlo como o nome do arquivo e o tamnho da fonte
ax.set_xlabel("Tempo (s)", fontsize=12, labelpad=16)     # Define o tamanho da fonte do eixo X
ax.set_ylabel("Amplitude (V)", fontsize=12, labelpad=16) # Define o tamanho da fonte do eixo Y
ax.tick_params(axis='both', labelsize=9)                 # Define o tamanho da fonte dos rótulos dos eixos
ax.legend(fontsize=8)  # Adiciona a legenda
ax.text(0.5, 1.02, info_subtitulo1, transform=ax.transAxes, ha='center', va='bottom', fontsize=10) # Subtítulo abaixo do título principal
plt.tight_layout()                       # Ajusta o layout para evitar sobreposição de elementos
plt.show()                               # Mostrar o gráfico e mantem ele na tela

#
#
#
# Análise no dominio da frequência e outros
#
#
#


# --- FFT ---
FFT_BRUTA = np.fft.fft(valores)
FREQUENCIAS_BRUTAS = np.fft.fftfreq(len(tempo), Mdiferenca)

FFT_BRUTA[0] = 0  # Remove componente DC
FFT_BRUTA = FFT_BRUTA / np.max(np.abs(FFT_BRUTA))  # Normaliza

# --- Apenas parte positiva ---
N = len(FFT_BRUTA)
f_pos = FREQUENCIAS_BRUTAS[:N // 2]
fft_mag = np.abs(FFT_BRUTA[:N // 2])
fft_db = 20 * np.log10(fft_mag + 1e-12)  # Em dB, evita log(0)

# --- Frequência de pico ---
idx_max = np.argmax(fft_mag[1:]) + 1  # ignora 0 Hz
freq_max = f_pos[idx_max]
db_max = fft_db[idx_max]
mag_max = fft_mag[idx_max]

# --- Configurações visuais globais ---

plt.rcParams['axes.edgecolor'] = 'gray'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = 'lightgray'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.linewidth'] = 0.5


# --- Gráfico em dB ---
fig1, ax1 = plt.subplots(figsize=(7, 3.5))
ax1.plot(f_pos, fft_db, color=plt.cm.viridis(0.4), label=f"DFT Normalizada em dB")
ax1.axvline(freq_max, color='k', linestyle='--', linewidth=1, alpha=0.8, label=(f"Frequência mais intensa em: {freq_max:.2f} Hz"))

ax1.set_title("DFT Normalizada (Escala dB)", fontsize=14, pad=20)
ax1.set_xlabel("Frequência (Hz)", fontsize=11, labelpad=16)
ax1.set_ylabel("Magnitude (dB)", fontsize=11, labelpad=16)
ax1.set_xscale("log")
ax1.set_xlim(left=max(f_pos[1], 1e-1))
ax1.tick_params(axis='both', labelsize=9)
ax1.legend(fontsize=9)
ax1.text(0.5, 1.02, info_subtitulo1, transform=ax1.transAxes, ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# --- Gráfico linear ---
fig2, ax2 = plt.subplots(figsize=(7, 3.5))
ax2.plot(f_pos, fft_mag, color=plt.cm.viridis(0.4), label=f"DFT Normalizada")
ax2.axvline(freq_max, color='red', linestyle='--', linewidth=1, alpha=0.8,label=(f"Frequência mais intensa em: {freq_max:.2f} Hz"))
ax2.set_title("DFT Normalizada", fontsize=14, pad=20)
ax2.set_xlabel("Frequência (Hz)", fontsize=11, labelpad=16)
ax2.set_ylabel("Magnitude", fontsize=11, labelpad=16)
ax2.tick_params(axis='both', labelsize=9)
ax2.legend(fontsize=9)
ax2.text(0.5, 1.02, info_subtitulo1, transform=ax2.transAxes, ha='center', fontsize=10)

plt.tight_layout()
plt.show()

 









def equalize_by_envelope_hilbert(tempo, sinal, alvo=None, fs=None, smooth_ms=20, max_gain=5.0):
    """
    Equaliza amplitude usando envelope por transformada de Hilbert e suavização (Savitzky-Golay).
    - fs: frequência de amostragem (Hz). Se None, calcula a partir de tempo.
    - smooth_ms: janela de suavização em ms (convertida para nº pontos)
    - max_gain: limite superior para ganho
    Retorna: sinal_equalizado, envelope_smooth, ganho
    """
    if fs is None:
        Ts = np.mean(np.diff(tempo))
        fs = 1.0 / Ts

    # 1) envelope via Hilbert
    analytic = hilbert(sinal)
    envelope = np.abs(analytic)

    # 2) suaviza envelope (janela em pontos)
    win_pts = int(max(3, round((smooth_ms/1000.0) * fs)))
    if win_pts % 2 == 0:
        win_pts += 1
    if win_pts >= len(envelope):
        win_pts = len(envelope)-1
        if win_pts % 2 == 0:
            win_pts -= 1

    if win_pts >= 5:
        envelope_smooth = savgol_filter(envelope, win_pts, 3)
    else:
        envelope_smooth = envelope

    # 3) define alvo (mediana das amostras do envelope ou valor dado)
    if alvo is None:
        alvo = np.median(envelope_smooth)

    # 4) evita zeros
    eps = 1e-9 * np.max(envelope_smooth)
    envelope_smooth = np.where(envelope_smooth < eps, eps, envelope_smooth)

    # 5) ganho e aplicação
    ganho = (alvo / envelope_smooth)
    ganho = np.clip(ganho, 0.0, max_gain)
    sinal_eq = sinal * ganho

    return sinal_eq, envelope_smooth, ganho




# opção B: por envelope Hilbert
sinal_eq_hilbert, env_hilb, ganho_hilb = equalize_by_envelope_hilbert(tempo, valores, fs=FS, smooth_ms=30, max_gain=2.0)
# Plot comparativo  
fig3, ax3 = plt.subplots(figsize=(7, 3.5))
ax3.plot(tempo, valores, 'o-', label='Original', color=plt.cm.viridis(0.6), alpha=0.6)
ax3.plot(tempo, sinal_eq_hilbert, '-', label='Equalizado (Hilbert)', color=plt.cm.viridis(0.1), linewidth=1.5)
ax3.set_title("Equalização por Envelope Hilbert", fontsize=14, pad=20)
ax3.set_xlabel("Tempo (s)", fontsize=11, labelpad=16)
ax3.set_ylabel("Amplitude", fontsize=11, labelpad=16)
ax3.tick_params(axis='both', labelsize=9)
ax3.legend(fontsize=9)
ax3.text(0.5, 1.02, info_subtitulo1, transform=ax3.transAxes, ha='center', fontsize=10)
plt.tight_layout()
plt.show()
# valores = sinal_eq_hilbert  # Atualiza valores para o sinal equalizado













#
#
#
# Filtragem do sinal com IRR
#
#
#

# Filtro IIR butter  Passa-Baixo

b_baixo, a_baixo = butter(2, 1600/ (0.5 * FS), btype='low') # calcula os coeficientes do Filtro IIR passa-baixo
#print("\n")
#print("Coeficientes do filtro IIR Passa-Baixo:") # Imprime os coeficientes do filtro
#print(b_baixo)
#print(a_baixo)

#b_baixo  = [0.02785972, 0.05571943, 0.02785972]
#a_baixo  = [ 1.0,         -1.47548095,  0.58691982]
Y_baixo = filtfilt(b_baixo, a_baixo, valores)  

# Plotar o gráfico do sinal com cada filtro para comparar
fig5, ex = plt.subplots(figsize=(7, 3.5))  # Cria figura e eixos
ex.plot(tempo, valores, color=plt.cm.viridis(0.7), label="Dados Brutos")      # Plota Dados Brutos
ex.plot(tempo,Y_baixo, color=plt.cm.viridis(0.2), label="Sinal Filtrado")     # Plota Passa-Baixo
ex.grid(color='gray', linestyle='--', linewidth=0.5)                          # Personaliza a grade
ex.set_title('Sinal filtrado', fontsize=14, pad=20)   # Define o titúlo como o nome do arquivo e o tamnho da fonte
ex.set_xlabel("Tempo (s)", fontsize=10, labelpad=16)  # Define o tamanho da fonte do eixo X
ex.set_ylabel("Amplitude", fontsize=10, labelpad=16)  # Define o tamanho da fonte do eixo Y
ex.legend()                              # Adiciona a legenda
ex.text(0.5, 1.02, info_subtitulo2, transform=ex.transAxes, ha='center', fontsize=10) # Subtítulo abaixo do título principal
plt.tight_layout()                       # Ajusta o layout para evitar sobreposição de elementos
plt.show()                               # Mostrar o gráfico e mantem ele na tela

#
#
#
# Identificação de picos para analise de velocidade
#
#
#

Altura_Minima    = None   # Altura minima para identificar um pico
Distancia_Minima = None  # Distanica minima entre um pico e outro
Proeminencia     = 25  # O quanto um pico deve se destacar entre o seu arredor para ser identificado
Largura          = None   # Largura minima para ser considerado um pico
Limite           = None   # Limite necessarios dos picos ? (Eu não entendi, mas me resolvo no dicionário depois)

# Identificação dos picos e vales
Picos_BAIXO, Dicionario_Pico_BAIXO = find_peaks(Y_baixo ,distance=Distancia_Minima,height=Altura_Minima,threshold=Limite,prominence=Proeminencia,width=Largura) # Identifica os picos
Vales_BAIXO, Dicionario_Vale_BAIXO = find_peaks(-Y_baixo,distance=Distancia_Minima,height=Altura_Minima,threshold=Limite,prominence=Proeminencia,width=Largura) # Identifica os vales

# Identifiação do tempo em que ocorre cada pico e cada vale
tempos_picos_BAIXO = np.array(tempo[Picos_BAIXO]) # Identifica o tempo dos picos usando o tempo de coleta e converte em um np.array
tempos_vales_BAIXO = np.array(tempo[Vales_BAIXO]) # Identifica o tempo dos vales usando o tempo de coleta e converte em um np.array
tempos_intercalado = np.sort(np.concatenate((tempos_picos_BAIXO, tempos_vales_BAIXO))) # Junuta os dois tempos e ordena eles

# Calcula a diferença de tempo para o calcula da velocidade
Diferenca_Tempo_Pico        = (np.diff(tempos_picos_BAIXO)) # Diferença de tempo entre picos
Diferenca_Tempo_Vale        = (np.diff(tempos_vales_BAIXO)) # Diferença de tempo entre vales
Diferença_Tempo_Intercalado = (np.diff(tempos_intercalado) ) # Diferença de tempo ente um pico e um vale

# Calcula a velocdade com base na distancia percorrida durante a variaçãod e tempo
Velocidade_Pico_BAIXO  = 2*DistanciaFaixa / Diferenca_Tempo_Pico        # Divide a distancia entre um pico e outro pela diferença de tempo
Velocidade_Vale_BAIXO  = 2*DistanciaFaixa / Diferenca_Tempo_Vale        # Divide a distancia entre um vale e outro pela diferença de tempo
Velocidade_Intercalado =   DistanciaFaixa / Diferença_Tempo_Intercalado # Divide a distancia entre um pico e um vale pela diferenaça de tempo
 
# Calcula a media dos valores obtidos e a velocidade considerando somente o primeiro e ultimo valor
Vmedia1_PICO_BAIXA = np.mean(Velocidade_Pico_BAIXO) # Média das velocidades

# Calcula a media dos valores obtidos e a velocidade considerando somente o primeiro e ultimo valor
Vmedia1_VALE_BAIXA = np.mean(Velocidade_Vale_BAIXO) # Média das velocidades

# Calcula a media dos valores obtidos e a velocidade considerando somente o primeiro e ultimo valor
Vmedia1_Intercalado = np.mean(Velocidade_Intercalado) # Média das velocidades instercaladas

# Calculo da rotação em RPM para utilizar em discos
RPM_Pico_BAIXO =  (Vmedia1_PICO_BAIXA * 60)  / (2*np.pi * raio)  # Converte a velocidade média dos picos em RPM
RPM_Vale_BAIXO =  (Vmedia1_VALE_BAIXA * 60)  / (2*np.pi * raio)  # Converte a velocidade média dos vales em RPM
RPM_Intercalado = (Vmedia1_Intercalado * 60) / (2*np.pi * raio)  # Converte a velocidade média intercalada em RPM


# Testes dos valores de velocidade em m/s e RPM
print()
print('Velocidade linear média:')
print('Velocidade média obtida com diferença de tempo dos picos: ', f'{Vmedia1_PICO_BAIXA:.3f}', "m/s")
print('Velocidade média obtida com diferença de tempo dos vales: ', f'{Vmedia1_VALE_BAIXA:.3f}', "m/s")
print('Velocidade média obtida com diferença de tempo intercalada: ', f'{Vmedia1_Intercalado:.3f}', "m/s")

print()
print('Velocidade média em RPM:')
print('Velocidade média obtida com diferença de tempo dos picos: ', f'{RPM_Pico_BAIXO:.3f}', "RPM")
print('Velocidade média obtida com diferença de tempo dos vales: ', f'{RPM_Vale_BAIXO:.3f}', "RPM")
print('Velocidade média obtida com diferença de tempo intercalada: ', f'{RPM_Intercalado:.3f}', "RPM")


# Média da média das velocidades
Vmedia1 = (Vmedia1_PICO_BAIXA + Vmedia1_VALE_BAIXA + Vmedia1_Intercalado) / 3  # Média das velocidades


# Plota o gráfico dos picos e vales no sinal filtrado pelo IIR Passa-Baixo
fig9, mx = plt.subplots(figsize=(8, 4.5))  # Tamanho customizado
mx.plot(tempo, Y_baixo, color=plt.cm.viridis(0.5), linewidth=1.8, label="Sinal Filtrado", zorder=1) # Sinal filtrado (cor suave)
mx.scatter(tempos_picos_BAIXO, Y_baixo[Picos_BAIXO], color=plt.cm.viridis(0.8), edgecolors='k', s=40, marker='o', label="Picos", zorder=3) # Picos (laranja vibrante com borda escura)
mx.scatter(tempos_vales_BAIXO, Y_baixo[Vales_BAIXO], color=plt.cm.viridis(0.2), edgecolors='k', s=40, marker='o', label="Vales", zorder=3) # Vales (verde vibrante com borda escura)
mx.grid(color='lightgray', linestyle='--', linewidth=0.6, alpha=0.85)
mx.set_title('Identificação dos picos e vales', fontsize=13, pad=20, fontweight='bold')
mx.set_xlabel("Tempo (s)", fontsize=11, labelpad=10)
mx.set_ylabel("Amplitude", fontsize=11, labelpad=10)
mx.tick_params(axis='both', labelsize=9)
mx.legend(fontsize=9, loc='upper right', frameon=True)
mx.text(0.5, 1.02, info_subtitulo1, transform=mx.transAxes,ha='center', fontsize=10)
plt.tight_layout()
plt.show()




# Cria figura e eixos com tamanho customizado
fig10, nx = plt.subplots(figsize=(8, 4.5))
nx.plot(tempos_picos_BAIXO[:-1], Velocidade_Pico_BAIXO, 'o-', color=plt.cm.viridis(0.1), linewidth=1.5, markersize=4, label='Velocidade (Picos)', zorder=2)
nx.plot(tempos_vales_BAIXO[:-1], Velocidade_Vale_BAIXO, 's-', color=plt.cm.viridis(0.5), linewidth=1.5, markersize=4, label='Velocidade (Vales)', zorder=2)
nx.plot(tempos_intercalado[:-1], Velocidade_Intercalado, '^-', color=plt.cm.viridis(0.2), linewidth=1.5, markersize=4, label='Velocidade (Intercalada)', zorder=2)
nx.axhline(Vmedia1_Intercalado, color='black', linestyle='--', linewidth=1.2, label=f'Velocidade média: {Vmedia1_Intercalado:.3f} m/s', zorder=1)
nx.grid(color='lightgray', linestyle='--', linewidth=0.7, alpha=0.8)# Título e rótulos com estilo profissional
nx.set_title("Velocidades por diferentes métodos", fontsize=14, fontweight='bold', pad=20)
nx.set_xlabel("Tempo (s)", fontsize=11, labelpad=10)
nx.set_ylabel("Velocidade (m/s)", fontsize=11, labelpad=10)
nx.tick_params(axis='both', labelsize=9)
nx.legend(fontsize=9, loc='best', frameon=True)
nx.text(0.5, 1.02, info_subtitulo2, transform=nx.transAxes, ha='center', fontsize=10)
plt.tight_layout()
plt.show()


# ============================================================
# MÓDULO: Cálculo e Plotagem da Aceleração
# ============================================================

# Cálculo das acelerações (derivada da velocidade)
Aceleracao_Pico_BAIXO = np.diff(Velocidade_Pico_BAIXO) / np.diff(tempos_picos_BAIXO[:-1])
Aceleracao_Vale_BAIXO = np.diff(Velocidade_Vale_BAIXO) / np.diff(tempos_vales_BAIXO[:-1])
Aceleracao_Intercalado = np.diff(Velocidade_Intercalado) / np.diff(tempos_intercalado[:-1])

# Tempos correspondentes (ajustados ao tamanho da derivada)
tempos_acel_pico = tempos_picos_BAIXO[1:-1]
tempos_acel_vale = tempos_vales_BAIXO[1:-1]
tempos_acel_intercalado = tempos_intercalado[1:-1]

# Cálculo das médias (opcional, apenas para referência)
Amedia_Pico_BAIXO = np.mean(Aceleracao_Pico_BAIXO)
Amedia_Vale_BAIXO = np.mean(Aceleracao_Vale_BAIXO)
Amedia_Intercalado = np.mean(Aceleracao_Intercalado)

# Impressão dos resultados médios
print()
print("Aceleração média:")
print(f"Aceleração média (Picos):        {Amedia_Pico_BAIXO:.3f} m/s²")
print(f"Aceleração média (Vales):        {Amedia_Vale_BAIXO:.3f} m/s²")
print(f"Aceleração média (Intercalada):  {Amedia_Intercalado:.3f} m/s²")

# ============================================================
# Plotagem da Aceleração
# ============================================================

fig11, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(tempos_acel_pico, Aceleracao_Pico_BAIXO, 'o-', color=plt.cm.viridis(0.1), linewidth=1.5, markersize=4, label='Aceleração (Picos)', zorder=2)
ax.plot(tempos_acel_vale, Aceleracao_Vale_BAIXO, 's-', color=plt.cm.viridis(0.5), linewidth=1.5, markersize=4, label='Aceleração (Vales)', zorder=2)
ax.plot(tempos_acel_intercalado, Aceleracao_Intercalado, '^-', color=plt.cm.viridis(0.2), linewidth=1.5, markersize=4, label='Aceleração (Intercalada)', zorder=2)

# Linha horizontal com aceleração média (intercalada)
ax.axhline(Amedia_Intercalado, color='black', linestyle='--', linewidth=1.2, 
           label=f"Aceleração média (Intercalada): {Amedia_Intercalado:.3f} m/s²", zorder=1)

# Estilo do gráfico
ax.grid(color='lightgray', linestyle='--', linewidth=0.7, alpha=0.8)
ax.set_title("Aceleração derivada das velocidades", fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel("Tempo (s)", fontsize=11, labelpad=10)
ax.set_ylabel("Aceleração (m/s²)", fontsize=11, labelpad=10)
ax.tick_params(axis='both', labelsize=9)
ax.legend(fontsize=9, loc='best', frameon=True)

# Subtítulo opcional (mesmo estilo dos gráficos anteriores)
ax.text(0.5, 1.02, info_subtitulo2, transform=ax.transAxes, ha='center', fontsize=10)

plt.tight_layout()
plt.show()

















# Variável para armazenar o sinal a ser trabalhado
# A variável Variavel é a que será usada para identificar os padrões de repetição
Variavel = Y_baixo # Sinal a ser trabalhado (o ideal até então é esse)


#
# Autocorrelação do sinal filtrado
#

# Autocorrelação normalizada no modo 'same'
padrao1 = np.correlate(Variavel, Variavel, mode='same')
padrao1 = padrao1 / np.max(padrao1)

# Detecta picos significativos da autocorrelação
pico_padrao, _ = find_peaks(padrao1, height=0.1, distance=int(0.002*FS))  # distância mínima de 2ms

# Cria o eixo de tempo dos lags
lags = np.arange(-len(Variavel) + 1, len(Variavel))
lags = lags[len(Variavel)-1:]            # parte positiva
tempos_lags = lags / FS                  # em segundos
tempos_picos = tempos_lags[pico_padrao]  # picos em tempo

# Diferença de tempo entre picos (repetições do padrão)
repeticoes = np.diff(tempos_picos)
frequencia_sinal = 1 / np.mean(repeticoes)

#
# Cálculo da velocidade tangencial
#
velocidade = frequencia_sinal * (2 * DistanciaFaixa)
RPM = (velocidade * 60) / (2 * np.pi * raio)

#
# Impressão dos resultados
#
print("\nDados da autocorrelação:")
print(f"Número de repetições detectadas: {len(repeticoes)}")
print(f"Período médio: {np.mean(repeticoes):.5f} s")
print(f"Frequência: {frequencia_sinal:.2f} Hz")
print(f"Velocidade linear média: {velocidade:.3f} m/s")
print(f"Rotação média: {RPM:.2f} RPM\n")


fig, axs = plt.subplots(2, 1, figsize=(9, 4))

# Sinal original
axs[0].plot(np.arange(len(Variavel)) / FS, Variavel, color=plt.cm.viridis(0.4), label='Sinal Filtrado')
axs[0].set_title('Sinal Filtrado no Tempo', fontsize=14, pad=20)
axs[0].set_xlabel('Tempo (s)')
axs[0].set_ylabel('Amplitude')
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].legend()
axs[0].text(0.5, 1.02, info_subtitulo2, transform=axs[0].transAxes, ha='center', fontsize=10)  # Subtítulo abaixo do título principal

# Autocorrelação
axs[1].plot(tempos_lags, padrao1, color='black', label='Autocorrelação')
axs[1].plot(tempos_picos, padrao1[pico_padrao], 'rx', label='Picos Detectados')
axs[1].set_title('Autocorrelação Normalizada', fontsize=14)
axs[1].set_xlabel('Defasagem (s)')
axs[1].set_ylabel('Amplitude')
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend()

plt.tight_layout()
plt.show()








def equalize_by_peaks(tempo, sinal, picos_idx, alvo=None, smoothing_window=51, polyorder=3, max_gain=5.0):
    """
    Equaliza amplitude usando interpolação entre amplitudes dos picos.
    - tempo: vetor tempo (1D)
    - sinal: vetor do sinal (1D)
    - picos_idx: índices dos picos (ex: saída de find_peaks)
    - alvo: amplitude alvo (se None -> mediana das amplitudes dos picos)
    - smoothing_window: janela Savitzky-Golay para suavizar o envelope interpolado (odd)
    - polyorder: ordem do SavGol
    - max_gain: limite superior para o fator de ganho
    Retorna: sinal_equalizado, envelope_interp, ganho_aplicado, tempo_envelope (mesma dimensão do tempo)
    """

    # 1) amplitudes nos picos
    amp_picos = sinal[picos_idx]
    tempo_picos = tempo[picos_idx]

    # 2) alvo
    if alvo is None:
        alvo = np.median(amp_picos)

    # 3) interpola uma curva contínua do envelope dos picos
    f_env = interp1d(tempo_picos, amp_picos, kind='cubic', bounds_error=False, fill_value="extrapolate")
    envelope_interp = f_env(tempo)

    # 4) suaviza o envelope (para evitar ganho abrupto)
    # garante janela impar e menor que tamanho do sinal
    if smoothing_window >= len(envelope_interp):
        smoothing_window = len(envelope_interp) - 1
        if smoothing_window % 2 == 0:
            smoothing_window -= 1
    if smoothing_window >= 5:
        envelope_smooth = savgol_filter(envelope_interp, smoothing_window, polyorder)
    else:
        envelope_smooth = envelope_interp

    # 5) evita valores muito pequenos
    eps = 1e-6 * np.max(np.abs(envelope_smooth))
    envelope_smooth = np.where(np.abs(envelope_smooth) < eps, eps, envelope_smooth)

    # 6) calcula ganho (capado por max_gain)
    ganho = (alvo / envelope_smooth)
    ganho = np.clip(ganho, 0.0, max_gain)

    # 7) aplica ganho ao sinal
    sinal_eq = sinal * ganho

    return sinal_eq, envelope_smooth, ganho



# ===========================
# Exemplo de uso após detectar picos (com find_peaks)
# ===========================
# supondo que 'tempo', 'valores' e 'Picos_BAIXO' existam (como no seu fluxo)
# opção A: por picos
sinal_eq_peaks, env_peaks, ganho_peaks = equalize_by_peaks(tempo, valores, Picos_BAIXO, smoothing_window=101, max_gain=4.0)



# ===========================
# Plots comparativos
# ===========================
plt.figure(figsize=(10,5))
plt.plot(tempo, valores, label='Original', alpha=0.6)
plt.plot(tempo, sinal_eq_peaks, label='Equalizado (picos)', alpha=0.9)
plt.plot(tempo, sinal_eq_hilbert, label='Equalizado (hilbert)', alpha=0.9)
plt.scatter(tempo[Picos_BAIXO], valores[Picos_BAIXO], facecolors='none', edgecolors='k', label='Picos originais')
plt.legend()
plt.title('Comparação: original vs equalizações')
plt.xlabel('Tempo (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plota envelopes e ganho (usar para diagnosticar)
plt.figure(figsize=(10,4))
plt.plot(tempo, env_peaks, label='Envelope interp (picos)', alpha=0.9)
plt.plot(tempo, env_hilb, label='Envelope Hilbert', alpha=0.6)
plt.legend(); plt.grid(); plt.title('Envelopes estimados')
plt.tight_layout()
plt.show()
