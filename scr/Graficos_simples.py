import pandas  as pd            # Manipulação do arquivo
import os                       # Para manipulação de caminhos
import tkinter as tk            # Interface Gráfica
from tkinter import filedialog  # Caixa de diálogo
import numpy as np
import matplotlib.pyplot as plt

def selecionar_arquivo():     #Função vai permitir abrir a janela para selecionar o caminho do arquivo                      
    
    root = tk.Tk()                                   # Base para parte gráfica
    root.withdraw()                                  # Oculta a janela principal do Tkinter
    arquivo = filedialog.askopenfilename(            # Abre a janela de seleção
        title="Selecione o arquivo .csv",            # Titulo do arquivo
        filetypes=[("Arquivo de Valores Separados por Vírgulas do Microsoft Excel", "*.csv")]   # Tipo do arquivo
    )
    return arquivo # Retorna o caminho do arquivo com o titulo na variavel "arquivo"


def carregar_dados(caminho_arquivo): # Lê o arquivo 
    
    dados = pd.read_csv(caminho_arquivo, delimiter=",", header=None) # Busca e Separa o arquivo em duas colunas usando o ","
    tempo = dados[0]      # Primeira coluna (tempo)
    valores = dados[1]    # Segunda coluna  (valores do sensor)
    return tempo, valores # Retorna primeiro a coluna de tempo depois a de valores 

if __name__ == "__main__":
    print("Selecione o arquivo .txt do arquivo no explorador de arquivos...")
    caminho = selecionar_arquivo() # Define o caminho para o arquivo com base na função anterior
    print(caminho)
    
    if caminho:
        print(f"Arquivo selecionado: {caminho}") # Imprime o caminho do arquivo
        tempo, valores = carregar_dados(caminho) # Carrega as variaveis 
        nome_arquivo = os.path.basename(caminho) # Carrega em uma string o nome do arquivo
        print("\nDados carregados com sucesso!")
        #print("Tempo:", tempo.values)           # Imprime os dados do tempo no terminal
        #print("Valores:", valores.values)       # Imprime os dados do valor no terminal
        print(f"Nome do arquivo: {nome_arquivo}")
    else:
        print("Nenhum arquivo foi selecionado.")


Nome = nome_arquivo.replace(".csv", "") # Tira a extensão do arquivo (.csv) do nome dele
Nome = Nome.replace("_", " ") # Tira o "_" e substitui por um espaço

tempo   = tempo.values    # Converte pandas.Series para numpy.ndarray
valores = valores.values  # Converte pandas.Series para numpy.ndarray

#tempo = tempo - tempo[0]        # Faz com que a array comece em zero

timestamps = tempo
leituras = valores




# Configurações globais de fontes e DPI
plt.rc('font', family='serif', size=12)   # Altera o tipo e o tamanho da fonte
plt.rcParams['axes.titleweight'] = "bold" # Títulos dos eixos e gráficos em negrito
plt.rcParams['figure.dpi'] = 140          # Define o DPI para todas as figuras
plt.rcParams['axes.labelweight'] = "bold" # Rótulos dos eixos
#plt.rcParams['lines.linestyle'] = '--'    # Linhas tracejadas por padrão
plt.rcParams['lines.linewidth'] = 1.0     # Espessura padrão das linhas

# Calcula jitter e frequência
delta_t = np.diff(timestamps)  # us
freq_amostragem_media = 1e6 / np.mean(delta_t)

# Mostra frequência
print(f"\nFrequência de amostragem média: {freq_amostragem_media:.2f} Hz")

# --- Gráfico do jitter ---
plt.figure(figsize=(10,5))
plt.plot(delta_t, label="Jitter (delta_t)")
plt.hlines(np.mean(delta_t), 0, len(delta_t), color='red', linestyle='--', label='Média')
plt.title("Jitter entre amostras (us)")
plt.xlabel("Índice da amostra")
plt.ylabel("Intervalo entre amostras (us)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfico do tempo absoluto (para verificar rampa) ---
plt.figure(figsize=(10,5))
plt.plot(timestamps, label='Tempo das amostras (us)')
plt.title("Tempo absoluto das amostras")
plt.xlabel("Índice da amostra")
plt.ylabel("Tempo (us)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Gráfico da leitura do ADC ---
plt.figure(figsize=(10,5))
plt.plot(timestamps, leituras, label='Leitura do ADC')
plt.title("Leitura do ADC ao longo do tempo")
plt.xlabel("Tempo (us)")
plt.ylabel("Leitura do ADC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()