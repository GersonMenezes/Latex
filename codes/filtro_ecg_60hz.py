import wfdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import signal
import os

# --- 1. CONFIGURAÇÕES GERAIS ---
RECORD_NAME = '100'     # Arquivo do MIT-BIH
JANELA_SEGUNDOS = 3     # Tamanho da janela de visualização
FREQ_RUIDO_REDE = 60.0  # Frequência da interferência elétrica
Q_FACTOR = 35           # Fator de qualidade do filtro (maior = corte mais estreito)
AMPLITUDE_RUIDO = 0.15  # Intensidade do ruído (ajuste se necessário)

# --- 2. CARREGAMENTO DOS DADOS REAIS (Base "Limpa") ---
print(f"Carregando registro {RECORD_NAME}...")
if not os.path.exists(f"{RECORD_NAME}.dat"):
    print("ERRO: Arquivos .dat/.hea não encontrados.")
    exit()

try:
    record = wfdb.rdrecord(RECORD_NAME, channels=[0])
    sinal_real_limpo = record.p_signal.flatten()
    fs = record.fs # Frequência de amostragem original (geralmente 360Hz)
    print(f"Sinal carregado. Fs: {fs} Hz. Amostras: {len(sinal_real_limpo)}")
except Exception as e:
    print(f"Erro ao ler arquivo: {e}")
    exit()

# --- 3. GERAÇÃO DO RUÍDO E CONTAMINAÇÃO ---
# Cria um vetor de tempo para todo o sinal
t_total = np.arange(len(sinal_real_limpo)) / fs
# Gera o ruído de rede (senóide pura em 60Hz)
ruido_rede = AMPLITUDE_RUIDO * np.sin(2 * np.pi * FREQ_RUIDO_REDE * t_total)
# Sinal que entra no sistema (Sinal Real + Ruído)
sinal_entrada_sujo = sinal_real_limpo + ruido_rede

# --- 4. PROJETO DO FILTRO DIGITAL (IIR NOTCH) ---
# Calcula os coeficientes do filtro para remover 60Hz na frequência de amostragem 'fs'
b_notch, a_notch = signal.iirnotch(FREQ_RUIDO_REDE, Q_FACTOR, fs)
# Estado inicial da memória do filtro para processamento em tempo real
zi = signal.lfilter_zi(b_notch, a_notch)

# --- 5. PREPARAÇÃO DA INTERFACE GRÁFICA (TCC STYLE) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
plt.subplots_adjust(hspace=0.4) # Espaço entre os gráficos

# Buffers para rolagem
tamanho_buffer = int(JANELA_SEGUNDOS * fs)
buffer_sujo = np.zeros(tamanho_buffer)
buffer_limpo = np.zeros(tamanho_buffer)
eixo_x = np.linspace(0, JANELA_SEGUNDOS, tamanho_buffer)

# Linhas dos gráficos
# Usando cor vermelha para "ruim/sujo" e azul padrão para "bom/limpo"
linha_suja, = ax1.plot(eixo_x, buffer_sujo, color='#d62728', lw=1.2)
linha_limpa, = ax2.plot(eixo_x, buffer_limpo, color='#1f77b4', lw=1.5)

# Estilização Profissional
def estilizar_eixo(ax, titulo, y_limites):
    ax.set_title(titulo, fontsize=11, fontweight='bold')
    ax.set_ylabel("Amplitude (mV)", fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlim(0, JANELA_SEGUNDOS)
    # Calcula limites baseados no sinal real para ficar sempre visível
    ax.set_ylim(y_limites[0] - 0.5, y_limites[1] + 0.5)

#limites_y = (np.min(sinal_entrada_sujo), np.max(sinal_entrada_sujo))
limites_y = (-0.5, 1.0)
estilizar_eixo(ax1, "Entrada: Sinal ECG Contaminado por Ruído de Rede (60Hz)", limites_y)
estilizar_eixo(ax2, "Saída: Filtragem Digital Notch (IIR) em Tempo Real", limites_y)
ax2.set_xlabel("Tempo na Janela (s)", fontsize=10)

# Variável de controle do loop
indice_global = 0
VELOCIDADE_SIMULACAO = 8 # Pontos por frame (aumente para acelerar)

# --- 6. LOOP DE PROCESSAMENTO EM TEMPO REAL ---
def atualizar(frame):
    global indice_global, buffer_sujo, buffer_limpo, zi

    # Simula a chegada de novos dados do "sensor"
    if indice_global + VELOCIDADE_SIMULACAO >= len(sinal_entrada_sujo):
        indice_global = 0 # Loop do arquivo
    
    # Pega um bloco de dados sujos
    novos_pontos_sujos = sinal_entrada_sujo[indice_global : indice_global + VELOCIDADE_SIMULACAO]
    indice_global += VELOCIDADE_SIMULACAO
    
    # --- O CORAÇÃO DO FILTRO DIGITAL ---
    # Aplica o filtro com memória de estado (zi -> zf)
    novos_pontos_limpos, zf = signal.lfilter(b_notch, a_notch, novos_pontos_sujos, zi=zi)
    zi = zf # Atualiza a memória para o próximo ciclo
    # ------------------------------------

    # Atualiza buffers (efeito rolagem)
    buffer_sujo = np.roll(buffer_sujo, -VELOCIDADE_SIMULACAO)
    buffer_sujo[-VELOCIDADE_SIMULACAO:] = novos_pontos_sujos
    buffer_limpo = np.roll(buffer_limpo, -VELOCIDADE_SIMULACAO)
    buffer_limpo[-VELOCIDADE_SIMULACAO:] = novos_pontos_limpos
    
    # Atualiza o plot
    linha_suja.set_ydata(buffer_sujo)
    linha_limpa.set_ydata(buffer_limpo)
    return linha_suja, linha_limpa

print("Iniciando simulação. Prepare o print screen!")
ani = FuncAnimation(fig, atualizar, interval=30, blit=True, cache_frame_data=False)
plt.show()