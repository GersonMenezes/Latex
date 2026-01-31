import wfdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import os

# --- CONFIGURAÇÕES ---
RECORD_NAME = '100'     # Nome do arquivo sem extensão (.dat e .hea devem estar na pasta)
JANELA_SEGUNDOS = 4     # Quantos segundos mostrar na tela
INTERVALO_ATT = 20      # Velocidade da animação (ms)
VELOCIDADE_SCROLL = 4   # Pontos por frame (ajuste se ficar muito rápido)

# --- 1. CARREGAMENTO DOS DADOS LOCAIS ---
print(f"Buscando arquivos locais: {RECORD_NAME}.dat e {RECORD_NAME}.hea ...")

if not os.path.exists(f"{RECORD_NAME}.dat") or not os.path.exists(f"{RECORD_NAME}.hea"):
    print(f"ERRO: Arquivos do registro '{RECORD_NAME}' não encontrados!")
    print("Por favor, baixe '100.dat' e '100.hea' do site PhysioNet (MIT-BIH Arrhythmia Database)")
    print("e coloque-os na mesma pasta deste script.")
    exit()

try:
    # O sinal está em record.p_signal e a frequência em record.fs
    record = wfdb.rdrecord(RECORD_NAME, channels=[0])
    
    # Verifica se o retorno é um objeto Record ou uma tupla (para compatibilidade)
    if isinstance(record, tuple):
        sinal = record[0]
        fs_real = record[1]['fs']
    else:
        sinal = record.p_signal
        fs_real = record.fs
    
    # O sinal vem como matriz (N, 1), transformamos em vetor simples
    ecg_full = sinal.flatten()
    
    print(f"Arquivo carregado com sucesso! Frequência: {fs_real} Hz")
    print(f"Total de amostras: {len(ecg_full)}")

except Exception as e:
    print(f"Erro ao ler arquivo: {e}")
    exit()

# --- 2. PREPARAÇÃO DA INTERFACE ---
# Calcula o tamanho do buffer baseado na frequência real do arquivo
tamanho_janela = int(JANELA_SEGUNDOS * fs_real)

dados_x = np.linspace(0, JANELA_SEGUNDOS, tamanho_janela)
dados_y = np.zeros(tamanho_janela)

# Configuração da Figura
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Plot inicial
linha, = ax.plot(dados_x, dados_y, 'g-', lw=1.8) # Linha verde tipo monitor

# Ajuste de eixos
ax.set_ylim(np.min(ecg_full), np.max(ecg_full)) # Ajusta escala vertical ao sinal real
ax.set_xlim(0, JANELA_SEGUNDOS)

# Textos
titulo = ax.set_title(f"Monitor ECG - (Tecle 'ESPAÇO': pausar/continuar e 'S': salvar)", fontsize=14, color='white')
ax.set_ylabel("Amplitude (mV)", color='white')
ax.set_xlabel("Tempo (s)", color='white')
ax.grid(True, color='green', linestyle=':', linewidth=0.5, alpha=0.5)

# Cores dos eixos
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Variáveis globais de controle
indice_atual = 0
pausado = False

# --- 3. CONTROLES (TECLADO) ---
def on_key(event):
    global pausado
    
    if event.key == ' ':
        pausado = not pausado
        estado = "PAUSADO" if pausado else "AO VIVO"
        cor = 'yellow' if pausado else 'white'
        titulo.set_text(f"Monitor ECG - {estado} (Tecle 'ESPAÇO': pausar/continuar e 'S': salvar)") 
        titulo.set_color(cor)
        plt.draw()
        
    if (event.key == 's' or event.key == 'S') and pausado:
        nome_arquivo = f"ecg_{RECORD_NAME}_{datetime.now().strftime('%H%M%S')}.png"
        plt.savefig(nome_arquivo, facecolor='black')
        print(f"Imagem salva: {nome_arquivo}")
        titulo.set_text(f"SALVO: {nome_arquivo}")
        titulo.set_color('cyan')
        plt.draw()

fig.canvas.mpl_connect('key_press_event', on_key)

# --- 4. FUNÇÃO DE ATUALIZAÇÃO (LOOP) ---
def atualizar(frame):
    global indice_atual, dados_y
    
    if pausado:
        return linha,
    
    # Loop infinito dos dados
    if indice_atual + VELOCIDADE_SCROLL >= len(ecg_full):
        indice_atual = 0
        
    # Pega novos pontos do arquivo carregado na memória
    novos_dados = ecg_full[indice_atual : indice_atual + VELOCIDADE_SCROLL]
    indice_atual += VELOCIDADE_SCROLL
    
    # Efeito de rolagem
    dados_y = np.roll(dados_y, -VELOCIDADE_SCROLL)
    dados_y[-VELOCIDADE_SCROLL:] = novos_dados
    
    linha.set_ydata(dados_y)
    return linha,

# --- 5. EXECUÇÃO ---
print("Iniciando visualização...")
print("Comandos: [BARRA DE ESPAÇO] = Pausar | [S] = Salvar (quando pausado)")

ani = FuncAnimation(fig, atualizar, interval=INTERVALO_ATT, blit=True, cache_frame_data=False)
plt.show()