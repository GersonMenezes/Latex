import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataset import PTBDBDataset
from model import DenoisingAutoencoder

# 1. Configurações e Carregamento do Modelo
device = torch.device("cpu")
modelo = DenoisingAutoencoder().to(device)

modelo.load_state_dict(torch.load("dcae_ecg_weights.pth", map_location=device, weights_only=True))
modelo.eval() 

# 2. Preparação dos Dados
DIR_PTBDB = "/media/gerson/IAData/TCC/ecg_database/ptbdb"
dataset = PTBDBDataset(DIR_PTBDB, window_seconds=2)
sinal_sujo_tensor, sinal_limpo_tensor = dataset[0]
sinal_sujo_batch = sinal_sujo_tensor.unsqueeze(0).to(device)

# 3. Inferência
with torch.no_grad(): 
    sinal_reconstruido_tensor = modelo(sinal_sujo_batch)

# 4. Convertendo para Arrays Numpy
sinal_sujo = sinal_sujo_tensor.squeeze().numpy()
sinal_limpo = sinal_limpo_tensor.squeeze().numpy()
sinal_reconstruido = sinal_reconstruido_tensor.squeeze().numpy()
t = np.arange(len(sinal_limpo)) / 1000.0

# 5. Exportação dos Dados Digitais para CSV
dados_exportacao = np.column_stack((t, sinal_limpo, sinal_sujo, sinal_reconstruido))
np.savetxt(
    "analise_sinais_ecg.csv", 
    dados_exportacao, 
    delimiter=",", 
    header="Tempo(s),Sinal_Limpo(mV),Sinal_Sujo(mV),Sinal_Reconstruido(mV)", 
    comments="",
    fmt="%.6f"
)
print("Dados exportados com sucesso para 'analise_sinais_ecg.csv'!")

# 6. Plotagem Gráfica (com linhas mais finas para ver o artefato)
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.patch.set_facecolor('#f8f9fa')
plt.subplots_adjust(hspace=0.4)

axs[0].plot(t, sinal_limpo, color='#2ca02c', linewidth=0.8)
axs[0].set_title("Sinal Original Limpo", fontweight='bold')
axs[0].set_ylabel("Amplitude (mV)")
axs[0].grid(True, linestyle='--', alpha=0.6)

axs[1].plot(t, sinal_sujo, color='#d62728', linewidth=0.8)
axs[1].set_title("Sinal Contaminado", fontweight='bold')
axs[1].set_ylabel("Amplitude (mV)")
axs[1].grid(True, linestyle='--', alpha=0.6)

axs[2].plot(t, sinal_reconstruido, color='#1f77b4', linewidth=0.8)
axs[2].set_title("Sinal Reconstruído pelo Autoencoder", fontweight='bold')
axs[2].set_xlabel("Tempo (segundos)", fontweight='bold')
axs[2].set_ylabel("Amplitude (mV)")
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.show()