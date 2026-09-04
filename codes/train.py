import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import PTBDBDataset
from model import DenoisingAutoencoder
from tqdm import tqdm # Para a barra de progresso bonitona

# --- 1. HIPERPARÂMETROS ---
WINDOW_SECONDS = 2
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50 # Quantas vezes a rede vai ver o banco inteiro

# --- 2. CONFIGURAÇÃO DE HARDWARE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Iniciando treinamento no dispositivo: {device}")

# --- 3. PREPARAÇÃO DOS DADOS ---
# Instancia o dataset completo
DIR_PTBDB = "/media/gerson/IAData/TCC/ecg_database/ptbdb"
dataset = PTBDBDataset(DIR_PTBDB, window_seconds=2)

# Divide o dataset: 80% para treinar, 20% para validar
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 4. INSTANCIANDO O MODELO ---
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss() # Erro Quadrático Médio
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 5. LOOP DE TREINAMENTO ---
print("Começando o aprendizado...")

for epoch in range(EPOCHS):
    # --- MODO DE TREINO ---
    model.train()
    train_loss = 0.0
    
    # Barra de progresso para o lote atual
    progresso = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Treino]")
    
    for noisy_sigs, clean_sigs in progresso:
        # Envia os dados para a CPU ou GPU
        noisy_sigs = noisy_sigs.to(device)
        clean_sigs = clean_sigs.to(device)

        # Zera os gradientes antigos
        optimizer.zero_grad()
        
        # Forward Pass (Tenta adivinhar/limpar o sinal)
        outputs = model(noisy_sigs)
        
        # Calcula o erro entre a previsão e o gabarito limpo
        loss = criterion(outputs, clean_sigs)
        
        # Backward Pass (Manda o erro de volta para corrigir os pesos)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * noisy_sigs.size(0)
        progresso.set_postfix({'Loss': f"{loss.item():.4f}"})

    # --- MODO DE VALIDAÇÃO ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad(): # Desliga o cálculo de gradientes para economizar memória
        for noisy_sigs, clean_sigs in val_loader:
            noisy_sigs = noisy_sigs.to(device)
            clean_sigs = clean_sigs.to(device)
            
            outputs = model(noisy_sigs)
            loss = criterion(outputs, clean_sigs)
            val_loss += loss.item() * noisy_sigs.size(0)

    # Imprime o resumo da Epoch
    avg_train_loss = train_loss / len(train_dataset)
    avg_val_loss = val_loss / len(val_dataset)
    print(f"Resumo da Epoch -> Loss Treino: {avg_train_loss:.4f} | Loss Validação: {avg_val_loss:.4f}\n")

# --- 6. SALVAR O MODELO ---
torch.save(model.state_dict(), "dcae_ecg_weights.pth")
print("Treinamento concluído! Pesos do modelo salvos em 'dcae_ecg_weights.pth'")