import os
import torch
import wfdb
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class PTBDBDataset(Dataset):
    def __init__(self, base_dir, window_seconds=2):
        """
        base_dir: Caminho raiz contendo as pastas dos pacientes (ex: patient001, patient002...)
        window_seconds: Tamanho do fragmento em segundos
        """
        self.base_dir = base_dir
        self.fs = 1000  # Frequência de amostragem cravada do PTBDB
        self.window_size = int(window_seconds * self.fs)
        self.windows = []
        
        # Mapeia todos os registros válidos navegando pelas subpastas
        record_paths = []
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                if file.endswith('.hea'):
                    # O wfdb exige o caminho sem a extensão
                    record_path = os.path.join(root, file[:-4])
                    record_paths.append(record_path)
        
        print(f"Encontrados {len(record_paths)} registros. Fatiando derivação D1...")
        
        # Barra de progresso para a leitura dos arquivos
        for path in tqdm(record_paths, desc="Carregando PTBDB na RAM"):
            try:
                record = wfdb.rdrecord(path, channels=[0])
                sinal_continuo = record.p_signal.flatten()
                sinal_continuo = np.nan_to_num(sinal_continuo)
                
                # Fatiamento do sinal contínuo em janelas de 2 segundos
                num_windows = len(sinal_continuo) // self.window_size
                for i in range(num_windows):
                    inicio = i * self.window_size
                    fim = inicio + self.window_size
                    self.windows.append(sinal_continuo[inicio:fim])
            except Exception:
                # Pula silenciosamente arquivos corrompidos ou fora do padrão
                continue

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        sinal_limpo = self.windows[idx]
        t = np.arange(self.window_size) / self.fs
        
        # Injeção de Ruído Não-Estacionário (Data Augmentation)
        # 1. Rede Elétrica (60 Hz)
        amp_60hz = np.random.uniform(0.05, 0.2)
        ruido_60hz = amp_60hz * np.sin(2 * np.pi * 60 * t)
        
        # 2. Artefato de Movimento / Flutuação de Linha de Base (0.1Hz a 0.5Hz)
        amp_bw = np.random.uniform(0.1, 0.4)
        freq_bw = np.random.uniform(0.1, 0.5)
        ruido_bw = amp_bw * np.sin(2 * np.pi * freq_bw * t)
        
        # Adição de ruído branco de altíssima frequência (mimetizando EMG/músculo)
        ruido_emg = np.random.normal(0, 0.02, self.window_size)
        
        sinal_sujo = sinal_limpo + ruido_60hz + ruido_bw + ruido_emg
        
        # Formatação para Conv1D: (Canais, Comprimento)
        tensor_sujo = torch.tensor(sinal_sujo, dtype=torch.float32).unsqueeze(0)
        tensor_limpo = torch.tensor(sinal_limpo, dtype=torch.float32).unsqueeze(0)
        
        return tensor_sujo, tensor_limpo

# --- TESTE RÁPIDO DO CARREGAMENTO EM LOTE ---
if __name__ == "__main__":
    caminho_ptbdb = "/media/gerson/IAData/TCC/ecg_database/ptbdb"
    
    dataset_completo = PTBDBDataset(caminho_ptbdb, window_seconds=2)
    print(f"\nTotal de janelas de 2 segundos geradas: {len(dataset_completo)}")
    
    loader = DataLoader(dataset_completo, batch_size=16, shuffle=True)
    sujo_batch, limpo_batch = next(iter(loader))
    print(f"Formato do Lote Sujo: {sujo_batch.shape}")