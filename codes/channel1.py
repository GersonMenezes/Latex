import wfdb
import matplotlib.pyplot as plt

# Nome do arquivo baixado do banco PTB
RECORD_NAME = 's0010_re'

# O parâmetro channels=[0] garante que apenas a derivação 'i' (D1) seja carregada na memória
record = wfdb.rdrecord(RECORD_NAME, channels=[2])

# Extrai a matriz unidimensional do sinal e a frequência de amostragem
sinal_d1_limpo = record.p_signal.flatten()
fs = record.fs  

print(f"Frequência de amostragem: {fs} Hz")
print(f"Derivação extraída: {record.sig_name[0]}")

# Visualizando os primeiros 3 segundos (3000 amostras, já que fs = 1000)
plt.figure(figsize=(10, 4))
plt.plot(sinal_d1_limpo[:3000], color='#1f77b4', linewidth=1.2)
plt.title(f"ECG Limpo - Derivação {record.sig_name[0].upper()} (Banco PTB)")
plt.xlabel("Amostras")
plt.ylabel("Amplitude (mV)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()