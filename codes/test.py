import torch

print(f"Versão do PyTorch: {torch.__version__}")
if torch.cuda.is_available():
    print("Aceleração por GPU (CUDA): ATIVADA! O treinamento vai voar.")
else:
    print("Aceleração por GPU: Desativada. Rodando via CPU (Normal para notebooks).")