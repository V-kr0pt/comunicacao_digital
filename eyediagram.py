import numpy as np
import matplotlib.pyplot as plt

def eyediagram(x, n, period=1, offset=0):
    """
    Gera um diagrama de olho semelhante ao MATLAB.

    Parâmetros:
    x : array-like
        Sinal de entrada.
    n : int
        Número de amostras por traço.
    period : float, opcional
        Período do traço. Default é 1.
    offset : int, opcional
        Deslocamento em amostras. Default é 0.
     Retorna:
    fig : objeto Figure do Matplotlib
        Figura do diagrama de olho.
    ax : objeto Axes do Matplotlib
        Eixo do diagrama de olho.
    """
    # Verifica se o offset está dentro do intervalo permitido
    if not (0 <= offset < n):
        raise ValueError("O offset deve estar no intervalo [0, n-1].")

    # Calcula o número de traços possíveis
    num_traces = (len(x) - offset) // n

    # Cria uma matriz para armazenar os traços
    traces = np.zeros((num_traces, n))

    # Preenche a matriz com os traços do sinal
    for i in range(num_traces):
        start_idx = offset + i * n
        end_idx = start_idx + n
        traces[i, :] = x[start_idx:end_idx]

    # Cria o eixo de tempo para um traço
    t = np.linspace(-period / 2, period / 2, n)

    # Plota o diagrama de olho
    fig, ax = plt.subplots(figsize=(10, 6))
    for trace in traces:
        ax.plot(t, trace, color='b', alpha=0.5)

    return fig, ax
    #plt.title("Diagrama de Olho")
    #plt.xlabel("Tempo")
    #plt.ylabel("Amplitude")
    #plt.grid(True)
    #plt.show()

