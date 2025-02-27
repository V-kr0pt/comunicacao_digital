import numpy as np
from scipy.signal import firwin
def pnrz(T):
    '''
    gera um pulso retangular de largura T e altura 1

    retorna:
    pout: np.array com o pulso retangular
    '''
    pout = np.ones(T)
    return pout

def prz(T):
    '''
    gera um pulso retangular de largura T/2 e altura 1

    retorna:
    pout: np.array com o pulso retangular
    '''
    pout = [np.zeros(T//4), np.ones(T//2), np.zeros(T//4)]
    pout = np.concatenate(pout)
    return pout

def psine(T):
    '''
    gera um pulso senoidal de período T

    retorna:
    pout: np.array com o pulso senoidal
    '''
    t = np.linspace(0, 2*np.pi, T)
    pout = np.sin(t)
    return pout


def rcosfir_firwin(rolloff, length, T, fs):
    """
    Implementação do filtro FIR de cosseno levantado usando firwin.
    
    rolloff: Fator de roll-off (entre 0 e 1)
    length: Número total de coeficientes do filtro (deve ser ímpar)
    T: Período do símbolo
    fs: Frequência de amostragem
    """
    nyquist = fs / 2  # Frequência de Nyquist
    cutoff = (1 + rolloff) / (2 * T)  # Frequência de corte normalizada

    # Criação do filtro FIR usando a janela de Hamming 
    h = firwin(length, cutoff / nyquist, window='hamming', scale=True)

    return h

def prcos(rollfact, length, T):
    '''
    gera um pulso cosseno levantado de período T

    rollfact: fator de roll-off
    length: comprimento do pulso
    T: período do pulso

    retorna:
    pout: np.array com o pulso cosseno levantado
    '''
    pout=rcosfir_firwin(rollfact, length, T, 1)

    return pout