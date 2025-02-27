import numpy as np

def sampandquant(x, nbits, td, ts):
    # x: sinal de entrada
    # nbits: número de bits do quantizador
    # td: taxa de amostragem do sinal de entrada
    # ts: taxa de amostragem do sinal de saída
    # s_out: sinal de saída
    
    # Quantização
    xmax = max(x) # valor máximo do sinal
    xmin = min(x) # valor mínimo do sinal
    L = 2**nbits # número de níveis de quantização

    Delta = (xmax - xmin)/L # intervalo de quantização
    xq = np.round(x/Delta)*Delta # sinal quantizado
    xq = np.clip(xq, xmin, xmax) # limita a amplitude do sinal quantizado entre xmin e xmax

    # Subamostragem ou discretização
    Nfactor = int(ts/td) # fator de subamostragem
    s_out = np.zeros(len(x)) # o sinal terá o mesmo comprimento do sinal de entrada
    s_out[0::Nfactor] = xq[0::Nfactor] # sinal amostrado
    sq_out = np.zeros(len(x)) # o sinal terá o mesmo comprimento do sinal de entrada
    sq_out[0::Nfactor] = xq[0::Nfactor] # sinal amostrado e quantizado com 0s
    
    # Reconstrução com zero-order hold
    sq_out_ = xq[0::Nfactor] # sinal amostrado e quantizado sem 0s
    sqh_out = np.repeat(sq_out_, Nfactor) # reconstrução do sinal com zero-order
    sqh_out = sqh_out[:len(x)] # ajusta o comprimento do sinal

    # Cálculo do SQNR
    eq = x - sqh_out # sinal de erro
    SQNR = 10*np.log10(np.var(x)/np.var(eq)) # cálculo do SQNR para um sinal com média zero

    return s_out, sq_out, sqh_out, Delta, SQNR