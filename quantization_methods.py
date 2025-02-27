import numpy as np

def old_sampandquant(x, nbits, td, ts):
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


def uniquan(sig_in, L):
    '''
    L - número de níveis de quantização uniforme
    sig_in- vetor para sinal de entrada

    Retorna:
    q_out -  saída quantizada
    Delta - intervalo de quantização
    SQNR - razão sinal ruído de quantização
    '''

    # Encontra o valor máximo e mínimo do sinal
    sig_pmax = np.max(sig_in)
    sig_nmax = np.min(sig_in)

    # Calcula o intervalo de quantização
    Delta = (sig_pmax - sig_nmax)/L

    # define os níveis de quantização
    q_level = np.arange(sig_nmax + Delta/2, sig_pmax, Delta)
    #L_sig = len(sig_in) # comprimento do sinal
    sigp = (sig_in-sig_nmax)/Delta + 1/2 # mapeamento do sinal para o intervalo [1/2, L+1/2]
    qindex = np.round(sigp).astype(int) # índices dos níveis de quantização
    qindex = qindex-1 # ajusta o índice para o intervalo [0, L-1]
    qindex = np.clip(qindex, 0, L-1) # limita o índice ao número de níveis (negativos tornam-se 0)
    q_out = q_level[qindex] # sinal quantizado
    SQNR = 20*np.log10(np.linalg.norm(sig_in)/np.linalg.norm(sig_in-q_out)) # cálculo do SQNR

    return q_out, Delta, SQNR

def sampandquant(sig_in, L, td, ts):
    '''
    sig_in - sinal de entrada
    L - número de níveis de quantização
    td - período de amostragem
    ts - período de amostragem
    Retorna:
    s_out - sinal amostrado
    sq_out - sinal amostrado e quantizado
    sqh_out - sinal amostrado e quantizado com zero-order hold
    Delta - intervalo de quantização
    SQNR - razão sinal ruído de quantização
    '''

    nfac = int(ts/td) # fator de subamostragem
    p_zoh = np.ones(nfac) # pulso de amostragem
    s_out_ = sig_in[::nfac] # sinal amostrado
    sq_out_, Delta, SQNR = uniquan(s_out_, L) # sinal amostrado e quantizado
    sqh_out = np.kron(sq_out_, p_zoh) # reconstrução do sinal com zero-order hold
    
    # upsampling
    s_out = np.zeros(len(sig_in)) # sinal amostrado
    s_out[::nfac] = s_out_ # sinal amostrado
    sq_out = np.zeros(len(sig_in)) # sinal amostrado e quantizado com zeros
    sq_out[::nfac] = sq_out_ # sinal amostrado e quantizado com zeros
    
    return s_out, sq_out, sqh_out, Delta, SQNR

def deltamod(sig_in, Delta, td, ts):
    '''
    sig_in - sinal de entrada
    Delta - incremento de quantização
    td - período de amostragem de sinal de sig_in
    ts - novo período de amostragem
    Retorna:
    s_DMOut - saída DM amostrada
    '''

    nfac = int(ts/td) # fator de subamostragem
    p_zoh = np.ones(nfac) # pulso de amostragem
    s_out_ = sig_in[::nfac] # sinal amostrado
    Num_it = len(s_out_) # número de iterações
    s_DMout = [-Delta/2] # sinal de saída

    for k in range(1, Num_it):
        xvar = s_DMout[k-1] # valor anterior
        signal = np.sign(s_out_[k-1]-xvar) # soma ou subtração de delta
        s_DMout.append(xvar + Delta*signal) # valor atual

    # upsampling
    s_DMOut = np.kron(s_DMout, p_zoh) # reconstrução do sinal com zero-order hold

    return s_DMOut