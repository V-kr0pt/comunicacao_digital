import numpy as np
import matplotlib.pyplot as plt
from quantization_methods import sampandquant

td = 0.002 # taxa de amostragem de 500 Hz
t = np.arange(0, 1, td) # vetor de tempo de 0 a 1 segundos com passo de 0.002 segundos
xsig = np.sin(2*np.pi*1*t) - np.sin(2*np.pi*3*t) # sinal senoidal com frequências de 1 Hz e 3 Hz

Lsig = len(xsig) # comprimento do sinal
Lfft = 2**np.ceil(np.log2(Lsig)+1) # comprimento da FFT, o dobro do comprimento do sinal
Fmax = 1/(2*td) # frequência máxima Nyquist

Xsig = np.fft.fft(xsig, int(Lfft)) # transformada de Fourier do sinal original
Xsig = np.fft.fftshift(Xsig) # rearranja a transformada de Fourier

Faxis = np.linspace(-Fmax, Fmax, int(Lfft)) # eixo da frequência
ts = 0.02 # nova taxa de amostragem de 50 Hz
#Nfact = ts/td # fator de subamostragem

# envia o sinal para um quantizador uniforme de 16 níveis
s_out, sq_out, sqh_out1, Delta, SQNR = sampandquant(xsig, 16, td, ts)

# a mesma coisa porém para PCM 4 níves
s_out, sq_out, sqh_out2, Delta, SQNR = sampandquant(xsig, 4, td, ts)



# gráfico do sinal original e do sinal PCM no tempo
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(t, xsig, 'b', label='Sinal original')
plt.plot(t, sqh_out1, 'r', label='Sinal PCM (L=16)')
plt.title('Sinal original g_t e correspondente PCM 16 níveis')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(t, xsig, 'b', label='Sinal original')
plt.plot(t, sqh_out2, 'r', label='Sinal PCM (L=4)')
plt.title('Sinal original g_t e correspondente PCM 4 níveis')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig('images/sinal_PCM.png')


# transformada de fourier do sinal PCM 
SQH1 = np.fft.fft(sqh_out1, int(Lfft))
SQH1 = np.fft.fftshift(SQH1) # transformada de Fourier do sinal PCM
SQH2 = np.fft.fft(sqh_out2, int(Lfft))
SQH2 = np.fft.fftshift(SQH2) # transformada de Fourier do sinal PCM

# usa passa-baixas para filtrar os dois sinais PCM
BW = 10 # largura de banda do filtro passa-baixa
H_1pf = np.zeros(int(Lfft))
H_1pf[int(Lfft/2-BW):int(Lfft/2+BW)] = 1 # filtro passa-baixa ideal

# sinal PCM L=16 filtrado
S1_recv = SQH1*H_1pf # sinal PCM filtrado
s_recv1 = np.real(np.fft.ifft(np.fft.ifftshift(S1_recv))) # sinal PCM filtrado no tempo
s_recv1 = s_recv1[:Lsig] # ajusta o comprimento do sinal

# sinal PCM L=4 filtrado
S2_recv = SQH2*H_1pf # sinal PCM filtrado
s_recv2 = np.real(np.fft.ifft(np.fft.ifftshift(S2_recv))) # sinal PCM filtrado no tempo
s_recv2 = s_recv2[:Lsig] # ajusta o comprimento do sinal

# gráfico do sinal original e do sinal PCM filtrado no tempo
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(t, xsig, 'b', label='Sinal original')
plt.plot(t, s_recv1, 'r--', label='Sinal PCM filtrado (L=16)')
plt.title('Sinal original g_t e correspondente PCM filtrado 16 níveis')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(t, xsig, 'b', label='Sinal original')
plt.plot(t, s_recv2, 'r--', label='Sinal PCM filtrado (L=4)')
plt.title('Sinal original g_t e correspondente PCM filtrado 4 níveis')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig('images/sinal_PCM_filtrado.png')

