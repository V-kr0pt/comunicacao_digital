import numpy as np
import matplotlib.pyplot as plt
from quantization_methods import sampandquant   

td = 0.002 # taxa de amostragem de 500 Hz
t = np.arange(0, 1, td) # vetor de tempo de 0 a 1 segundos com passo de 0.002 segundos
xsig = np.sin(2*np.pi*1*t) - np.sin(2*np.pi*3*t) # sinal senoidal com frequências de 1 Hz e 3 Hz
Lsig = len(xsig) # comprimento do sinal

ts= 0.02 # nova taxa de amostragem de 50 Hz
Nfactor = ts/td # fator de subamostragem

# envia o sinal por meio de um quatizador uniforme de 16 bits
s_out, sq_out, sqh_out, Delta, SQNR = sampandquant(xsig, 16, td, ts)

# Calcula a transformada de Fourier
Lfft = 2**np.ceil(np.log2(Lsig)+1) # comprimento da FFT, o dobro do comprimento do sinal
Fmax = 1/(2*td) # frequência máxima Nyquist
Faxis = np.linspace(-Fmax, Fmax, int(Lfft)) # eixo da frequência
Xsig = np.fft.fft(xsig, int(Lfft)) # transformada de Fourier do sinal original
Xsig = np.fft.fftshift(Xsig) # rearranja a transformada de Fourier
S_out = np.fft.fft(s_out, int(Lfft)) # transformada de Fourier do sinal reconstruído
S_out = np.fft.fftshift(S_out) # rearranja a transformada de Fourier

# Plota o sinal original e o sinal reconstruído
plt.figure(figsize=(10, 8))
plt.subplot(3,1,1)
plt.plot(t, xsig, 'b', label='Sinal original')
plt.plot(t, s_out, 'r', label='Sinal reconstruído')
plt.title('Sinal g(t) e suas amostras uniformes')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

# Plota o espectro do sinal original 
plt.subplot(3,1,2)
plt.plot(Faxis, np.abs(Xsig), 'b', label='Sinal original')
plt.title('Espectro do sinal g(t)')
plt.xlabel('Frequência [Hz]')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()

# Plota o espectro do sinal reconstruído
plt.subplot(3,1,3)
plt.plot(Faxis, np.abs(S_out), 'r', label='Sinal reconstruído')
plt.title('Espectro do sinal reconstruído g_T(t)')
plt.xlabel('Frequência [Hz]')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig('images/sinal_original_reconstruido_sampleado.png')

# calcula sinal obtido a partir da amostragem ideal e low pass filter ideal
BW = 10 # largura de banda do filtro passa-baixa 
# filtro ideal passa-baixa
H_1pf = np.zeros(int(Lfft))
H_1pf[int(Lfft/2-BW):int(Lfft/2+BW)-1] = 1

# filtragem
S_recv = Nfactor * S_out * H_1pf
s_recv = np.real(np.fft.ifft(np.fft.ifftshift(S_recv))) # sinal reconstruído
s_recv = s_recv[0:Lsig] # ajusta o comprimento do sinal

# Plota o sinal original e o sinal reconstruído
figure = plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(Faxis, np.abs(S_recv), 'r', label='Sinal reconstruído')
plt.title('Espectro de filtragem ideal')
plt.xlabel('Frequência [Hz]')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(t, xsig, 'b-.', label='Sinal original')
plt.plot(t, s_recv, 'r', label='Sinal reconstruído')
plt.title('Sinal original e sinal reconstruído idealmente')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('images/sinal_original_reconstruido_filtrado_ideal.png')

# subamostragem e retenção de ordem zero
ZOH = np.ones(int(Nfactor))
s_ni = np.kron(s_out[::int(Nfactor)], ZOH) # sinal reconstruído com subamostragem e retenção de ordem zero
S_ni = np.fft.fft(s_ni, int(Lfft)) # transformada de Fourier do sinal reconstruído
S_ni = np.fft.fftshift(S_ni) # rearranja a transformada de
S_recv2 = S_ni*H_1pf # sinal reconstruído com subamostragem e retenção de ordem zero e filtragem ideal
s_recv2 = np.real(np.fft.ifft(np.fft.ifftshift(S_recv2))) # sinal reconstruído
s_recv2 = s_recv2[0:Lsig] # ajusta o comprimento do sinal

# Plota o sinal original e o sinal reconstruído
figure = plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(t, xsig, 'b', label='Sinal original')
plt.plot(t, s_ni, 'r', label='Sinal reconstruído')
plt.title('Sinal original e sinal reconstruído com pulso retangular')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(t, xsig, 'b', label='Sinal original')
plt.plot(t, s_recv2[0:Lsig], 'b--', label='Sinal reconstruído')
plt.title('Sinal original e sinal reconstruído com pulso retangular após LPF')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('images/sinal_original_reconstruido_pulso_retangular.png')