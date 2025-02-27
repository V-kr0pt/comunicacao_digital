import numpy as np
import matplotlib.pyplot as plt
from quantization_methods import deltamod

td = 0.002 # taxa de amostragem 500Hz
t = np.arange(0, 1, td)
xsig = np.sin(2*np.pi*1*t) - np.sin(2*np.pi*3*t) # sinal senoidal

Lsig = len(xsig)
ts = 0.02 # nova taxa de amostragem 50Hz
Nfact = ts/td


Delta1 = 0.2 # selecionando um pequeno delta
s_DMout1 = deltamod(xsig, Delta1, td, ts)

Delta2 = 2*Delta1 # dobrando delta1
s_DMout2 = deltamod(xsig, Delta2, td, ts)

Delta3 = 2*Delta2 # dobrando delta2
s_DMout3 = deltamod(xsig, Delta3, td, ts)

# gráficos
plt.figure(1, figsize=(10, 8))
plt.subplot(311)
plt.plot(t, xsig, 'b', label='sinal original')
plt.plot(t, s_DMout1, 'r--', label=f'sinal quantizado delta')
plt.title(f'Sinal original e quantizado com delta = {Delta1}')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.subplot(312)
plt.plot(t, xsig, 'b', label='sinal original')
plt.plot(t, s_DMout2, 'r--', label=f'sinal quantizado delta')
plt.title(f'Sinal original e quantizado com delta = {Delta2}')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.subplot(313)
plt.plot(t, xsig, 'b', label='sinal original')
plt.plot(t, s_DMout3, 'r--', label=f'sinal quantizado delta')
plt.title(f'Sinal original e quantizado com delta = {Delta3}')
plt.xlabel('Tempo [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig('images/sinal_original_reconstruido_delta.png')