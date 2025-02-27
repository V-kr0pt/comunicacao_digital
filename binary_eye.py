import numpy as np
from function_utils import prz, pnrz, psine, prcos
from matplotlib import pyplot as plt
from eyediagram import eyediagram


data = np.sign(np.random.randn(400)) # gera 400 bits aleatórios
Tau = 64 # período de símbolo

# upsampling
dataup = np.zeros(Tau*len(data)) # inicializa vetor de zeros
dataup[::Tau] = data # amostra os bits

yrz = np.convolve(dataup, prz(Tau)) # sinal polar com retorno à zero
yrz = yrz[:-Tau+1]

ynrz = np.convolve(dataup, pnrz(Tau)) # sinal polar sem retorno à zero
ynrz = ynrz[:-Tau+1]

ysine = np.convolve(dataup, psine(Tau)) # sinal polar de meia senoide
ysine = ysine[:-Tau+1]

Td=4 # trunca cosseno levantado em 4 períodos
yrcos = np.convolve(dataup, prcos(0.5, Td, Tau)) # sinal polar com cosseno levantado fator de decaimento 0.5
yrcos = yrcos[2*Td*Tau:-2*Td*Tau+1] # trunca o sinal

fig1, ax1 = eyediagram(yrz, 2*Tau, Tau, Tau//2)
fig2, ax2 = eyediagram(ynrz, 2*Tau, Tau, Tau//2)
fig3, ax3 = eyediagram(ysine, 2*Tau, Tau, Tau//2)
fig4, ax4 = eyediagram(yrcos, 2*Tau, Tau, Tau//2)

ax1.set_xlabel('Tempo [s]')
ax1.set_ylabel('Amplitude')
ax1.grid()
ax1.set_title('Diagrama de olho RZ')
ax2.set_xlabel('Tempo [s]')
ax2.set_ylabel('Amplitude')
ax2.grid()
ax2.set_title('Diagrama de olho NRZ')
ax3.set_xlabel('Tempo [s]')
ax3.set_ylabel('Amplitude')
ax3.grid()
ax3.set_title('Diagrama de olho de meia senoide')
ax4.set_xlabel('Tempo [s]')
ax4.set_ylabel('Amplitude')
ax4.grid()
ax4.set_title('Diagrama de olho de cosseno levantado')


fig1.savefig('images/binary_eyediagram_RZ.png')
fig2.savefig('images/binary_eyediagram_NRZ.png')
fig3.savefig('images/binary_eyediagram_senoide.png')
fig4.savefig('images/binary_eyediagram_cosseno.png')