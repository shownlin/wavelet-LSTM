import pywt
import matplotlib.pyplot as plt

wavelets = ['haar', 'db3', 'coif3', 'sym3']
for i in range(len(wavelets)):
    wavelet = pywt.Wavelet(wavelets[i])
    phi, psi, x = wavelet.wavefun()
    plt.subplot(2, 4, i+1)
    plt.plot(x, psi)
    plt.title(wavelets[i])
    plt.subplot(2, 4, i+5)
    plt.plot(x, phi)
    plt.title(wavelets[i])

plt.show()
