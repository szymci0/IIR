import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

# Filtr
N = 8
fc = 1670 / (48000 / 2)
char = 'lowpass'
type = 'butter'

b, a = sig.iirfilter(N, fc, btype=char, ftype=type)
sos = sig.iirfilter(N, fc, btype=char, ftype=type, output='sos')
z, p, k = sig.iirfilter(N, fc, btype=char, ftype=type, output='zpk')
sos_ = sos.copy()
print(f"k = {k}")
print(f"sosik: {sos}")

sos1 = sos_[:1, :]
b1, a1 = sig.sos2tf(sos1)
_, h1 = sig.freqz(b1, a1)
k1 = np.max(np.abs(h1))
print('k1 = ', k1)
# dzielimy współczynniki b sekcji 1 przez k1
sos_[0, :3] /= k1
print(f"sos1: {sos_}")

# Teraz bierzemy pierwsze dwie sekcje SOS
sos2 = sos_[:2, :]
# Obliczamy maksymalny poziom charakterystyki amplitudowej
b2, a2 = sig.sos2tf(sos2)
_, h2 = sig.freqz(b2, a2)
k2 = np.max(np.abs(h2))
print('k2 = ', k2)
# dzielimy współczynniki b sekcji 2 przez k2
sos_[1, :3] /= k2
print(f"sos2: {sos_}")

# Następnie pierwsze trzy sekcje SOS
sos3 = sos_[:3, :]
# Obliczamy maksymalny poziom charakterystyki amplitudowej
b3, a3 = sig.sos2tf(sos3)
_, h3 = sig.freqz(b3, a3)
k3 = np.max(np.abs(h3))
print('k3 = ', k3)
# dzielimy współczynniki b sekcji 3 przez k3
sos_[2, :3] /= k3
print(f"sos3: {sos_}")

# I została jeszcze ostatnia sekcja
sos4 = sos_[:4, :]
# Obliczamy maksymalny poziom charakterystyki amplitudowej
b4, a4 = sig.sos2tf(sos4)
_, h4 = sig.freqz(b4, a4)
k4 = np.max(np.abs(h4))
print('k4 = ', k4)
# dzielimy współczynniki b sekcji 4 przez k4
sos_[3, :3] /= k4
print(f"sos4: {sos_}")

# Sprawdźmy maksymalne wzmocnienie całego filtru
b, a = sig.sos2tf(sos_)
_, h = sig.freqz(b, a)
k = np.max(np.abs(h))
print('k = ', k)

# Kwantyzacja współczynników do formatu Q15
# qsos = np.around(((2 ** 15) - 1) * sos_).astype(np.int16)
qsos = np.around(16384 * sos_).astype(np.int16)
print(f"Q15: {qsos}")

# Sprawdzenie maksymalnego wzmocnienia po każdej sekcji (po kwantyzacji)
for i in range(1, 5):
    bi, ai = sig.sos2tf(qsos[:i, :])
    _, hi = sig.freqz(bi, ai)
    ki = np.max(np.abs(hi))
    print('k{} = {}'.format(i, ki))

# Sprawdzenie czy filtr jest stabilny - moduł biegunów nie może przekraczać 1
z, p, k = sig.sos2zpk(qsos)
print(np.abs(p))

# Wykreślmy jeszcze charakterystykę obu filtrów
w, hn = sig.freqz(*sig.sos2tf(sos))
w = w * 48000 / (2 * np.pi) / 1000
hn = 20 * np.log10(np.abs(hn))
_, hq = sig.freqz(*sig.sos2tf(qsos))
hq = 20 * np.log10(np.abs(hq))
fig, ax = plt.subplots()
ax.plot(w, hn, color='b', linewidth=2, label='Zmiennoprzecinkowy')
ax.plot(w, hq, color='r', linewidth=2, label='Stałoprzecinkowy')
ax.set_xlabel('Częstotliwość [kHz]')
ax.set_ylabel('Amplituda [dB]')
ax.legend()
plt.show()

