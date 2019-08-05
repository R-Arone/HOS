# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:43:38 2019

@author: Rafael Arone
"""

import numpy as np
import scipy.signal as sg
import matplotlib.pyplot as plt
import scipy.stats as ss
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import scipy.fftpack as sfft
import random
random.seed()
#Gerando o sinal
fs = 256 #Hz - Frequência de Amostragem
f1 = 10#Hz
f2 = 30#Hz
f3 = 30
A1 = 10
A2 = 10
A3 = 10
phi1 = 0*np.pi*random.random()
phi2 = 0*np.pi*random.random()
t = np.linspace(0,5,5*fs)
x = A1*np.sin(2*np.pi*f1*t+phi1) + A2*np.sin(2*np.pi*f2*t+phi2)#+  A2*np.sin(2*np.pi*50*t+phi1+phi2)#+10*np.random.beta(1,2,size=len(t))
#x = x + x**3 #+ np.random.normal(0,5,len(t))
x = x-np.mean(x)

x = x = [1 if ((i%int(fs/10))==0) else 0 for i in range(len(t))]


plt.figure()

plt.plot(t,x)

#Periodograma - Estatísticas de Primeira Ordem
f,Sxx = sg.periodogram(x,fs)
plt.figure()
plt.plot(f,Sxx)
plt.title("Periodograma do Sinal")
plt.xlabel("Frequência [Hz]")
plt.ylabel("Potência [W/Hz]")


#Cálculo das estastísticas de segunda ordem

#Método 1: A partir da FFT
X  = sfft.fft(x,n=fs)/(5*fs)
plt.figure()
plt.plot(abs(X))
plt.xlabel("Frequência [Hz]")
plt.ylabel("Potência [W/Hz]")


def Bispectrum_indirect(X, fs):
    L = len(X)
    M = int(fs)
    window = sg.windows.dpss(M, 2.5)
    R = np.zeros([M,M])
    rxx = np.correlate(window*X[0:M],window*X[0:M],mode='full')
    for k in range(0,int(L/M)):
        Xa = window*X[k*M:(k+1)*M]
        for m in  range(0,M-1):
            for n in  range(0,M-1):
                r = 0;
                for l in  range(0,min([M-n-1,M-m-1])):
                    r = r + Xa[l]*Xa[l+m]*Xa[l+n]
                    #r = r - np.mean(X)*(rxx[M-1+m]+rxx[M-1+n]+rxx[M-1+m-n])+2*np.mean(X)**3
                    R[m,n] = R[m,n] +  r/float(M**2)
    Sxxx = np.fft.fft2(R/float(int(L/M)))
    Sxxx = Sxxx/M**2
    return Sxxx[0:int(M/2),0:int(M/2)]


def Bispectrum(X):
    n = int(len(X)/2)
    Sxxx = (np.zeros([n,n]))
    Sxxx = Sxxx.astype(complex)
    for f1 in range(n):
        #k = f1
        for f2 in range(n):
                Sxxx[f1,f2] = (np.conj(X[f1+f2])*(X[f1])*(X[f2]))
    return np.transpose(Sxxx)
    
    
    
Sxxx = Bispectrum(X)
plt.figure()
plt.pcolor(abs(Sxxx),cmap = plt.get_cmap('Greys'))
plt.colorbar()
plt.xlabel('f1 [Hz]')
plt.ylabel('f2 [Hz]')
plt.show()

Sxxxi = Bispectrum_indirect(x,fs)
plt.figure()
plt.pcolor(abs(Sxxxi),cmap = plt.get_cmap('Greys'))
plt.colorbar()
plt.xlabel('f1 [Hz]')
plt.ylabel('f2 [Hz]')
plt.show()

def Trispectrum_transform2plot(X,threshold):
    n = int(len(X)/3)
    Txxxx = []
    for f1 in range(n):
        for f2 in range(n):
            for f3 in range(n):
                value = np.conj(X[f1])*np.conj(X[f2])*np.conj(X[f3])*X[f1+f2+f3]
                if abs(value)>threshold:
                    Txxxx.append([f1,f2,f3,abs(value)])
    return Txxxx

def Trispectrum(X):
    n = int(len(X)/3)
    Txxxx = np.zeros([n,n,n])
    Txxxx = Txxxx.astype(complex)
    for f1 in range(n):
        for f2 in range(n):
            for f3 in range(n):
                Txxxx[f1,f2,f3] = abs(np.conj(X[f1])*np.conj(X[f2])*np.conj(X[f3])*X[f1+f2+f3])
    return Txxxx


def Trispectrum_indirect(X, fs,threshold):
    L = len(X)
    M = int(fs)
    window = sg.windows.dpss(M, 2.5)
    R = np.zeros([M,M,M])
    for k in range(0,int(L/M)):
        Xa = window*X[k*M:(k+1)*M]
        m1 = np.mean(Xa)
        m2 = sg.correlate(Xa,Xa,mode='full')
        #m2 = m2[M-1:]
        for m in  range(0,M-1):
            for n in  range(0,M-1):
                for z in range(0,M-1):
                    r = 0;
                    for l in  range(0,min([M-n-1,M-m-1,M-z-1])):
                        r = r + Xa[l]*Xa[l+m]*Xa[l+n]*Xa[l+z]
                    R[m,n,z] =R[m,n,z] +  r/float(M)
                    R[m,n,z] =  R[m,n,z] - m2[M-1+n]*m2[M-1+z-n]- m2[M-1+m]*m2[M-1+m-z]- m2[M-1+z]*m2[M-1+n - m]
    Sxxxx = np.fft.fftn(R/float(int(L/M)),axes = (0,1,2))
    Txxxx = []
    for i in range(len(Sxxxx[0])):
        for j in range(len(Sxxxx[1])):
            for k in range(len(Sxxxx[2])):
                value = Sxxxx[i,j,k]
                if value>=threshold:
                    Txxxx.append([f1,f2,f3,abs(value)])
    return Txxxx

Txxxx = Trispectrum(X)


Txxxx = np.transpose(Trispectrum_transform2plot(X,0.0000001))
scale = abs(Txxxx[3])
c = 100*(scale - min(scale))/(max(scale) - min(scale))
fig = plt.figure()
ax=fig.add_subplot(111,projection='3d')
plt.gca().patch.set_facecolor('white')

p = ax.scatter(abs(Txxxx[0]),abs(Txxxx[1]),abs(Txxxx[2]),s =10**7*(abs(Txxxx[3])),c = c,cmap = plt.get_cmap('YlGn'))
ax.set_xlabel('f1 [Hz]')
ax.set_ylabel('f2 [Hz]')
ax.set_zlabel('f3 [Hz]')
fig.colorbar(p)