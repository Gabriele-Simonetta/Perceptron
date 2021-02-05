# Gabriele Simonetta
## Simple Perceptron Example
# basato sul codice del video di Daniel Shiffman, maggiori informazioni al link: https://www.youtube.com/watch?v=ntKn5TPHHAk&t=1976s
import time  # per vedere quanto impiega la rete neurale per giungere a soluzione
start = time.time()
import matplotlib.pyplot as plt  # per creare i grafici
import numpy as np
# Create a list of evenly-spaced numbers over the range
x = np.linspace(0, 10, 1000)  # larghezza finestra da 0 a 100 con 1000 punti totali
lr = 0.1  # learning rate
pt = 1000  # punti randomici generati
## Funzioni accessorie
def funzione(x):  # scegli la funzione che vuoi usare e commenta le altre
    #y = x**2+5*x+5  #il quadrato si fa con il doppio per
    # y = x ** 3 + 7 * x ** 2 + 5 * x - 4
    #y=5*x+2
    #y = np.sin(x)
    y = np.cos(x)


    return y

def activator(y):  # activator function
    if y > 0:
        target = 1
    else:
        target = -1
    return target
def rand(minimum, maximum, pt):  # definisce vettore di lunghezza pt compreso nel range di minumum e maximum
    r = np.array(minimum + (maximum - minimum) * np.random.rand(pt, 1))
    return r

y = funzione(x)
a = rand(min(x), max(x), pt)  # x variabili randomiche
b = rand(min(y), max(y), pt)  # y variabili randomiche
plt.plot(x, y)  # Plot the f(x) of each x point
plt.plot(a, b, marker=".", linestyle="")  # Plot point
plt.gca().legend(('f(x)', 'p.random'))
plt.show()  # Display the plot

# funzione per vedere nelle immagini se la rete non commette errori colorando tutti i punti di verde
def pltcolor(lst):
    cols = []
    for u in lst:
        if u == 0:
            cols.append('green')  # previsione corretta
        else:
            cols.append('red')  # previsione sbagliata
    return cols


##  pongo a valori noti le nuove variabili
yasp = np.zeros(len(b))
target = np.zeros(len(b))
active = np.zeros(len(b))
guess = np.zeros(len(b))
error = np.zeros(len(b))
accuracy = np.zeros(len(b))
accuratezza = 0
errore = False
epochs = 1  # tiene in memoria quante volte il ciclo while si ripete
# pesi
weights_a = np.random.rand(len(b))
weights_b = np.random.rand(len(b))
##Neurone
while abs(errore) != True:
    for i in range(len(b)):
        yasp[i] = funzione(a[i])  # calcola la y relativa alla funzione
        if b[i] > yasp[i]:  # se il valore delle y randomiche è maggiore della y=f(x) allora assegna 1
            target[i] = 1
        else:
            target[i] = -1
        active[i] = a[i] * weights_a[i] + b[i] * weights_b[i]  # funzione della cellula neurale
        guess[i] = activator(active[i])  # attivatore
        error[i] = target[i] - guess[i]  # calcolo errore
        if error[i] == 0:
            accuracy[i] = 1
        else:
            accuracy[i] = 0
        # backward correzione errori
        weights_a[i] = weights_a[i] + error[i] * a[i] * lr
        weights_b[i] = weights_b[i] + error[i] * b[i] * lr
    # Visualizzazione grafica della progressione della rete neurale
    accuratezza = (sum(accuracy) / len(b)) * 100
    cols = pltcolor(error)
    plt.scatter(a, b, c=cols, marker=".")  # Pass on the list created by the function here
    plt.plot(x, y, )  # Plot the sine of each x point
    plt.annotate('accuracy: ' + str(round(accuratezza, 2)) + ' %', xy=(1, 0), xycoords='axes fraction', fontsize=11,
                 xytext=(0, -17), textcoords='offset points',
                 ha='right', va='top')
    plt.annotate('epochs: ' + str(round(epochs, 2)), xy=(0, 0), xycoords='axes fraction', fontsize=11,
                 xytext=(0, -17), textcoords='offset points',
                 ha='right', va='top')
    plt.show()
    errore = all(value == 0 for value in error)
    epochs = epochs + 1
end = time.time()
print('tempo impiegato:' + str(round(end - start, 2)) + ' [sec] \nepochs:'+ str(epochs)) # stampa a schermo il tempo impiegato e le epoche