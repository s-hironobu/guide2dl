# Developed environment:
#  Python                   3.9.13
#  pip                      23.1.2
#  conda                    22.11.1
#  numpy                    1.23.3
#  matplotlib               3.6.0
#
#   Copyright (c) 2024, Hironobu Suzuki @ interdb.jp

#import numpy as np
import matplotlib.pyplot as plt
import DataSet as ds

# start-end dot
def plot_fig3(wave, start=0, end=25):
    for i in range(start, end):
        plt.plot(i - start, wave[i], marker='.', color="b")

    i = end
    plt.plot(i - start, wave[i], marker='.', color="r")
    #plt.legend()
    plt.ylim([-1.5, 1.5])
    plt.xlim([-10, 110])
    #plt.grid(True)
    plt.show()

# start-end dot
def plot_fig4(wave, start=0, end=25, input_end=25):
    for i in range(start, end):
        plt.plot(i, wave[i], marker='.', color="g")

    i = i + 1
    plt.plot(i, wave[i], marker='.', color="r")

    for i in range(start, input_end):
        plt.plot(i, wave[i], marker='.', color="b")
    
    #plt.legend()
    plt.ylim([-1.5, 1.5])
    plt.xlim([-10, 110])
    #plt.grid(True)
    plt.show()

    
n_data = 100
sin_data = ds.create_wave(n_data, 0.0)


plot_fig3(sin_data, 0, 25)
plot_fig4(sin_data, 0, 25)

plot_fig3(sin_data, 1, 26)
plot_fig4(sin_data, 0, 26)

plot_fig3(sin_data, 2, 27)
plot_fig4(sin_data, 0, 27)



plot_fig3(sin_data, 74, 99)
plot_fig4(sin_data, 0, 99)
