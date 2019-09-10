import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")
sns.set_context("paper",font_scale=2.0)
sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

def loglogplot(figsize=(7,5.8)):
    plt.rc('text', usetex=True)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.loglog()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='minor', colors='black', width=1.0, length=3.0)
    return fig, ax

def plot(figsize=(7,5.8),loglog=False,semilogx=False,semilogy=False): #figure out what the standard figure size is for journal articles...
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.rc('text', usetex=True)

    if(loglog == True):
        ax.loglog()
    if(semilogx == True):
        ax.semilogx()
    if(semilogy == True):
        ax.semilogy()
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='both', which='minor', colors='black', width=1.0, length=3.0)
    return fig, ax

