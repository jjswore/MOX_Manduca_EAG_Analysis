import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import pickle
def name_con(f):
    n = os.path.basename(f)
    tn = n.split("_")
    if tn[1].lower() == '100':
        x = '1:100'
    elif tn[1].lower() == '1k':
        x = '1:1k'
    elif tn[1].lower() == '10k':
        x = '1:10k'
    else:
        x = ''
    name = x + ' ' + tn[2]
    return name

def Open_Pickle_Jar(filepath, perms='rb', array=True):
    file = open(filepath, perms)
    data = pickle.load(file)
    file.close()
    if array == True:
        data = np.asarray(data, dtype=object)  # this turns the current list of lists into a nested array (easier to index)
    return data



def pickle_to_DF(FILE):
    f=Open_Pickle_Jar(FILE)
    dd = defaultdict(list)

    for d in (f[0]): # you can list as many input dicts as you want here
        for key, value in d.items():
            dd[key].append(value)
    return pd.DataFrame.from_dict(dd)

def ClassifierResults(BASE, BUTTER, PROCESS,CH, QCTHRESH,Feature, MODEL):

    df=pickle_to_DF(f'{BASE}Butter{BUTTER}/{PROCESS}/{CH}/ClassifierResults/{Feature}/_QC_T_{QCTHRESH}/{MODEL}_Results.pickle')
    return df

def rearrange_for_ctrl(CM, labels):
    CM = CM.copy()

    ctrl_index = labels.index('CTRL')
    middle_index = len(labels) // 2

    if ctrl_index != middle_index:
        # Swap 'CTRL' label position with middle label
        labels[ctrl_index], labels[middle_index] = labels[middle_index], labels[ctrl_index]

        # Swap corresponding rows and columns in the confusion matrix
        CM[[ctrl_index, middle_index], :] = CM[[middle_index, ctrl_index], :]
        CM[:, [ctrl_index, middle_index]] = CM[:, [middle_index, ctrl_index]]

    return CM, labels
def extract_CM(DF, cumulative=False, ACC=True):
    if cumulative == True:
        sumCM=DF['confusion_matrix'].sum()
        return sumCM
    if ACC == True:
        cumulative_CM = DF['confusion_matrix'].sum()
        normalized_CM = cumulative_CM / cumulative_CM.sum(axis=1, keepdims=True)

        return normalized_CM
#
def plot_CM(CM,LABELS,TITLE, YROT=0, XROT=90, REARRANGE = False, SAVEDIR=None):
    if REARRANGE == True:
        CM, LABELS = rearrange_for_ctrl(CM, LABELS)

    fig, ax = plt.subplots(figsize=(10, 10))

    disp=ConfusionMatrixDisplay(CM,display_labels=LABELS)
    CMDISP=disp.plot(cmap=plt.cm.Reds,ax=ax)

    for labels in disp.text_:
        for label in labels:
            label.set_fontsize(30)

    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=25, weight='bold', rotation=XROT)
    ax.set_yticklabels(LABELS, fontsize=25, weight='bold', rotation=YROT)
    plt.ylabel('True', fontsize=30, weight='bold')
    plt.xlabel('Predicted', fontsize=30, weight='bold')

    cbar = ax.images[-1].colorbar
    cbar.mappable.set_clim(0, 1)

    for label in ax.get_yticklabels():
        label.set_va('center')

    plt.subplots_adjust(left=.1, right=1.05, top=.99, bottom=.01)

    if SAVEDIR is not None:

        print('Saving Fig...')
        plt.savefig(os.path.join(f'{SAVEDIR}Confusion_Matrix.jpg'))
        plt.savefig(os.path.join(f'{SAVEDIR}Confusion_Matrix.svg'))

    else:
        print('Figure is not saved')
    plt.show()
    return CMDISP

def ViPlot(DATA, TITLE, N_Odors, INNER='box', DisplayMean=True, SAVEDIR=None):
    plt.figure(figsize=(10, 7))
    plt.xticks(None)
    plt.gca().set(xticklabels=[])
    plt.yticks(fontsize=24, weight='bold')
    #plt.xlabel('Dataset', fontsize=20, weight='bold')
    plt.ylabel('Accuracy', fontsize=24, weight='bold')
    plt.title(TITLE, fontsize=20)
    plt.ylim(0, 1)
    plt.axhline(y=(1 / N_Odors), linestyle='--', color='black')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Create the violin plot
    ax = sns.violinplot(data=DATA, color="salmon", alpha = .5, linewidth=5, inner=INNER)
    if INNER != 'box':
        sns.swarmplot(data=DATA, color="steelblue", size=8)


    # Calculate the mean for each column
    if DisplayMean == True:
        means = DATA.mean()

        # Add annotations to the plot
        for i, mean in enumerate(means):
            ax.text(i, mean, f'Mean: {mean:.2f}', ha='center', va='top', fontsize=20)

    plt.ylim(0, 1.1)
    plt.tight_layout()
    if SAVEDIR is not None:
        print('Saving Fig...')
        plt.savefig(f'{SAVEDIR}ViPlot.jpg')
        plt.savefig(f'{SAVEDIR}ViPlot.svg')
    plt.show()


def BoxPlot(DATA, TITLE, N_Odors, SAVEDIR=None):
    plt.figure(figsize=(18, 9))
    plt.xticks(rotation=0, fontsize=12, weight='bold')
    plt.yticks(fontsize=14, weight='bold')
    #plt.xlabel('Dataset', fontsize=20, weight='bold')
    plt.ylabel('Mean Accuracy', fontsize=20, weight='bold')
    plt.title(TITLE, fontsize=20)
    plt.ylim(0, 1)
    plt.axhline(y=(1 / N_Odors), linestyle='--', color='black')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Create the violin plot
    #ax = sns.boxplot(data=DATA, color="skyblue")

    sns.boxplot(data=DATA, color=".9")#, inner=None)
    #sns.swarmplot(data=DATA, size=3)
    # Calculate the mean for each column
    means = DATA.mean()

    # Add annotations to the plot
    #for i, mean in enumerate(means):
        #ax.text(i, mean, f'Mean: {mean:.2f}', ha='center', va='top', fontsize=20)

    plt.ylim(0, 1.1)
    plt.tight_layout()
    if SAVEDIR is not None:
        print('Saving Fig...')
        plt.savefig(f'{SAVEDIR}ViPlot.jpg')
        plt.savefig(f'{SAVEDIR}ViPlot.svg')
    plt.show()
