import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
  
def plotLine(name, xData, yData, color, symbol, line='-', doPoints=True):
    if len(xData)!=len(yData):
        print("Len x and y do not match: %d vs %d" %(len(xData), len(yData)));
    #Array of sampling vals for polyfit line
    xp = np.linspace(xData[0], xData[-1]*0.98, 100)
    #polyfit
    default_z = np.polyfit(xData, yData, 6)
    default_fit = np.poly1d(default_z)
    # plt.plot(
        # xp, 
        # default_fit(xp), 
        # str(color)+str(line),
        # label=str(name),
        # lw=1
    # );
    #points
    if(doPoints):
        default_h = plt.plot(
           xData,yData, 
           str(color)+str(symbol),
           label=str(name),
           lw=1
        );
    
if len(sys.argv)<4 or int(sys.argv[1]) > 2 or int(sys.argv[2]) > 1:
    print("Usage: crossSection.py <y-axis> <x-axis> <x-filter>")
    print("\ty-axis: 0(overall step), 1(model kernel), 2(pbm)")
    print("\tfilter: 0(neighbourhood size), 1(agent count)")
    print("\tfilter value: Required value for opposite column to x-axis")
    sys.exit(0);

###
### Config, labelling
###
models = ['Smart Fov', 'Control'];
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k'];
symbols = ['*', 'o', '^', 'x', 's', '+', 'h','p'];
lines = ['-','--',':', '-.']
co = 6;#5: overall step time, #6: kernel time, #7: rebuild/texture time
csType = 7;#7 agentcount, 8 density
plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["legend.fontsize"] = 8
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('axes', labelsize=8)
fig = plt.figure()
fig.set_size_inches(3.5, 3.5/1.4)
co = int(sys.argv[1])+5;
csType = int(sys.argv[2])+7;
filterArg = int(sys.argv[3]);#Either a problem size or neighbourhood size
#Label axis
if csType==8:
    plt.xlabel('Population');
    plt.title('Neighbourhood Size: %d, Scaling Agent Count' % filterArg);
    print("Filtering Density");
elif csType==7:
    plt.xlabel('Est Radial Neighbourhood Size');
    plt.title('Agent Count: %d, Scaling Neighbourhood Size' % filterArg);
    print("Filtering AgentCount");
else:
    print('Unexpected csType (7-8 required)');
    sys.exit(0);
if co==5:
    plt.ylabel('Iteration Time (ms)');
elif co==6:
    plt.ylabel('Model Kernel Time (ms)');
elif co==7:
    plt.ylabel('PBM Rebuild Time (ms)');
else:
    print('Unexpected column (5-6 required)');
    sys.exit(0);
###
### Load Data
###
csv = np.loadtxt(
    '2D Param Sweep.csv',
     dtype=[('radialNeighboursEst','int'), ('agentCount','int'), ('envWidth','int'), ('pbm_control','float'), ('kernel_control','float'), ('pbm_smartfov','float'), ('kernel_smartfov','float'), ('failures','int')],
     skiprows=3,
     delimiter=',',
     usecols=(0,1,2,3,4,5,6,7,8),
     unpack=True
 );
failures = csv.pop(-1)
#kernel_smartfov, pbm_smartfov, kernel_control, pbm_control
results = [csv.pop(-1), csv.pop(-1), csv.pop(-1), csv.pop(-1)];
envWidth = csv.pop(-1);
agentCount = csv.pop(-1);
neighbours = csv.pop(-1);
###
### Preprocessing
###
#Add best point from each to new arrays
xVals = [];
yVals = [[] for x in range(int(len(results)/2))];
#For each param config tested, find the model with best result
for i in range(len(agentCount)):
    if csType==7:#Filter agent count
        if agentCount[i]==filterArg:
            xVals.append(neighbours[i]);
            for j in range(len(yVals)):
                if co==5: #Iteration Time
                    yVals[j].append(results[(j*2)][i]+results[(j*2)+1][i]);
                elif co==6: #Model Kernel Time (ms)
                    yVals[j].append(results[(j*2)][i]);
                elif co==7: #PBM Rebuild Time (ms)
                    yVals[j].append(results[(j*2)+1][i]);
    elif csType==8:#Filter neighbourAvg
        if neighbours[i]==filterArg:
            xVals.append(agentCount[i]);
            for j in range(len(yVals)):
                if co==5: #Iteration Time
                    yVals[j].append(results[(j*2)][i]+results[(j*2)+1][i]);
                elif co==6: #Model Kernel Time (ms)
                    yVals[j].append(results[(j*2)][i]);
                elif co==7: #PBM Rebuild Time (ms)
                    yVals[j].append(results[(j*2)+1][i]);
#Ensure we have data
if len(xVals)==0:
    print("Filter val %f was not found."%filterArg);
    sys.exit(0);      
#Line plot the data
for i in range(len(yVals)):
    #plotLine(models[i], xVals, yVals[i], 'k', symbols[i], lines[i], False);
    plotLine(models[i], xVals, yVals[i], colors[i], '-');

#select right corner for legend
#locPos = 1 if d1[0]>d1[-1] else 2;
#plt.legend(loc=locPos,numpoints=1);
#Show plot
if csType==7:
    plt.legend(loc='lower right',numpoints=1);
else :
    plt.legend(loc='upper left',numpoints=1);
plt.tight_layout();
plt.show();
