import numpy as np

###
### Config, labelling
###
models = ['kernel', 'pbm'];
co = 6;#5: overall step time, #6: kernel time, #7: rebuild/texture time
###
### Load Data
###
#d1, s1, m1, h1, agentCount, neighbours = np.loadtxt(
csv = np.loadtxt(
    '2D Param Sweep.csv',
     dtype=[('radialNeighboursEst','int'), ('agentCount','int'), ('envWidth','int'), ('pbm_control','float'), ('kernel_control','float'), ('pbm_smartfov','float'), ('kernel_smartfov','float'), ('failures','int')],
     skiprows=3,
     delimiter=',',
     usecols=(0,1,2,3,4,5,6,7,8),
     unpack=True
 );
failures = csv.pop(-1)
#kernel_smartfov, pbm_smartfov
results_smartfov = [csv.pop(-1), csv.pop(-1)];
#kernel_control, pbm_control
results_control = [csv.pop(-1), csv.pop(-1)];
envWidth = csv.pop(-1);
agentCount = csv.pop(-1);
neighbours = csv.pop(-1);
###
### Preprocessing
###
#Add best point from each to new arrays
minI = [-1] *(len(results_smartfov));
maxI = [-1] *(len(results_smartfov));
minV = [float("inf")] *(len(results_smartfov));
maxV = [-float("inf")] *(len(results_smartfov));
#For each param config tested, find the model with best result
for i in range(len(agentCount)):
    #Find which is lowest at this index
    for j in range(len(results_smartfov)):
        #calc diff
        #diff = csv[0][i]-csv[j+1][i];
        diff = results_control[j][i]/results_smartfov[j][i];
        if diff<minV[j]:
            minV[j] = diff;
            minI[j] = i;
        if diff>maxV[j]:
            maxV[j] = diff;
            maxI[j] = i;
#Output results to console
#Log it as the winner for the given params
for x in range(len(results_smartfov)):
    if(minI[x]>=0 or maxI[x]>=0):
        print("%s:" % models[x])
        if(minI[x]>=0):
            print("Worst improvement @ agentCount(%d), neighbourAvg(%.1f): Control(%.2fms), Smart FoV(%.2fms) [%.1f%%]" % (agentCount[minI[x]],neighbours[minI[x]],results_control[x][minI[x]],results_smartfov[x][minI[x]],minV[x]*100));
        if(maxI[x]>=0):
            print("Best improvement @ agentCount(%d), neighbourAvg(%.1f): Control(%.2fms), Smart FoV(%.2fms) [%.1f%%]" % (agentCount[maxI[x]],neighbours[maxI[x]],results_control[x][maxI[x]],results_smartfov[x][maxI[x]],maxV[x]*100));