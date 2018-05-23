from modAL.models import ActiveLearner
import pdb
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import max_std_sampling
from sklearn.svm import SVR
from xgboost import XGBRegressor

data = pd.read_csv(sys.argv[1])
X = data.loc[:, data.columns[1:-3]].values
y = data['HeartRate'].values
time = data['timestamp'].values


rf= ActiveLearner(estimator=RandomForestRegressor(n_estimators=150, n_jobs=-1), )
svrl = ActiveLearner(estimator=SVR(kernel='rbf'))
ada = ActiveLearner(estimator=AdaBoostRegressor(n_estimators=250))
gbt = ActiveLearner(estimator=GradientBoostingRegressor(n_estimators=500, criterion='mae'))
xgb1 = ActiveLearner(estimator=XGBRegressor())


learner_list = [rf, svrl, ada, gbt, xgb1 ]

committee = CommitteeRegressor(
    learner_list=learner_list,
    query_strategy=max_std_sampling
)


from sklearn.metrics import mean_absolute_error
print(sys.argv[1])
xt, yt = X[:100], y[:100]
print(xt)
print(yt)
print(xt.shape)
print(yt.shape)

committee.teach(xt, yt)
idx = 0
stds = []
varl = []
modup = 0
diffs = []
sqdiffs =[ ]
refits =0
llidx = 0
recentX=[]
recentY=[]
learners_time = []
# the history to keep..
most_recent_n = int(sys.argv[4])
# set all learner alive time to 0
for i in range(len(learner_list)):
    learners_time.append(0)

# not iterate over the remainder of the data
for i in range(len(X[100:])):
    _, std, votes, var = committee.predict(X[idx].reshape(1,-1), return_std=True,)
    stds.append(std)
    varl.append(var)
    print("----")
    print("pred "+str(_))
    print("true "+str(y[idx]))
    print("std: "+str(std))
    print("var: "+str(var))
    # keep the absolute differences between the prediction and true hr
    diffs.append(abs(_-y[idx]))
    # keep the squared differences between the prediction and true hr
    sqdiffs.append(abs(_-y[idx])**2)
    diff = (abs(_-y[idx]))
    # reshape the current feature vector..
    xt = X[idx].reshape(1,-1)
    yt = y[idx].reshape(-1, )
    print(learners_time)
    # all leaeners have made a prediction.. so increase alive time
    for lt in range(len(learner_list)):
        learners_time[lt] += 1
    # we only keep the most revent history of prediction certainities
    if len(stds) > most_recent_n:
        stds = stds[-most_recent_n:]
    if len(varl) > most_recent_n:
        varl = varl[-most_recent_n:]
    # if there is no variance, or its greater than some X times the standard deviation
    if var ==0 or var > (int(sys.argv[3])*np.std(varl)):
        #print("current std of stds is (10 his ) "+str(np.std(stds)))
        print("current std of var is (10 his ) "+str(np.std(varl)))
       # xt = X[idx].reshape(1,-1)
       # yt = y[idx].reshape(-1, )
        # since we assume we have queried the true HR - lets store it for future learning
        recentX.append(X[idx])
        recentY.append(y[idx])
        # again only keep only most recent history
        if len(recentX) > most_recent_n:
            recentX = recentX[-most_recent_n:]
            recentY = recentY[-most_recent_n:]
        # increase the number of modelupdates (or labels queried)
        modup += 1
        # if the difference between the true HR and predicted is greater than some threshold
        if diff > 5:
            torefreshid = -1
            for vid, v in enumerate(votes[0]):
                # update any models greater than the threshold..
                if abs(v - y[idx]) > 5:
                    torefreshid = vid
                    print("REFITTING learner "+str(torefreshid))
                    recentXt =np.reshape(recentX, (len(recentX), len(recentX[0])))
                    recentYt =np.reshape(recentY, (len(recentY),))
                    # retrain the model on the most recent data..
                    committee.learner_list[torefreshid].teach(recentXt, recentYt, only_new=True)
                    # set alive time to 0
                    learners_time[torefreshid] = 0
        else:
            # if we have uncertainity.. but the true difference isnt over a threshold then just update the models
            print("UPDATING MODEL")
            print(votes[0])
            committee.teach(xt, yt)
    # after making the prediction lets check if any models have been alive longer than allowed..
    for torefreshid, learner in enumerate(learners_time):
        if learner > int(sys.argv[5]):
            print("REFITTING learner TIME EXP "+str(torefreshid))
            recentXt =np.reshape(recentX, (len(recentX), len(recentX[0])))
            recentYt =np.reshape(recentY, (len(recentY),))
            committee.learner_list[torefreshid].teach(recentXt, recentYt, only_new=True)
            learners_time[torefreshid] = 0
    upds =str((float(modup) / (i+1))*100)+"% "+(str(modup)+"/"+str((i+1)))
    print("upds "+upds)
    mae = str(np.mean(diffs))
    print("mae "+mae)
    mse = str(np.mean(sqdiffs))
    print("mse "+mse)

    print("----")
    idx+=1
    with open(sys.argv[2], 'a') as f:
        f.write(mae+","+str(modup)+","+str(i+1)+","+str(_)+","+str(y[idx])+","+str(mse)+","+str(time[idx])+"\n")
    continue


