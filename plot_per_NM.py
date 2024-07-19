"""
18July2024
KB loop through the good sessions 
GOOD SESSIONS in excel MICE AND SESSIONS 

Goal: make plots for more than 1 session
        for example, joining all the DA sessions 
""" 
#%%
#imports 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from ibldsp.utils import parabolic_max 
from iblphotometry.preprocessing import jove2019, psth, preprocess_sliding_mad, photobleaching_lowpass 
from one.api import ONE #always after the imports 
one = ONE(cache_dir="/mnt/h0/kb/data/one") 

#functions 
def get_eid(mouse,date): 
    eids = one.search(subject=mouse, date=date) 
    eid = eids[0]
    ref = one.eid2ref(eid)
    print(eid)
    print(ref) 
    try:
        # Try to load the trials directly
        a = one.load_object(eid, 'trials')
        trials = a.to_df()
    except Exception as e:
        # If loading fails, use the alternative method
        print("Failed to load trials directly. Using alternative method...")
        session_path_behav = f'/home/kceniabougrova/Documents/nph/Behav_2024Mar20/{rec.mouse}/{rec.date}/001/'
        df_alldata = extract_all(session_path_behav)
        table_data = df_alldata[0]['table']
        trials = pd.DataFrame(table_data) 
    return eid, trials 

test_02 = pd.read_csv('/home/ibladmin/Downloads/Mice_GOOD_sorted.csv') 
test_02['Date'] = pd.to_datetime(test_02['Date'], format='%m/%d/%Y')
test_03 = test_02[['Mouse', 'Date', 'NM', 'region']] 
EVENT = 'stimOnTrigger_times'

# Initialize empty containers
psth_combined = None
df_trials_combined = pd.DataFrame()

EXCLUDES = []  
IMIN = 0

test_04 = test_03[test_03["NM"]=="DA"].reset_index(drop=True)
# test_04 = test_03[test_03["NM"]=="5HT"].reset_index(drop=True) 
# test_04 = test_03[test_03["NM"]=="NE"].reset_index(drop=True)
# test_04 = test_03[test_03["NM"]=="ACh"].reset_index(drop=True) 

####################################
#for DA: 
EXCLUDES = [5,6,8,12]  
IMIN = 0


#%%
for i in range(len(test_04)): 
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue
    mouse = test_04.Mouse[i] 
    date = test_04.Date[i]
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')
    region = test_04.region[i]
    eid, df_trials = get_eid(mouse,date)
    print(f"{mouse} | {date} | {region} | {eid}")
    df_trials['trialNumber'] = range(1, len(df_trials) + 1) 
    df_trials["mouse"] = mouse
    df_trials["date"] = date
    df_trials["region"] = region
    df_trials["eid"] = eid 

    path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/' 
    path = path_initial + f'preprocess_calcium_jove2019_{EVENT}_{mouse}_{date}_Region{region}G_{eid}.npy'

    # Load psth_idx from file
    psth_idx = np.load(path)

    # Concatenate psth_idx arrays
    if psth_combined is None:
        psth_combined = psth_idx
    else: 
        psth_combined = np.hstack((psth_combined, psth_idx))

    # create allContrasts 
    idx = 2
    new_col = df_trials['contrastLeft'].fillna(df_trials['contrastRight']) 
    df_trials.insert(loc=idx, column='allContrasts', value=new_col) 
    # create allSContrasts 
    df_trials['allSContrasts'] = df_trials['allContrasts']
    df_trials.loc[df_trials['contrastRight'].isna(), 'allSContrasts'] = df_trials['allContrasts'] * -1
    df_trials.insert(loc=3, column='allSContrasts', value=df_trials.pop('allSContrasts'))
    # create reactionTime
    reactionTime = np.array((df_trials["firstMovement_times"])-(df_trials["stimOnTrigger_times"]))
    df_trials["reactionTime"] = reactionTime 

    # Concatenate df_trials DataFrames
    df_trials_combined = pd.concat([df_trials_combined, df_trials], axis=0)

    # Reset index of the combined DataFrame
    df_trials_combined.reset_index(drop=True, inplace=True)

    # Print shapes to verify
    print("Shape of psth_combined:", psth_combined.shape)
    print("Shape of df_trials_combined:", df_trials_combined.shape)
#%%
    ##################################################################################################
    # PLOT 
    psth_good = psth_combined[:,(df_trials_combined.feedbackType == 1)]
    psth_error = psth_combined[:,(df_trials_combined.feedbackType == -1)]
    # Calculate averages and SEM
    psth_good_avg = psth_good.mean(axis=1)
    sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
    psth_error_avg = psth_error.mean(axis=1)
    sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

    # Create the figure and gridspec
    fig = plt.figure(figsize=(10, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

    # Plot the heatmap and line plot for correct trials
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
    ax1.invert_yaxis()
    ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax1.set_title('Correct Trials')

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
    # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
    ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
    ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax2.set_ylabel('Average Value')
    ax2.set_xlabel('Time')

    # Plot the heatmap and line plot for incorrect trials
    ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
    sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
    ax3.invert_yaxis()
    ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax3.set_title('Incorrect Trials')

    ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
    ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
    ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
    ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax4.set_ylabel('Average Value')
    ax4.set_xlabel('Time')

    fig.suptitle(f'jove2019_{EVENT}_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
    plt.tight_layout()
    plt.show()
    ##################################################################################################


#%%
""" SAVE THE PSTH AND DF_TRIALS """
#save the psth npy and the df_trials pqt 
# path_initial = '/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_feedback_times_etc/' 
np.save(path_initial+f'RESULTS/jove2019_psth_combined_NE_{EVENT}.npy', psth_combined)
df_trials_combined.to_parquet(path_initial+f'RESULTS/jove2019_df_trials_combined_NE_{EVENT}.pqt')

#%%
#import the saved files 
what_to_load = 'combined_NE'
test_psth = np.load(path_initial+f'RESULTS/jove2019_psth_{what_to_load}.npy')
test_trials = pd.read_parquet(path_initial+f'RESULTS/jove2019_df_trials_{what_to_load}.pqt')

# %%
##################################################################################################
""" MAKE DIFFERENT PLOTS """
""" 0. functions """

def avg_sem(psth_array): 
    psth_avg = psth_array.mean(axis=1)
    psth_sem = psth_array.std(axis=1) / np.sqrt(psth_array.shape[1])
    return psth_avg, psth_sem

def filter_data(psth_array=psth_combined, df_trials=df_trials_combined, event1=None, event2=None, event3=None, 
                mice=None): 
    if mouse!=None: 
        mask = df_trials["mouse"].isin(mice)
        test = psth_array[:, mask]
    return test 
    # test = filter_data(mice=["ZFM-04019","ZFM-04022"])

    if event1=="feedbackType": 
        psth_correct = psth_array[:,(df_trials[event1] == 1)]
        psth_incorrect = psth_array[:,(df_trials[event1] == -1)] 
    
    if event2=="allContrasts": 
        psth_100 = psth_array[:,(df_trials[event2] == 1)]
        psth_25 = psth_array[:,(df_trials[event2] == 0.25)]
        psth_12 = psth_array[:,(df_trials[event2] == 0.125)]
        psth_06 = psth_array[:,(df_trials[event2] == 0.0625)]
        psth_0 = psth_array[:,(df_trials[event2] == 0)] 
    
    if event3=="probabilityLeft": 
        psth_50 = psth_array[:,(df_trials[event3] == 0.5)]
        psth_20 = psth_array[:,(df_trials[event3] == 0.2)]
        psth_80 = psth_array[:,(df_trials[event3] == 0.8)] 
    
    if event1=="feedbackType" and event2=="allContrasts": 

    
    return psth_correct, psth_incorrect








""" 1. feedbackType """
psth_good = psth_combined[:,(df_trials_combined.feedbackType == 1)] 
psth_error = psth_combined[:,(df_trials_combined.feedbackType == -1)]
# Calculate averages and SEM
psth_good_avg = psth_good.mean(axis=1)
sem_good = psth_good.std(axis=1) / np.sqrt(psth_good.shape[1])
psth_error_avg = psth_error.mean(axis=1)
sem_error = psth_error.std(axis=1) / np.sqrt(psth_error.shape[1])

# Create the figure and gridspec
fig = plt.figure(figsize=(10, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

# Plot the heatmap and line plot for correct trials
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(psth_good.T, cbar=False, ax=ax1) #, center = 0.0)
ax1.invert_yaxis()
ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
ax1.set_title('Correct Trials')

ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
# ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
ax2.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax2.set_ylabel('Average Value')
ax2.set_xlabel('Time')

# Plot the heatmap and line plot for incorrect trials
ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
sns.heatmap(psth_error.T, cbar=False, ax=ax3) #, center = 0.0)
ax3.invert_yaxis()
ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
ax3.set_title('Incorrect Trials')

ax4 = fig.add_subplot(gs[1, 1], sharex=ax3, sharey=ax2)
ax4.plot(psth_error_avg, color='#d62828', linewidth=3)
ax4.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax4.set_ylabel('Average Value')
ax4.set_xlabel('Time')

fig.suptitle(f'calcium_mad_{EVENT}_{mouse}_{date}_{region}_{eid}', y=1, fontsize=14)
plt.tight_layout()
plt.show()

#%%
""" 2. allContrasts """

psth_array=psth_combined
df_trials=df_trials_combined 
event2="allContrasts"
psth_100 = psth_array[:,(df_trials[event2] == 1)]
avg100, sem100 = avg_sem(psth_100)
psth_25 = psth_array[:, (df_trials[event2] == 0.25)]
avg25, sem25 = avg_sem(psth_25)
psth_12 = psth_array[:, (df_trials[event2] == 0.125)]
avg12, sem12 = avg_sem(psth_12)
psth_06 = psth_array[:, (df_trials[event2] == 0.0625)]
avg06, sem06 = avg_sem(psth_06)
psth_0 = psth_array[:, (df_trials[event2] == 0)]
avg0, sem0 = avg_sem(psth_0)

plt.plot(avg100, color="black", label="100")
plt.plot(avg25, color="black", alpha=0.6, label="25")
plt.plot(avg12, color="black", alpha=0.4, label="12")
plt.plot(avg06, color="black", alpha=0.2, label="6")
plt.plot(avg0, color="black", alpha=0.05, label="0") 
plt.axvline(x=30, linestyle='dashed', color='black')
plt.legend()
plt.title(f'All contrasts, aligned to {EVENT}') 
plt.show()
##################################################################################################
# %%
