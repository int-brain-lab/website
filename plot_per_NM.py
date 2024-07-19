"""
18July2024
KB loop through the good sessions 
GOOD SESSIONS in excel MICE AND SESSIONS 

""" 
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
EVENT = 'feedback_times'

EXCLUDES = []  

IMIN = 0

# Initialize empty containers
psth_combined = None
df_trials_combined = pd.DataFrame()

# test_04 = test_03[test_03["NM"]=="DA"].reset_index(drop=True)
# test_04 = test_03[test_03["NM"]=="5HT"].reset_index(drop=True) 
# test_04 = test_03[test_03["NM"]=="NE"].reset_index(drop=True)
test_04 = test_03[test_03["NM"]=="ACh"].reset_index(drop=True)

for i in range(len(test_04)): 
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue
    EVENT = "feedback_times"
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

    path_initial = '/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_feedback_times_etc/' 
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
    psth_good = psth_idx[:,(df_trials.feedbackType == 1)]
    psth_error = psth_idx[:,(df_trials.feedbackType == -1)]
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
    ##################################################################################################


#%%
np.save('/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_feedback_times_etc/RESULTS/jove2019_psth_combined_ACh.npy', psth_combined)
df_trials_combined.to_parquet('/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_feedback_times_etc/RESULTS/jove2019_df_trials_combined_ACh.pqt')


