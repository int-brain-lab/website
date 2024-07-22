"""
18July2024
KB loop through the good sessions 
GOOD SESSIONS in excel MICE AND SESSIONS 

Goal: make plots for more than 1 session
        for example, joining all the DA sessions 

Important to have: 
    1. mouse, date, region values 
    2. eid to extract the behavior 
    3. preprocessed psth npy files - for each preprocessing, for each event, across sessions 

Important info: 
    1. nph data aligned to behav data: 
        example: 
            feedback #X happens at 5.5197
            nph has the following [index]=timestamp: 
                [465]=5.50195715, [466]=5.53526852 
            the code will pick 466
            or
            feedback #Y happens at 9.4941
                nph has [584]=9.46840962, [585]=9.50175299 
            the code will pick 585
        diff was 0.0156 and 0.0077 after the event occured 
        use x=30 for the vertical line! 
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

# Get the list of good sessions and their info 
df_goodsessions = pd.read_csv('/home/ibladmin/Downloads/Mice_GOOD_sorted.csv') 
df_goodsessions['Date'] = pd.to_datetime(df_goodsessions['Date'], format='%m/%d/%Y')
df_gs = df_goodsessions[['Mouse', 'Date', 'NM', 'region']] 

# Edit the event! 
EVENT = 'stimOnTrigger_times'

# Initialize empty containers
psth_combined = None
df_trials_combined = pd.DataFrame()

EXCLUDES = []  
IMIN = 0

# Choose the NM
NM="ACh" #"DA", "5HT", "NE", "ACh"
df_goodsessions = df_gs[df_gs["NM"]==NM].reset_index(drop=True)

####################################
#test_04 = test_04.drop(32).reset_index(drop=True)
#for DA: 
# EXCLUDES = [5,6,8,12]  
#for 5HT: 
# EXCLUDES = [32] 
#for NE: 
# EXCLUDES = [43]  
#for ACh: 
# EXCLUDES = []  
# IMIN = 0 


#%%
for i in range(len(df_goodsessions)): 
    print(i,df_goodsessions['Mouse'][i])
    if i < IMIN:
        continue
    if i in EXCLUDES:
        continue
    mouse = df_goodsessions.Mouse[i] 
    date = df_goodsessions.Date[i]
    if isinstance(date, pd.Timestamp):
        date = date.strftime('%Y-%m-%d')
    region = df_goodsessions.region[i]
    eid, df_trials = get_eid(mouse,date)
    print(f"{mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")
    print(f"i | {mouse} | {date} | {region} | {eid}")


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
    # PLOT heatmap and correct vs incorrect 
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

    fig.suptitle(f'jove2019_{EVENT}_{mouse}_{date}_{region}_{NM}_{eid}', y=1, fontsize=14)
    plt.tight_layout()
    plt.show()
    ##################################################################################################


#%%
""" SAVE THE PSTH AND DF_TRIALS """
#save the psth npy and the df_trials pqt 
# path_initial = '/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_feedback_times_etc/' 
np.save(path_initial+f'RESULTS/jove2019_psth_combined_{NM}_{EVENT}.npy', psth_combined)
df_trials_combined.to_parquet(path_initial+f'RESULTS/jove2019_df_trials_combined_{NM}_{EVENT}.pqt')








##################################################################################################
##################################################################################################
##################################################################################################
"""
KB 18JULY2024 
Load the psth and behav, combined
and plot them 
"""
#%%
# #import the saved files 
# NM = "DA"
# EVENT = "feedback_times"
# path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/' 
# what_to_load = f'combined_{NM}_{EVENT}'
# psth_combined = np.load(path_initial+f'RESULTS/jove2019_psth_{what_to_load}.npy')
# df_trials_combined = pd.read_parquet(path_initial+f'RESULTS/jove2019_df_trials_{what_to_load}.pqt')

#%%
""" IMPORTING ALL """ 
# all checked and corrected 22July2024
EVENT = "feedback_times"
NM = "DA" 
path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/' 
what_to_load = f'combined_{NM}_{EVENT}'
psth_combined_DA = np.load(path_initial+f'RESULTS/jove2019_psth_{what_to_load}.npy')
df_trials_combined_DA = pd.read_parquet(path_initial+f'RESULTS/jove2019_df_trials_{what_to_load}.pqt')
NM = "5HT"
what_to_load = f'combined_{NM}_{EVENT}'
psth_combined_5HT = np.load(path_initial+f'RESULTS/jove2019_psth_{what_to_load}.npy')
df_trials_combined_5HT = pd.read_parquet(path_initial+f'RESULTS/jove2019_df_trials_{what_to_load}.pqt')
NM = "NE"
what_to_load = f'combined_{NM}_{EVENT}'
psth_combined_NE = np.load(path_initial+f'RESULTS/jove2019_psth_{what_to_load}.npy')
df_trials_combined_NE = pd.read_parquet(path_initial+f'RESULTS/jove2019_df_trials_{what_to_load}.pqt')
NM = "ACh"
what_to_load = f'combined_{NM}_{EVENT}'
psth_combined_ACh = np.load(path_initial+f'RESULTS/jove2019_psth_{what_to_load}.npy')
df_trials_combined_ACh = pd.read_parquet(path_initial+f'RESULTS/jove2019_df_trials_{what_to_load}.pqt') 

#%%
EVENT = "stimOnTrigger_times"
NM = "DA"
path_initial = f'/mnt/h0/kb/data/psth_npy/preprocess_calcium_jove2019_{EVENT}_etc/'
what_to_load = f'combined_{NM}_{EVENT}'
psth_combined_DA_stim = np.load(path_initial+f'RESULTS/jove2019_psth_{what_to_load}.npy')
df_trials_combined_DA_stim = pd.read_parquet(path_initial+f'RESULTS/jove2019_df_trials_{what_to_load}.pqt')
NM = "5HT"
what_to_load = f'combined_{NM}_{EVENT}'
psth_combined_5HT_stim = np.load(path_initial+f'RESULTS/jove2019_psth_{what_to_load}.npy')
df_trials_combined_5HT_stim = pd.read_parquet(path_initial+f'RESULTS/jove2019_df_trials_{what_to_load}.pqt')
NM = "NE"
what_to_load = f'combined_{NM}_{EVENT}'
psth_combined_NE_stim = np.load(path_initial+f'RESULTS/jove2019_psth_{what_to_load}.npy')
df_trials_combined_NE_stim = pd.read_parquet(path_initial+f'RESULTS/jove2019_df_trials_{what_to_load}.pqt')
NM = "ACh"
what_to_load = f'combined_{NM}_{EVENT}'
psth_combined_ACh_stim = np.load(path_initial+f'RESULTS/jove2019_psth_{what_to_load}.npy')
df_trials_combined_ACh_stim = pd.read_parquet(path_initial+f'RESULTS/jove2019_df_trials_{what_to_load}.pqt')


# %%
##################################################################################################
""" MAKE DIFFERENT PLOTS """
""" 0. functions """

def avg_sem(psth_array): 
    psth_avg = psth_array.mean(axis=1)
    psth_sem = psth_array.std(axis=1) / np.sqrt(psth_array.shape[1])
    return psth_avg, psth_sem 

def corr_incorr_avg_sem(psth_combined,df_trials): 
    psth_correct = psth_combined[:, (df_trials.feedbackType == 1)]
    psth_incorrect = psth_combined[:, (df_trials.feedbackType == -1)]
    psth_correct_avg, sem_correct = avg_sem(psth_correct)
    psth_incorrect_avg, sem_incorrect = avg_sem(psth_incorrect) 
    return psth_correct, psth_incorrect, psth_correct_avg, sem_correct, psth_incorrect_avg, sem_incorrect 

##################################################################################################

#%%
# """ 1. feedbackType """
""" 1. Heatmap and lineplot for 4 NMs for corr incorrect - WORKS """ 
# DA
psth_correct_DA, psth_incorrect_DA, psth_correct_avg_DA, sem_correct_DA, psth_incorrect_avg_DA, sem_incorrect_DA = corr_incorr_avg_sem(psth_combined_DA, df_trials_combined_DA)
# 5HT
psth_correct_5HT, psth_incorrect_5HT, psth_correct_avg_5HT, sem_correct_5HT, psth_incorrect_avg_5HT, sem_incorrect_5HT = corr_incorr_avg_sem(psth_combined_5HT, df_trials_combined_5HT)
# NE
psth_correct_NE, psth_incorrect_NE, psth_correct_avg_NE, sem_correct_NE, psth_incorrect_avg_NE, sem_incorrect_NE = corr_incorr_avg_sem(psth_combined_NE, df_trials_combined_NE)
# ACh
psth_correct_ACh, psth_incorrect_ACh, psth_correct_avg_ACh, sem_correct_ACh, psth_incorrect_avg_ACh, sem_incorrect_ACh = corr_incorr_avg_sem(psth_combined_ACh, df_trials_combined_ACh)

# DA
psth_correct_DA_s, psth_incorrect_DA_s, psth_correct_avg_DA_s, sem_correct_DA_s, psth_incorrect_avg_DA_s, sem_incorrect_DA_s = corr_incorr_avg_sem(psth_combined_DA_stim, df_trials_combined_DA_stim)
# 5HT
psth_correct_5HT_s, psth_incorrect_5HT_s, psth_correct_avg_5HT_s, sem_correct_5HT_s, psth_incorrect_avg_5HT_s, sem_incorrect_5HT_s = corr_incorr_avg_sem(psth_combined_5HT_stim, df_trials_combined_5HT_stim)
# NE
psth_correct_NE_s, psth_incorrect_NE_s, psth_correct_avg_NE_s, sem_correct_NE_s, psth_incorrect_avg_NE_s, sem_incorrect_NE_s = corr_incorr_avg_sem(psth_combined_NE_stim, df_trials_combined_NE_stim)
# ACh
psth_correct_ACh_s, psth_incorrect_ACh_s, psth_correct_avg_ACh_s, sem_correct_ACh_s, psth_incorrect_avg_ACh_s, sem_incorrect_ACh_s = corr_incorr_avg_sem(psth_combined_ACh_stim, df_trials_combined_ACh_stim)


##################################################################################################

def plot_heatmap_lineplot(psth_good_s, psth_error_s, psth_good_avg_s, sem_good_s, psth_error_avg_s, sem_error_s, 
                          psth_good, psth_error, psth_good_avg, sem_good, psth_error_avg, sem_error, 
                          titleNM=None): 
    # Create the figure and gridspec
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1])

    # Plot the heatmap and line plot for correct trials stimOn
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(psth_good_s.T, cbar=False, ax=ax1) #, center = 0.0)
    ax1.invert_yaxis()
    ax1.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax1.set_title('Correct Trials')

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.plot(psth_good_avg_s, color='#2f9c95', linewidth=3) 
    # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
    ax2.fill_between(range(len(psth_good_avg_s)), psth_good_avg_s - sem_good_s, psth_good_avg_s + sem_good_s, color='#2f9c95', alpha=0.15)
    ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax2.set_ylabel('Average Value')
    ax2.set_xlabel('Time')

    # Plot the heatmap and line plot for incorrect trials
    ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
    sns.heatmap(psth_error_s.T, cbar=False, ax=ax3) #, center = 0.0)
    ax3.invert_yaxis()
    ax3.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax3.set_title('Incorrect Trials')

    ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax2) 
    ax4.plot(psth_error_avg_s, color='#d62828', linewidth=3)
    ax4.fill_between(range(len(psth_error_avg_s)), psth_error_avg_s - sem_error_s, psth_error_avg_s + sem_error_s, color='#d62828', alpha=0.15)
    ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax4.set_ylabel('Average Value')
    ax4.set_xlabel('Time')


    # Plot the heatmap and line plot for correct trials feedback outcome
    ax5 = fig.add_subplot(gs[0, 2])
    sns.heatmap(psth_good.T, cbar=False, ax=ax5) #, center = 0.0)
    ax5.invert_yaxis()
    ax5.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax5.set_title('Correct Trials')

    ax6 = fig.add_subplot(gs[1, 2], sharex=ax1, sharey=ax2)
    ax6.plot(psth_good_avg, color='#2f9c95', linewidth=3) 
    # ax2.plot(psth_good, color='#2f9c95', linewidth=0.1, alpha=0.2)
    ax6.fill_between(range(len(psth_good_avg)), psth_good_avg - sem_good, psth_good_avg + sem_good, color='#2f9c95', alpha=0.15)
    ax6.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax6.set_ylabel('Average Value')
    ax6.set_xlabel('Time')

    # Plot the heatmap and line plot for incorrect trials
    ax7 = fig.add_subplot(gs[0, 3], sharex=ax1)
    sns.heatmap(psth_error.T, cbar=False, ax=ax7) #, center = 0.0)
    ax7.invert_yaxis()
    ax7.axvline(x=30, color="white", alpha=0.9, linewidth=3, linestyle="dashed") 
    ax7.set_title('Incorrect Trials')

    ax8 = fig.add_subplot(gs[1, 3], sharex=ax1, sharey=ax2)
    ax8.plot(psth_error_avg, color='#d62828', linewidth=3)
    ax8.fill_between(range(len(psth_error_avg)), psth_error_avg - sem_error, psth_error_avg + sem_error, color='#d62828', alpha=0.15)
    ax8.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
    ax8.set_ylabel('Average Value')
    ax8.set_xlabel('Time') 

    plt.suptitle("Correct and incorrect for stimOn and Feedback for: " + titleNM)

    plt.tight_layout()
    plt.show() 

# Plotting for Dopamine (DA)
plot_heatmap_lineplot(psth_correct_DA_s, psth_incorrect_DA_s, psth_correct_avg_DA_s, sem_correct_DA_s, psth_incorrect_avg_DA_s, sem_incorrect_DA_s, 
                      psth_correct_DA, psth_incorrect_DA, psth_correct_avg_DA, sem_correct_DA, psth_incorrect_avg_DA, sem_incorrect_DA, 
                      titleNM="DA")

# Plotting for Serotonin (5HT)
plot_heatmap_lineplot(psth_correct_5HT_s, psth_incorrect_5HT_s, psth_correct_avg_5HT_s, sem_correct_5HT_s, psth_incorrect_avg_5HT_s, sem_incorrect_5HT_s, 
                      psth_correct_5HT, psth_incorrect_5HT, psth_correct_avg_5HT, sem_correct_5HT, psth_incorrect_avg_5HT, sem_incorrect_5HT, 
                      titleNM="5HT")

# Plotting for Norepinephrine (NE)
plot_heatmap_lineplot(psth_correct_NE_s, psth_incorrect_NE_s, psth_correct_avg_NE_s, sem_correct_NE_s, psth_incorrect_avg_NE_s, sem_incorrect_NE_s, 
                      psth_correct_NE, psth_incorrect_NE, psth_correct_avg_NE, sem_correct_NE, psth_incorrect_avg_NE, sem_incorrect_NE, 
                      titleNM="NE")

# Plotting for Acetylcholine (ACh)
plot_heatmap_lineplot(psth_correct_ACh_s, psth_incorrect_ACh_s, psth_correct_avg_ACh_s, sem_correct_ACh_s, psth_incorrect_avg_ACh_s, sem_incorrect_ACh_s, 
                      psth_correct_ACh, psth_incorrect_ACh, psth_correct_avg_ACh, sem_correct_ACh, psth_incorrect_avg_ACh, sem_incorrect_ACh, 
                      titleNM="ACh")

##################################################################################################
#%%
""" 1.2.1 feedback_times correct inc 4NMs - WORKS """ 
# Create the figure and gridspec
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 3])

# Plot for DA
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(psth_correct_avg_DA, color='#2f9c95', linewidth=3, label='Correct')
ax1.fill_between(range(len(psth_correct_avg_DA)), psth_correct_avg_DA - sem_correct_DA, psth_correct_avg_DA + sem_correct_DA, color='#2f9c95', alpha=0.15)
ax1.plot(psth_incorrect_avg_DA, color='#d62828', linewidth=3, label='Incorrect')
ax1.fill_between(range(len(psth_incorrect_avg_DA)), psth_incorrect_avg_DA - sem_incorrect_DA, psth_incorrect_avg_DA + sem_incorrect_DA, color='#d62828', alpha=0.15)
ax1.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax1.set_ylabel('Average Value')
ax1.set_xlabel('Time')
ax1.set_title('DA')
ax1.legend()

# Plot for 5HT
ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)
ax2.plot(psth_correct_avg_5HT, color='#2f9c95', linewidth=3, label='Correct')
ax2.fill_between(range(len(psth_correct_avg_5HT)), psth_correct_avg_5HT - sem_correct_5HT, psth_correct_avg_5HT + sem_correct_5HT, color='#2f9c95', alpha=0.15)
ax2.plot(psth_incorrect_avg_5HT, color='#d62828', linewidth=3, label='Incorrect')
ax2.fill_between(range(len(psth_incorrect_avg_5HT)), psth_incorrect_avg_5HT - sem_incorrect_5HT, psth_incorrect_avg_5HT + sem_incorrect_5HT, color='#d62828', alpha=0.15)
ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax2.set_ylabel('Average Value')
ax2.set_xlabel('Time')
ax2.set_title('5HT')
ax2.legend()

# Plot for NE
ax3 = fig.add_subplot(gs[1, 0], sharex=ax1, sharey=ax1)
ax3.plot(psth_correct_avg_NE, color='#2f9c95', linewidth=3, label='Correct')
ax3.fill_between(range(len(psth_correct_avg_NE)), psth_correct_avg_NE - sem_correct_NE, psth_correct_avg_NE + sem_correct_NE, color='#2f9c95', alpha=0.15)
ax3.plot(psth_incorrect_avg_NE, color='#d62828', linewidth=3, label='Incorrect')
ax3.fill_between(range(len(psth_incorrect_avg_NE)), psth_incorrect_avg_NE - sem_incorrect_NE, psth_incorrect_avg_NE + sem_incorrect_NE, color='#d62828', alpha=0.15)
ax3.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax3.set_ylabel('Average Value')
ax3.set_xlabel('Time')
ax3.set_title('NE')
ax3.legend()

# Plot for ACh
ax4 = fig.add_subplot(gs[1, 1], sharex=ax1, sharey=ax1)
ax4.plot(psth_correct_avg_ACh, color='#2f9c95', linewidth=3, label='Correct')
ax4.fill_between(range(len(psth_correct_avg_ACh)), psth_correct_avg_ACh - sem_correct_ACh, psth_correct_avg_ACh + sem_correct_ACh, color='#2f9c95', alpha=0.15)
ax4.plot(psth_incorrect_avg_ACh, color='#d62828', linewidth=3, label='Incorrect')
ax4.fill_between(range(len(psth_incorrect_avg_ACh)), psth_incorrect_avg_ACh - sem_incorrect_ACh, psth_incorrect_avg_ACh + sem_incorrect_ACh, color='#d62828', alpha=0.15)
ax4.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax4.set_ylabel('Average Value')
ax4.set_xlabel('Time')
ax4.set_title('ACh')
ax4.legend()

fig.suptitle('Neuromodulator activity at Feedback Outcome', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


##################################################################################################
#%% 
""" 1.2.2 feedback_times correct vs inc 4NMs - WORKS """ 
# # Calculate average and SEM for DA
# Define colors for correct and incorrect trials
colors_correct = {
    'DA': '#d62828',  # Red
    '5HT': '#8e44ad',  # Purple
    'NE': '#3498db',  # Blue
    'ACh': '#3cb371'  # Green
}
colors_incorrect = {
    'DA': '#d62828',  # Red
    '5HT': '#8e44ad',  # Purple
    'NE': '#3498db',  # Blue
    'ACh': '#3cb371'  # Green
}

def get_trial_numbers(psth_combined,df_trials): 
    psth_correct = psth_combined[:, (df_trials.feedbackType == 1)]
    psth_incorrect = psth_combined[:, (df_trials.feedbackType == -1)]
    n_correct = str(psth_correct.shape[1])
    n_incorrect = str(psth_incorrect.shape[1])
    return n_correct, n_incorrect 
da_c, da_i = get_trial_numbers(psth_combined_DA, df_trials_combined_DA)
ht_c, ht_i = get_trial_numbers(psth_combined_5HT, df_trials_combined_5HT)
ne_c, ne_i = get_trial_numbers(psth_combined_NE, df_trials_combined_NE)
ac_c, ac_i = get_trial_numbers(psth_combined_ACh, df_trials_combined_ACh)


# Create the figure and gridspec for side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

# Plot correct trials
ax1.plot(psth_correct_avg_DA, color=colors_correct['DA'], linewidth=3, label='DA Correct trials: ' + da_c)
ax1.fill_between(range(len(psth_correct_avg_DA)), psth_correct_avg_DA - sem_correct_DA, psth_correct_avg_DA + sem_correct_DA, color=colors_correct['DA'], alpha=0.15)
ax1.plot(psth_correct_avg_5HT, color=colors_correct['5HT'], linewidth=3, label='5HT Correct trials: ' + ht_c)
ax1.fill_between(range(len(psth_correct_avg_5HT)), psth_correct_avg_5HT - sem_correct_5HT, psth_correct_avg_5HT + sem_correct_5HT, color=colors_correct['5HT'], alpha=0.15)
ax1.plot(psth_correct_avg_NE, color=colors_correct['NE'], linewidth=3, label='NE Correct trials: ' + ne_c)
ax1.fill_between(range(len(psth_correct_avg_NE)), psth_correct_avg_NE - sem_correct_NE, psth_correct_avg_NE + sem_correct_NE, color=colors_correct['NE'], alpha=0.15)
ax1.plot(psth_correct_avg_ACh, color=colors_correct['ACh'], linewidth=3, label='ACh Correct trials: ' + ac_c)
ax1.fill_between(range(len(psth_correct_avg_ACh)), psth_correct_avg_ACh - sem_correct_ACh, psth_correct_avg_ACh + sem_correct_ACh, color=colors_correct['ACh'], alpha=0.15)
ax1.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax1.set_ylabel('Average Value')
ax1.set_xlabel('Time')
ax1.set_title('Correct Trials')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.legend()

# Plot incorrect trials
ax2.plot(psth_incorrect_avg_DA, color=colors_incorrect['DA'], linewidth=3, linestyle='dashed', label='DA Incorrect trials: ' + da_i)
ax2.fill_between(range(len(psth_incorrect_avg_DA)), psth_incorrect_avg_DA - sem_incorrect_DA, psth_incorrect_avg_DA + sem_incorrect_DA, color=colors_incorrect['DA'], alpha=0.15)
ax2.plot(psth_incorrect_avg_5HT, color=colors_incorrect['5HT'], linewidth=3, linestyle='dashed', label='5HT Incorrect trials: ' + ht_i)
ax2.fill_between(range(len(psth_incorrect_avg_5HT)), psth_incorrect_avg_5HT - sem_incorrect_5HT, psth_incorrect_avg_5HT + sem_incorrect_5HT, color=colors_incorrect['5HT'], alpha=0.15)
ax2.plot(psth_incorrect_avg_NE, color=colors_incorrect['NE'], linewidth=3, linestyle='dashed', label='NE Incorrect trials: ' + ne_i)
ax2.fill_between(range(len(psth_incorrect_avg_NE)), psth_incorrect_avg_NE - sem_incorrect_NE, psth_incorrect_avg_NE + sem_incorrect_NE, color=colors_incorrect['NE'], alpha=0.15)
ax2.plot(psth_incorrect_avg_ACh, color=colors_incorrect['ACh'], linewidth=3, linestyle='dashed', label='ACh Incorrect trials: ' + ac_i)
ax2.fill_between(range(len(psth_incorrect_avg_ACh)), psth_incorrect_avg_ACh - sem_incorrect_ACh, psth_incorrect_avg_ACh + sem_incorrect_ACh, color=colors_incorrect['ACh'], alpha=0.15)
ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax2.set_ylabel('Average Value')
ax2.set_xlabel('Time')
ax2.set_title('Incorrect Trials')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.legend()
fig.suptitle('Neuromodulator activity at Feedback Outcome', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()



##################################################################################################
#%%

def get_trial_numbers(psth_combined, df_trials):
    psth_correct = psth_combined[:, (df_trials.feedbackType == 1)]
    psth_incorrect = psth_combined[:, (df_trials.feedbackType == -1)]
    n_correct = str(psth_correct.shape[1])
    n_incorrect = str(psth_incorrect.shape[1])
    return n_correct, n_incorrect

da_c, da_i = get_trial_numbers(psth_combined_DA_stim, df_trials_combined_DA_stim)
ht_c, ht_i = get_trial_numbers(psth_combined_5HT_stim, df_trials_combined_5HT_stim)
ne_c, ne_i = get_trial_numbers(psth_combined_NE_stim, df_trials_combined_NE_stim)
ac_c, ac_i = get_trial_numbers(psth_combined_ACh_stim, df_trials_combined_ACh_stim)


# Create the figure and gridspec for side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharex=True, sharey=True)

# Plot correct trials
ax1.plot(psth_correct_avg_DA_s, color=colors_correct['DA'], linewidth=3, label='DA Correct trials: ' + da_c)
ax1.fill_between(range(len(psth_correct_avg_DA_s)), psth_correct_avg_DA_s - sem_correct_DA_s, psth_correct_avg_DA_s + sem_correct_DA_s, color=colors_correct['DA'], alpha=0.15)
ax1.plot(psth_correct_avg_5HT_s, color=colors_correct['5HT'], linewidth=3, label='5HT Correct trials: ' + ht_c)
ax1.fill_between(range(len(psth_correct_avg_5HT_s)), psth_correct_avg_5HT_s - sem_correct_5HT_s, psth_correct_avg_5HT_s + sem_correct_5HT_s, color=colors_correct['5HT'], alpha=0.15)
ax1.plot(psth_correct_avg_NE_s, color=colors_correct['NE'], linewidth=3, label='NE Correct trials: ' + ne_c)
ax1.fill_between(range(len(psth_correct_avg_NE_s)), psth_correct_avg_NE_s - sem_correct_NE_s, psth_correct_avg_NE_s + sem_correct_NE_s, color=colors_correct['NE'], alpha=0.15)
ax1.plot(psth_correct_avg_ACh_s, color=colors_correct['ACh'], linewidth=3, label='ACh Correct trials: ' + ac_c)
ax1.fill_between(range(len(psth_correct_avg_ACh_s)), psth_correct_avg_ACh_s - sem_correct_ACh_s, psth_correct_avg_ACh_s + sem_correct_ACh_s, color=colors_correct['ACh'], alpha=0.15)
ax1.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax1.set_ylabel('Average Value')
ax1.set_xlabel('Time')
ax1.set_title('Correct Trials')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.legend()

# Plot incorrect trials
ax2.plot(psth_incorrect_avg_DA_s, color=colors_incorrect['DA'], linewidth=3, linestyle='dashed', label='DA Incorrect trials: ' + da_i)
ax2.fill_between(range(len(psth_incorrect_avg_DA_s)), psth_incorrect_avg_DA_s - sem_incorrect_DA_s, psth_incorrect_avg_DA_s + sem_incorrect_DA_s, color=colors_incorrect['DA'], alpha=0.15)
ax2.plot(psth_incorrect_avg_5HT_s, color=colors_incorrect['5HT'], linewidth=3, linestyle='dashed', label='5HT Incorrect trials: ' + ht_i)
ax2.fill_between(range(len(psth_incorrect_avg_5HT_s)), psth_incorrect_avg_5HT_s - sem_incorrect_5HT_s, psth_incorrect_avg_5HT_s + sem_incorrect_5HT_s, color=colors_incorrect['5HT'], alpha=0.15)
ax2.plot(psth_incorrect_avg_NE_s, color=colors_incorrect['NE'], linewidth=3, linestyle='dashed', label='NE Incorrect trials: ' + ne_i)
ax2.fill_between(range(len(psth_incorrect_avg_NE_s)), psth_incorrect_avg_NE_s - sem_incorrect_NE_s, psth_incorrect_avg_NE_s + sem_incorrect_NE_s, color=colors_incorrect['NE'], alpha=0.15)
ax2.plot(psth_incorrect_avg_ACh_s, color=colors_incorrect['ACh'], linewidth=3, linestyle='dashed', label='ACh Incorrect trials: ' + ac_i)
ax2.fill_between(range(len(psth_incorrect_avg_ACh_s)), psth_incorrect_avg_ACh_s - sem_incorrect_ACh_s, psth_incorrect_avg_ACh_s + sem_incorrect_ACh_s, color=colors_incorrect['ACh'], alpha=0.15)
ax2.axvline(x=30, color="black", alpha=0.9, linewidth=3, linestyle="dashed")
ax2.set_ylabel('Average Value')
ax2.set_xlabel('Time')
ax2.set_title('Incorrect Trials')
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.legend()

fig.suptitle('Neuromodulator activity at StimOnset', y=1.02, fontsize=16)
plt.tight_layout()
plt.show()


#%%
################################################################################################## 
##################################################################################################
############################## WORKS ############################################
""" 2.0 a) Individual plots each NM allContrasts at stimOn and feedback """
# colors and functions 
colors_contrast = {
    'DA': '#d62828',  # Red
    '5HT': '#8e44ad',  # Purple
    'NE': '#3498db',  # Blue
    'ACh': '#3cb371'  # Green
} 

def avg_sem(data):
    avg = np.mean(data, axis=1)
    sem = np.std(data, axis=1) / np.sqrt(data.shape[1])
    return avg, sem

def split_contrasts(psth_array, df_trials, event2="allContrasts"): 
    psth_100 = psth_array[:, (df_trials[event2] == 1)]
    avg100, sem100 = avg_sem(psth_100)
    psth_25 = psth_array[:, (df_trials[event2] == 0.25)]
    avg25, sem25 = avg_sem(psth_25)
    psth_12 = psth_array[:, (df_trials[event2] == 0.125)]
    avg12, sem12 = avg_sem(psth_12)
    psth_06 = psth_array[:, (df_trials[event2] == 0.0625)]
    avg06, sem06 = avg_sem(psth_06)
    psth_0 = psth_array[:, (df_trials[event2] == 0)]
    avg0, sem0 = avg_sem(psth_0) 
    return avg100, sem100, avg25, sem25, avg12, sem12, avg06, sem06, avg0, sem0 

def plot_contrasts(ax, NM, colors, event, psth_array, df_trials): 
    event2 = "allContrasts"
    avg100, sem100, avg25, sem25, avg12, sem12, avg06, sem06, avg0, sem0 = split_contrasts(psth_array=psth_array, df_trials=df_trials, event2="allContrasts") 

    ax.plot(avg100, color=colors, label="100")
    ax.fill_between(range(len(avg100)), avg100 - sem100, avg100 + sem100, color='gray', alpha=0.1)
    ax.plot(avg25, color=colors, alpha=0.6, label="25")
    ax.fill_between(range(len(avg25)), avg25 - sem25, avg25 + sem25, color='gray', alpha=0.1)
    ax.plot(avg12, color=colors, alpha=0.4, label="12")
    ax.fill_between(range(len(avg12)), avg12 - sem12, avg12 + sem12, color='gray', alpha=0.1)
    ax.plot(avg06, color=colors, alpha=0.2, label="6")
    ax.fill_between(range(len(avg06)), avg06 - sem06, avg06 + sem06, color='gray', alpha=0.1)
    ax.plot(avg0, color=colors, alpha=0.05, label="0")
    ax.fill_between(range(len(avg0)), avg0 - sem0, avg0 + sem0, color='gray', alpha=0.1)
    ax.axvline(x=30, linestyle='dashed', color='black')
    ax.legend()
    ax.set_title(f'All contrasts {NM}, aligned to {event}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

##########################################################################
# Plot for DA 
fig, axes = plt.subplots(1, figsize=(5, 5))
plot_contrasts(
    ax=axes,
    NM="DA",
    colors=colors_contrast["DA"], 
    event="stimOnTrigger_times",
    psth_array=psth_combined_DA_stim,
    df_trials=df_trials_combined_DA_stim
) 
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 

# Plot for 5HT 
fig, axes = plt.subplots(1, figsize=(5, 5))
plot_contrasts(
    ax=axes,
    NM="5HT",
    colors=colors_contrast["5HT"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_5HT_stim,
    df_trials=df_trials_combined_5HT_stim
)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 

# Plot for NE
fig, axes = plt.subplots(1, figsize=(5, 5))
plot_contrasts(
    ax=axes,
    NM="NE",
    colors=colors_contrast["NE"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_NE_stim,
    df_trials=df_trials_combined_NE_stim
)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 

# Plot for ACh 
fig, axes = plt.subplots(1, figsize=(5, 5))
plot_contrasts(
    ax=axes,
    NM="ACh",
    colors=colors_contrast["ACh"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_ACh_stim,
    df_trials=df_trials_combined_ACh_stim
) 
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 

##########################################################################
# Plot for DA 
fig, axes = plt.subplots(1, figsize=(5, 5))
plot_contrasts(
    ax=axes,
    NM="DA",
    colors=colors_contrast["DA"],
    event="feedback_times",
    psth_array=psth_combined_DA,
    df_trials=df_trials_combined_DA
) 
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 

# Plot for 5HT 
fig, axes = plt.subplots(1, figsize=(5, 5))
plot_contrasts(
    ax=axes,
    NM="5HT",
    colors=colors_contrast["5HT"],
    event="feedback_times",
    psth_array=psth_combined_5HT,
    df_trials=df_trials_combined_5HT
)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 

# Plot for NE
fig, axes = plt.subplots(1, figsize=(5, 5))
plot_contrasts(
    ax=axes,
    NM="NE",
    colors=colors_contrast["NE"],
    event="feedback_times",
    psth_array=psth_combined_NE,
    df_trials=df_trials_combined_NE
)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 

# Plot for ACh 
fig, axes = plt.subplots(1, figsize=(5, 5))
plot_contrasts(
    ax=axes,
    NM="ACh",
    colors=colors_contrast["ACh"],
    event="feedback_times",
    psth_array=psth_combined_ACh,
    df_trials=df_trials_combined_ACh
) 
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()

#%%
##################################################################################################
##################################################################################################
############################## WORKS ############################################
""" 2.1 2x2 4NMs allContrasts no diff between correct and incorrect """
colors_contrast = {
    'DA': '#d62828',  # Red
    '5HT': '#8e44ad',  # Purple
    'NE': '#3498db',  # Blue
    'ACh': '#3cb371'  # Green
} 

def avg_sem(data):
    avg = np.mean(data, axis=1)
    sem = np.std(data, axis=1) / np.sqrt(data.shape[1])
    return avg, sem

def split_contrasts(psth_array, df_trials, event2="allContrasts"): 
    psth_100 = psth_array[:, (df_trials[event2] == 1)]
    avg100, sem100 = avg_sem(psth_100)
    psth_25 = psth_array[:, (df_trials[event2] == 0.25)]
    avg25, sem25 = avg_sem(psth_25)
    psth_12 = psth_array[:, (df_trials[event2] == 0.125)]
    avg12, sem12 = avg_sem(psth_12)
    psth_06 = psth_array[:, (df_trials[event2] == 0.0625)]
    avg06, sem06 = avg_sem(psth_06)
    psth_0 = psth_array[:, (df_trials[event2] == 0)]
    avg0, sem0 = avg_sem(psth_0) 
    return avg100, sem100, avg25, sem25, avg12, sem12, avg06, sem06, avg0, sem0 

def plot_contrasts(ax, NM, colors, event, psth_array, df_trials): 
    event2 = "allContrasts"
    avg100, sem100, avg25, sem25, avg12, sem12, avg06, sem06, avg0, sem0 = split_contrasts(psth_array=psth_array, df_trials=df_trials, event2="allContrasts") 

    ax.plot(avg100, color=colors, label="100")
    ax.fill_between(range(len(avg100)), avg100 - sem100, avg100 + sem100, color='gray', alpha=0.1)
    ax.plot(avg25, color=colors, alpha=0.6, label="25")
    ax.fill_between(range(len(avg25)), avg25 - sem25, avg25 + sem25, color='gray', alpha=0.1)
    ax.plot(avg12, color=colors, alpha=0.4, label="12")
    ax.fill_between(range(len(avg12)), avg12 - sem12, avg12 + sem12, color='gray', alpha=0.1)
    ax.plot(avg06, color=colors, alpha=0.2, label="6")
    ax.fill_between(range(len(avg06)), avg06 - sem06, avg06 + sem06, color='gray', alpha=0.1)
    ax.plot(avg0, color=colors, alpha=0.05, label="0")
    ax.fill_between(range(len(avg0)), avg0 - sem0, avg0 + sem0, color='gray', alpha=0.1)
    ax.axvline(x=30, linestyle='dashed', color='black')
    ax.legend()
    ax.set_title(f'All contrasts {NM}, aligned to {event}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#%%
##########################################################################
""" 2.1 a) 2x2 4NMs at stimOn """
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
# Plot for DA
plot_contrasts(
    ax=axes[0, 0],
    NM="DA",
    colors=colors_contrast["DA"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_DA_stim,
    df_trials=df_trials_combined_DA_stim
)
# Plot for 5HT 
plot_contrasts(
    ax=axes[0, 1],
    NM="5HT",
    colors=colors_contrast["5HT"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_5HT_stim,
    df_trials=df_trials_combined_5HT_stim
) 
# Plot for NE
plot_contrasts(
    ax=axes[1, 0],
    NM="NE",
    colors=colors_contrast["NE"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_NE_stim,
    df_trials=df_trials_combined_NE_stim
) 
# Plot for ACh
plot_contrasts(
    ax=axes[1, 1],
    NM="ACh",
    colors=colors_contrast["ACh"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_ACh_stim,
    df_trials=df_trials_combined_ACh_stim
)
# Repeat for other neuromodulators as needed
fig.suptitle('Neuromodulator activity at Stim Onset', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 

##########################################################################
""" 2.1 b) 2x2 4NMs at feedback """
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
# Plot for DA
plot_contrasts(
    ax=axes[0, 0],
    NM="DA",
    colors=colors_contrast["DA"],
    event="feedback_times",
    psth_array=psth_combined_DA,
    df_trials=df_trials_combined_DA
)
# Plot for 5HT 
plot_contrasts(
    ax=axes[0, 1],
    NM="5HT",
    colors=colors_contrast["5HT"],
    event="feedback_times",
    psth_array=psth_combined_5HT,
    df_trials=df_trials_combined_5HT
) 
# Plot for NE
plot_contrasts(
    ax=axes[1, 0],
    NM="NE",
    colors=colors_contrast["NE"],
    event="feedback_times",
    psth_array=psth_combined_NE,
    df_trials=df_trials_combined_NE
) 
# Plot for ACh
plot_contrasts(
    ax=axes[1, 1],
    NM="ACh",
    colors=colors_contrast["ACh"],
    event="feedback_times",
    psth_array=psth_combined_ACh,
    df_trials=df_trials_combined_ACh
)
# Repeat for other neuromodulators as needed
fig.suptitle('Neuromodulator activity at Feedback Outcome', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 

##################################################################################################
##################################################################################################
##################################################################################################

#%%
##################################################################################################
##################################################################################################
############################## WORKS ############################################
""" 3.0 2x2 4NMs allContrasts diff between correct and incorrect """ 
"""DOUBLE CHECK"""
# Define colors for neuromodulators
colors_contrast = {
    'DA': '#d62828',  # Red
    '5HT': '#8e44ad',  # Purple
    'NE': '#3498db',  # Blue
    'ACh': '#3cb371'  # Green
}

def avg_sem(data):
    avg = np.mean(data, axis=1)
    sem = np.std(data, axis=1) / np.sqrt(data.shape[1])
    return avg, sem

def split_contrasts(psth_array, df_trials, event1="feedbackType", event2="allContrasts"): 
    # Correct trials
    psth_100_c = psth_array[:, (df_trials[event1] == 1) & (df_trials[event2] == 1)]
    avg100_c, sem100_c = avg_sem(psth_100_c)
    psth_25_c = psth_array[:, (df_trials[event1] == 1) & (df_trials[event2] == 0.25)]
    avg25_c, sem25_c = avg_sem(psth_25_c)
    psth_12_c = psth_array[:, (df_trials[event1] == 1) & (df_trials[event2] == 0.125)]
    avg12_c, sem12_c = avg_sem(psth_12_c)
    psth_06_c = psth_array[:, (df_trials[event1] == 1) & (df_trials[event2] == 0.0625)]
    avg06_c, sem06_c = avg_sem(psth_06_c)
    psth_0_c = psth_array[:, (df_trials[event1] == 1) & (df_trials[event2] == 0)]
    avg0_c, sem0_c = avg_sem(psth_0_c)
    
    # Incorrect trials
    psth_100_i = psth_array[:, (df_trials[event1] == -1) & (df_trials[event2] == 1)]
    avg100_i, sem100_i = avg_sem(psth_100_i)
    psth_25_i = psth_array[:, (df_trials[event1] == -1) & (df_trials[event2] == 0.25)]
    avg25_i, sem25_i = avg_sem(psth_25_i)
    psth_12_i = psth_array[:, (df_trials[event1] == -1) & (df_trials[event2] == 0.125)]
    avg12_i, sem12_i = avg_sem(psth_12_i)
    psth_06_i = psth_array[:, (df_trials[event1] == -1) & (df_trials[event2] == 0.0625)]
    avg06_i, sem06_i = avg_sem(psth_06_i)
    psth_0_i = psth_array[:, (df_trials[event1] == -1) & (df_trials[event2] == 0)]
    avg0_i, sem0_i = avg_sem(psth_0_i)
    
    return (avg100_c, sem100_c, avg25_c, sem25_c, avg12_c, sem12_c, avg06_c, sem06_c, avg0_c, sem0_c, 
            avg100_i, sem100_i, avg25_i, sem25_i, avg12_i, sem12_i, avg06_i, sem06_i, avg0_i, sem0_i)

def plot_contrasts(ax, NM, colors, event, psth_array, df_trials): 
    (avg100_c, sem100_c, avg25_c, sem25_c, avg12_c, sem12_c, avg06_c, sem06_c, avg0_c, sem0_c, 
     avg100_i, sem100_i, avg25_i, sem25_i, avg12_i, sem12_i, avg06_i, sem06_i, avg0_i, sem0_i) = split_contrasts(psth_array=psth_array, df_trials=df_trials, event1="feedbackType", event2="allContrasts") 

    # Correct trials
    ax.plot(avg100_c, color=colors, label="100 Correct")
    ax.fill_between(range(len(avg100_c)), avg100_c - sem100_c, avg100_c + sem100_c, color=colors, alpha=0.1)
    ax.plot(avg25_c, color=colors, alpha=0.6, label="25 Correct")
    ax.fill_between(range(len(avg25_c)), avg25_c - sem25_c, avg25_c + sem25_c, color=colors, alpha=0.1)
    ax.plot(avg12_c, color=colors, alpha=0.4, label="12 Correct")
    ax.fill_between(range(len(avg12_c)), avg12_c - sem12_c, avg12_c + sem12_c, color=colors, alpha=0.1)
    ax.plot(avg06_c, color=colors, alpha=0.2, label="6 Correct")
    ax.fill_between(range(len(avg06_c)), avg06_c - sem06_c, avg06_c + sem06_c, color=colors, alpha=0.1)
    ax.plot(avg0_c, color=colors, alpha=0.05, label="0 Correct")
    ax.fill_between(range(len(avg0_c)), avg0_c - sem0_c, avg0_c + sem0_c, color=colors, alpha=0.1)
    
    # Incorrect trials
    ax.plot(avg100_i, color=colors, linestyle='--', label="100 Incorrect")
    ax.fill_between(range(len(avg100_i)), avg100_i - sem100_i, avg100_i + sem100_i, color=colors, alpha=0.1)
    ax.plot(avg25_i, color=colors, linestyle='--', alpha=0.6, label="25 Incorrect")
    ax.fill_between(range(len(avg25_i)), avg25_i - sem25_i, avg25_i + sem25_i, color=colors, alpha=0.1)
    ax.plot(avg12_i, color=colors, linestyle='--', alpha=0.4, label="12 Incorrect")
    ax.fill_between(range(len(avg12_i)), avg12_i - sem12_i, avg12_i + sem12_i, color=colors, alpha=0.1)
    ax.plot(avg06_i, color=colors, linestyle='--', alpha=0.2, label="6 Incorrect")
    ax.fill_between(range(len(avg06_i)), avg06_i - sem06_i, avg06_i + sem06_i, color=colors, alpha=0.1)
    ax.plot(avg0_i, color=colors, linestyle='--', alpha=0.05, label="0 Incorrect")
    ax.fill_between(range(len(avg0_i)), avg0_i - sem0_i, avg0_i + sem0_i, color=colors, alpha=0.1)

    ax.axvline(x=30, linestyle='dashed', color='black')
    ax.legend()
    ax.set_title(f'{NM}, aligned to {event}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

##########################################################################
""" 3.0 a) 2x2 4NMs at stimOn """
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
# Plot for DA
plot_contrasts(
    ax=axes[0, 0],
    NM="DA",
    colors=colors_contrast["DA"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_DA_stim,
    df_trials=df_trials_combined_DA_stim
)
# Plot for 5HT 
plot_contrasts(
    ax=axes[0, 1],
    NM="5HT",
    colors=colors_contrast["5HT"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_5HT_stim,
    df_trials=df_trials_combined_5HT_stim
) 
# Plot for NE
plot_contrasts(
    ax=axes[1, 0],
    NM="NE",
    colors=colors_contrast["NE"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_NE_stim,
    df_trials=df_trials_combined_NE_stim
) 
# Plot for ACh
plot_contrasts(
    ax=axes[1, 1],
    NM="ACh",
    colors=colors_contrast["ACh"],
    event="stimOnTrigger_times",
    psth_array=psth_combined_ACh_stim,
    df_trials=df_trials_combined_ACh_stim
)
fig.suptitle('Neuromodulator activity at Stim Onset', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 

##########################################################################
""" 3.0 b) 2x2 4NMs at feedback """
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=True)
# Plot for DA
plot_contrasts(
    ax=axes[0, 0],
    NM="DA",
    colors=colors_contrast["DA"],
    event="feedback_times",
    psth_array=psth_combined_DA,
    df_trials=df_trials_combined_DA
)
# Plot for 5HT 
plot_contrasts(
    ax=axes[0, 1],
    NM="5HT",
    colors=colors_contrast["5HT"],
    event="feedback_times",
    psth_array=psth_combined_5HT,
    df_trials=df_trials_combined_5HT
) 
# Plot for NE
plot_contrasts(
    ax=axes[1, 0],
    NM="NE",
    colors=colors_contrast["NE"],
    event="feedback_times",
    psth_array=psth_combined_NE,
    df_trials=df_trials_combined_NE
) 
# Plot for ACh
plot_contrasts(
    ax=axes[1, 1],
    NM="ACh",
    colors=colors_contrast["ACh"],
    event="feedback_times",
    psth_array=psth_combined_ACh,
    df_trials=df_trials_combined_ACh
)
fig.suptitle('Neuromodulator activity at Feedback', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show() 


#%%
##################################################################################################
##################################################################################################
##############################  ############################################
""" 4.1 2x2 4NMs reaction times diff between correct and incorrect """ 
import numpy as np
import matplotlib.pyplot as plt

# Define colors for neuromodulators
colors_contrast = {
    'DA': '#d62828',  # Red
    '5HT': '#8e44ad',  # Purple
    'NE': '#3498db',  # Blue
    'ACh': '#3cb371'  # Green
}

def avg_sem(data):
    avg = np.mean(data, axis=1)
    sem = np.std(data, axis=1) / np.sqrt(data.shape[1])
    return avg, sem

def split_reaction_times(psth_array, df_trials, event1="feedbackType", reaction_time_col="reactionTime"): 
    # Define reaction time thresholds
    rt_thresholds = [0.5, 1.5]
    
    # Split data by reaction time thresholds and feedback type (correct vs incorrect)
    splits = {}
    for feedback in [1, -1]:
        for i, rt_threshold in enumerate(rt_thresholds):
            if i == 0:
                mask = df_trials[reaction_time_col] < rt_threshold
            else:
                mask = df_trials[reaction_time_col] > rt_threshold
            key = f'{"lt_05" if i == 0 else "gt_15"}_{"c" if feedback == 1 else "i"}'
            splits[key] = psth_array[:, (df_trials[event1] == feedback) & mask]
    
    # Get average and SEM for each split
    results = {}
    for key, data in splits.items():
        if data.shape[1] > 0:  # Ensure there's data to calculate avg and sem
            results[key] = avg_sem(data)
        else:
            print(f"No data for {key}")
            results[key] = (np.array([]), np.array([]))
    
    return results

def plot_reaction_times(ax, NM, colors, event, psth_array, df_trials): 
    results = split_reaction_times(psth_array=psth_array, df_trials=df_trials, event1="feedbackType", reaction_time_col="reactionTime") 

    rt_colors = {'c': 'blue', 'i': 'red'}  # Blue for correct, red for incorrect

    for i, rt in enumerate(['lt_05', 'gt_15']):
        for feedback in ['c', 'i']:
            avg, sem = results[f'{rt}_{feedback}']
            if avg.size > 0:  # Check if there's data to plot
                linestyle = '-' if rt == 'lt_05' else '--'
                label = f'{rt} {"Correct" if feedback == "c" else "Incorrect"}'
                ax.plot(avg, color=rt_colors[feedback], linestyle=linestyle, label=label)
                ax.fill_between(range(len(avg)), avg - sem, avg + sem, color=rt_colors[feedback], alpha=0.1)

    ax.axvline(x=30, linestyle='dashed', color='black')
    ax.legend()
    ax.set_title(f'{NM}, aligned to {event}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Create 2x2 subplots for DA, 5HT, NE, ACh
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=False)

# Plot for DA
plot_reaction_times(
    ax=axes[0, 0],
    NM="DA",
    colors=colors_contrast["DA"],
    event="feedback_times",
    psth_array=psth_combined_DA,
    df_trials=df_trials_combined_DA
)

# Plot for 5HT
plot_reaction_times(
    ax=axes[0, 1],
    NM="5HT",
    colors=colors_contrast["5HT"],
    event="feedback_times",
    psth_array=psth_combined_5HT,
    df_trials=df_trials_combined_5HT
)

# Plot for NE
plot_reaction_times(
    ax=axes[1, 0],
    NM="NE",
    colors=colors_contrast["NE"],
    event="feedback_times",
    psth_array=psth_combined_NE,
    df_trials=df_trials_combined_NE
)

# Plot for ACh
plot_reaction_times(
    ax=axes[1, 1],
    NM="ACh",
    colors=colors_contrast["ACh"],
    event="feedback_times",
    psth_array=psth_combined_ACh,
    df_trials=df_trials_combined_ACh
)

fig.suptitle('Neuromodulator activity at Feedback', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()










#%% 
""" StimOn """

# Define colors for neuromodulators
colors_contrast = {
    'DA': '#d62828',  # Red
    '5HT': '#8e44ad',  # Purple
    'NE': '#3498db',  # Blue
    'ACh': '#3cb371'  # Green
}

def avg_sem(data):
    avg = np.mean(data, axis=1)
    sem = np.std(data, axis=1) / np.sqrt(data.shape[1])
    return avg, sem

def split_reaction_times(psth_array, df_trials, event1="feedbackType", reaction_time_col="reactionTime"): 
    # Define reaction time thresholds
    rt_thresholds = {
        'lt_05': df_trials[reaction_time_col] < 0.5,
        'gt_15': df_trials[reaction_time_col] > 1.5
    }
    
    # Split data by reaction time thresholds and feedback type (correct vs incorrect)
    splits = {}
    for feedback in [1, -1]:
        for category, mask in rt_thresholds.items():
            key = f'{category}_{"c" if feedback == 1 else "i"}'
            splits[key] = psth_array[:, (df_trials[event1] == feedback) & mask]
    
    # Get average and SEM for each split
    results = {}
    for key, data in splits.items():
        if data.shape[1] > 0:  # Ensure there's data to calculate avg and sem
            results[key] = avg_sem(data)
        else:
            print(f"No data for {key}")
            results[key] = (np.array([]), np.array([]))
    
    return results

def plot_reaction_times(ax, NM, colors, event, psth_array, df_trials): 
    results = split_reaction_times(psth_array=psth_array, df_trials=df_trials, event1="feedbackType", reaction_time_col="reactionTime") 

    rt_styles = {
        'lt_05': ('-', colors[0]),  # Solid line for <0.5s
        'gt_15': ('--', colors[0])  # Dashed line for >1.5s
    }
    
    feedback_colors = {'c': 'blue', 'i': 'red'}  # Blue for correct, red for incorrect

    for category, (linestyle, color) in rt_styles.items():
        for feedback in ['c', 'i']:
            avg, sem = results[f'{category}_{feedback}']
            if avg.size > 0:  # Check if there's data to plot
                label = f'{category} {"Correct" if feedback == "c" else "Incorrect"}'
                ax.plot(avg, color=feedback_colors[feedback], linestyle=linestyle, label=label)
                ax.fill_between(range(len(avg)), avg - sem, avg + sem, color=feedback_colors[feedback], alpha=0.1)

    ax.axvline(x=30, linestyle='dashed', color='black')
    ax.legend()
    ax.set_title(f'{NM}, aligned to {event}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Create 2x2 subplots for DA, 5HT, NE, ACh
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=True, sharey=False)

# Plot for DA
plot_reaction_times(
    ax=axes[0, 0],
    NM="DA",
    colors=['#d62828'],  # Use DA's color for reaction times
    event="stim_times",
    psth_array=psth_combined_DA_stim,
    df_trials=df_trials_combined_DA_stim
)
# Plot for 5HT
plot_reaction_times(
    ax=axes[0, 1],
    NM="5HT",
    colors=['#8e44ad'],  # Use 5HT's color for reaction times
    event="stim_times",
    psth_array=psth_combined_5HT_stim,
    df_trials=df_trials_combined_5HT_stim
)
# Plot for NE
plot_reaction_times(
    ax=axes[1, 0],
    NM="NE",
    colors=['#3498db'],  # Use NE's color for reaction times
    event="stim_times",
    psth_array=psth_combined_NE_stim,
    df_trials=df_trials_combined_NE_stim
)
# Plot for ACh
plot_reaction_times(
    ax=axes[1, 1],
    NM="ACh",
    colors=['#3cb371'],  # Use ACh's color for reaction times
    event="stim_times",
    psth_array=psth_combined_ACh_stim,
    df_trials=df_trials_combined_ACh_stim
)

fig.suptitle('Neuromodulator activity at Stimulus', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()














#%%
""" ADDING RESPONSE TIME """ 

df_trials_combined_NE["responseTime"] = df_trials_combined_NE["feedback_times"] - df_trials_combined_NE["stimOnTrigger_times"]
df_trials_combined_NE_stim["responseTime"] = df_trials_combined_NE_stim["feedback_times"] - df_trials_combined_NE_stim["stimOnTrigger_times"]
df_trials_combined_ACh["responseTime"] = df_trials_combined_ACh["feedback_times"] - df_trials_combined_ACh["stimOnTrigger_times"] 
df_trials_combined_ACh_stim["responseTime"] = df_trials_combined_ACh_stim["feedback_times"] - df_trials_combined_ACh_stim["stimOnTrigger_times"] 
df_trials_combined_DA_stim["responseTime"] = df_trials_combined_DA_stim["feedback_times"] - df_trials_combined_DA_stim["stimOnTrigger_times"] 
df_trials_combined_5HT_stim["responseTime"] = df_trials_combined_5HT_stim["feedback_times"] - df_trials_combined_5HT_stim["stimOnTrigger_times"] 
















# %%
