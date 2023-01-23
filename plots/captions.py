CAPTIONS = {
    'figure1': {
        'A': 'Raster plot of the spikes of good units along the probe depth (um), throughout the whole experiment time (s)',
        'B': 'Amplitude (uV) of units along the probe depth (um), colored by the quality: good units in green, multi-units (mua) in '
             'red',
        'C': 'Amplitude (uV) of good units along the probe depth (um), colored by their firing rate (Hz)',
        'D': 'Root mean square (RMS) values of the action potential (AP) band along the probe depth (um)',
        'E': 'Power (dB) in the delta band (0-4Hz) of the low-frequency (LFP) band',
        'F': 'Brain region acronyms along the probe depth (um)',
        'G': 'TODO: session overview showing reaction time , blue and red indicate and background shades indicate different probability '
             'blocks within the session.',
        'H': 'Coronal slice of Allen Atlas showing location of probe insertion',
        'I': '3 raw electrophysiology data snippets of 50 millisecond (ms) duration are shown (one at T=600s, 1800s, and 3000s). '
             'The raw data recorded on each channel is plotted as a gray code, and ordered along the probe depth. Divergence from '
             'the gray color (black or white) indicates a shift from baseline activity. Overlaid dots indicate when spikes were '
             'detected (green: good units, red: multi-units).',
        'J': 'Brain region acronyms along the probe depth (um)',
    },
    'figure2': {
        'A': 'Psychometric curves computed in different types of trial block ; for a trial block type, the probability of the visual '
             'stimulus being on the left is either 0.5 (black), 0.2 (orange), or 0.8 (blue)',
        'B': 'Reaction time (s) curves computed in different trial blocks (same color scheme as the psychometric curve plots)',
        'C': 'Reaction time (s) for each trial, plotted throughout the whole experiment',
        'D': 'Raster plot of the left paw speed, sorted by trial outcome (correct in blue, incorrect in red) and aligned to the '
             'visual stimulus onset; averages (mean and STD) for each condition are displayed at the top with the same color scheme. '
             'Trial numbers within a condition are kept ascending from bottom to top. Left paw speed has been computed from DLC '
             'points detected on the left camera',
        'E': 'Raster plot of the nose tip speed, sorted by trial outcome (correct in blue, incorrect in red) and aligned to the '
             'visual stimulus onset; averages (mean and STD) for each condition are displayed at the top with the same color scheme. '
             'Trial numbers within a condition are kept ascending from bottom to top. Nose tip speed has been computed from DLC '
             'points detected on the left camera',
        'F': 'Raster plot of the whisker pad motion energy, sorted by trial outcome (correct in blue, incorrect in red) and aligned '
             'to the visual stimulus onset; averages (mean and STD) for each condition are displayed at the top with the same color '
             'scheme. Trial numbers within a condition are kept ascending from bottom to top. Motion energy has been computed using '
             'the left camera',
        'G': 'Raster plot of pupil diameter, sorted by trial outcome (correct in blue, incorrect in red) and aligned to the visual '
             'stimulus onset; averages (mean and STD) for each condition are displayed at the top with the same color scheme. '
             'Trial numbers within a condition are kept ascending from bottom to top. The pupil diameter has been computed from DLC '
             'points on the left camera ',
        'H': 'Raster plot of wheel velocity, sorted by visual stimulus side (right in yellow, left in green) and aligned to the '
             'movement onset ; averages (mean and STD) for each condition are displayed at the top with the same color scheme. '
             'Trial numbers within a condition are kept ascending from bottom to top',
        'I': 'Raster plot of lick events, sorted by visual stimulus side (right in yellow, left in green) and aligned to feedback '
             'time ; averages (mean and STD) for each condition are displayed at the top with the same color scheme. Trial numbers '
             'within a condition are kept ascending from bottom to top. Lick events have been computed using DLC points detected '
             'on both the left and right cameras',
    },


    'figure3': {
        'A': 'Raster plot of the spikes of all good units along the probe depth (um), throughout the whole '
             'experiment time (s) ; the dash line indicates the trial selected for visualization',
        'B': 'Zoomed-in view of the raster around the time of the selected trial ; the lines indicate events in the trial '
             '(go cue in blue, first movement in green, and feedback in red).',
        'C': 'Brain region acronyms along the probe depth (um)',
        'D': 'Wheel position (normalised to position at first) during selected trial; the lines indicate events in '
             'the trial (go cue in blue, first movement in green, and feedback in red)',
        'E': 'DLC detected left paw speed during selected trial; the lines indicate events in the trial (go cue in blue, first ' \
             'movement in green, and feedback in red)',
    },


    'figure4': {
        'A': 'Average of the firing rate (z-scored) for good units along the probe depths (um), aligned to visual stimulus onset',
        'B': 'Average of the firing rate (z-scored) for good units along the probe depths (um), aligned to first movement',
        'C': 'Average of the firing rate (z-scored) for good units along the probe depths (um), aligned to feedback',
        'D': 'Brain region acronyms along the probe depth (um)',
    },


    'figure5': {
        'A': 'Amplitude (uV) of good units along the probe depth (um), colored by the brain region color ; the selected unit for '
             'visualization is highlighted in black',
        'B': 'Raster plot of the spikes of the selected good unit, with the trial block type shown (the probability of the visual '
             'stimulus being on the left is either 0.5 (black), 0.2 (orange), or 0.8 (blue) and aligned to the visual stimulus onset ;'
             ' averages (mean and STD) for each condition are displayed at the top with the same color scheme. Trial numbers are '
             'ascending from bottom to top, and kept in a similar order as during the experiment',
        'C': 'Raster plot of the spikes of the selected good unit, sorted by visual stimulus contrasts (0, 6.25, 12.5, 25, 100%, '
             'from pale to dark gray) and aligned to the visual stimulus onset ; averages (mean and STD) for each condition are '
             'displayed at the top with the same color scheme. Trial numbers within a condition are kept ascending from bottom to '
             'top.',
        'D': 'Raster plot of the spikes of the selected good unit, sorted by visual stimulus side (right in yellow, left in green) '
             'and aligned to the movement onset ; averages (mean and STD) for each condition are displayed at the top with the same '
             'color scheme. Trial numbers within a condition are kept ascending from bottom to top.',
        'E': 'Raster plot of the spikes of the selected good unit, sorted by trial outcome (correct in blue, incorrect in red) and '
             'aligned to feedback time ; averages (mean and STD) for each condition are displayed at the top with the same color '
             'scheme. Trial numbers within a condition are kept ascending from bottom to top.',
        'F': 'Template waveforms of the selected good unit, presented for the electrode channels where spikes were detected. '
             'Scale bar indicates amplitude of waveforms in uV and as a ratio to the standard deviation of the noise on the peak '
             'channel. The location of electrode channels within the whole probe is presented on the right.',
        'G': 'Autocorrelogram of the selected good unit.',
        'H': 'Inter-spike-interval (ISI) distribution for the selected good unit.',
        'I': 'Amplitude (uV) of the spikes of the selected good unit across the experiment time (s).'
    },
    'figure5_qc': {
        'A': 'Amplitude (uV) of all units along the probe depth (um), colored by the brain region color ; the selected unit for '
             'visualization is highlighted in black',
        'B': 'Raster plot of the spikes of the selected unit, with the trial block type shown (the probability of the visual '
             'stimulus being on the left is either 0.5 (black), 0.2 (orange), or 0.8 (blue) and aligned to the visual stimulus onset ;'
             ' averages (mean and STD) for each condition are displayed at the top with the same color scheme. Trial numbers are '
             'ascending from bottom to top, and kept in a similar order as during the experiment',
        'C': 'Raster plot of the spikes of the selected unit, sorted by visual stimulus contrasts (0, 6.25, 12.5, 25, 100%, '
             'from pale to dark gray) and aligned to the visual stimulus onset ; averages (mean and STD) for each condition are '
             'displayed at the top with the same color scheme. Trial numbers within a condition are kept ascending from bottom to '
             'top.',
        'D': 'Raster plot of the spikes of the selected unit, sorted by visual stimulus side (right in yellow, left in green) '
             'and aligned to the movement onset ; averages (mean and STD) for each condition are displayed at the top with the same '
             'color scheme. Trial numbers within a condition are kept ascending from bottom to top.',
        'E': 'Raster plot of the spikes of the selected unit, sorted by trial outcome (correct in blue, incorrect in red) and '
             'aligned to feedback time ; averages (mean and STD) for each condition are displayed at the top with the same color '
             'scheme. Trial numbers within a condition are kept ascending from bottom to top.',
        'F': 'Template waveforms of the selected unit, presented for the electrode channels where spikes were detected. '
             'Scale bar indicates amplitude of waveforms in uV and as a ratio to the standard deviation of the noise on the peak '
             'channel. The location of electrode channels within the whole probe is presented on the right.',
        'G': 'Amplitude (uV) of the spikes of the selected unit across the experiment time (s) and the distribution of the amplitude.',
        'H': 'Autocorrelogram of the selected unit'
    }
}