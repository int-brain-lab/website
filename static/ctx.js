var CTX = {
    dset: QUERY_PARAMS.dset || FLASK_CTX.DEFAULT_DSET, // initial tab dataset
    pid: QUERY_PARAMS.pid || FLASK_CTX.DEFAULT_PID, // initial probe UUID
    tid: parseInt(QUERY_PARAMS.tid, 10) || 0, // trial ID
    cid: parseInt(QUERY_PARAMS.cid, 10) || 0, // cluster ID
    trials: [],
    trial_onsets: [], // for all trials, including nan ones, so we can index by the tid
    trial_offsets: [],
    dur: 0, // session duration
    qc: parseInt(QUERY_PARAMS.qc, 10) || 0, // qc mode
    spikesorting: QUERY_PARAMS.spikesorting || DEFAULT_SPIKESORTING

};

console.log(CTX.spikesorting)
