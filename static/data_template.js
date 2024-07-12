// Query string parameters.
const QUERY_PARAMS = new Proxy(new URLSearchParams(window.location.search), {
    get: (searchParams, prop) => searchParams.get(prop),
});

// Use LZString library for decompression
function decompressFromBase64(compressedString) {
    return LZString.decompressFromBase64(compressedString);
}

// Compressed dictionary passed by Flask in the render() function.
// NOTE: automatically generated.
const FLASK_CTX_COMPRESSED = "N4IgygomYJIPIDkwgFwAIDaBdANGkAMhAOIQIAiy6wAvniORAGICCAqgQCoD6ACjOVT4AZgFMAzAA4ADAHYAnOIC0M8QFYlAFgCMm0UoBG040unyDYtaIDGo+QCYAhiHqNWHHpQichIAwHcAWxAaIA==";

// Load the compressed data from FLASK_CTX_COMPRESSED
const decompressedData = decompressFromBase64(FLASK_CTX_COMPRESSED);

// Parse the decompressed data as JSON
const FLASK_CTX = JSON.parse(decompressedData);

// Current context.
var CTX = {
    dset: QUERY_PARAMS.dset || FLASK_CTX.DEFAULT_DSET, // initial tab dataset
    eid: QUERY_PARAMS.eid || FLASK_CTX.DEFAULT_EID, // initial probe UUID
    tid: parseInt(QUERY_PARAMS.tid, 10) || 0, // trial ID
    trials: [],
    trial_onsets: [], // for all trials, including nan ones, so we can index by the tid
    trial_offsets: [],
    dur: 0, // session duration
};