
/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

const DEFAULT_PARAMS = {
};
var CTX = {
    pid: null,
    tid: 0,
};


/*************************************************************************************************/
/*  Utils                                                                                        */
/*************************************************************************************************/

function isEmpty(obj) {
    // https://stackoverflow.com/a/679937/1595060
    return Object.keys(obj).length === 0;
};



function throttle(func, wait, options) {
    var context, args, result;
    var timeout = null;
    var previous = 0;
    if (!options) options = {};
    var later = function () {
        previous = options.leading === false ? 0 : Date.now();
        timeout = null;
        result = func.apply(context, args);
        if (!timeout) context = args = null;
    };
    return function () {
        var now = Date.now();
        if (!previous && options.leading === false) previous = now;
        var remaining = wait - (now - previous);
        context = this;
        args = arguments;
        if (remaining <= 0 || remaining > wait) {
            if (timeout) {
                clearTimeout(timeout);
                timeout = null;
            }
            previous = now;
            result = func.apply(context, args);
            if (!timeout) context = args = null;
        } else if (!timeout && options.trailing !== false) {
            timeout = setTimeout(later, remaining);
        }
        return result;
    };
};



// Display an array buffer.
function show(arrbuf) {
    const blob = new Blob([arrbuf]);
    const url = URL.createObjectURL(blob);
    const img = document.getElementById('imgRaster');
    let w = img.offsetWidth;

    // Update the raster plot
    // const img = document.getElementById('imgRaster');
    // img.src = url;

    var t0 = px2time(0);
    var t1 = px2time(w);
    var t = .5 * (t0 + t1);

    Plotly.update('imgRaster', {}, {
        "images[0].source": url,
        "xaxis.ticktext": [t0.toFixed(3), t.toFixed(3), t1.toFixed(3)],
    });

    setLineOffset();
};


function tablefromjson(json, elementID) {

    var table_data = `<table>`
    for (let key in json)  {
        var row = `<tr>
                    <th>${key}</th>
                    <td>${json[key]}</td>
                   </tr>`
        table_data += row
    }
    table_data += `</table>`

    document.getElementById(elementID).innerHTML = table_data
}



/*************************************************************************************************/
/*  Sliders                                                                                      */
/*************************************************************************************************/

function initSlider(id, initRange, fullRange) {

    var el = document.getElementById(id);
    if (el.noUiSlider)
        el.noUiSlider.destroy();

    noUiSlider.create(el, {
        start: initRange,
        connect: true,
        range: {
            'min': fullRange[0],
            'max': fullRange[1]
        },
        tooltips: true,
    });
};



function onSliderChange(id, callback) {
    var el = document.getElementById(id);
    el.noUiSlider.on('update',
        function (values, handle, unencoded, tap, positions, noUiSlider) {
            min = parseFloat(values[0]);
            max = parseFloat(values[1]);
            callback(min, max);
        });
};




/*************************************************************************************************/
/*  Setup functions                                                                              */
/*************************************************************************************************/

function setupSliders() {

    // Alpha slider

    // initSlider('sliderAlpha', window.params.alpha_range, window.params.alpha_lims);

    // onSliderChange('sliderAlpha', function (min, max) {
    //     window.params.alpha_range = [min, max];
    //     updateParamsData();
    // });

};



async function selectSession(pid) {
    CTX.pid = pid;

    // Show the session details.
    var url = `/api/session/${pid}/details`;
    var r = await fetch(url);
    var details = await r.json();
    // Pop the cluster ids into a new variable
    var cluster_ids = details["cluster_ids"]
    delete details["cluster_ids"];

    // Make table with session details
    tablefromjson(details, 'sessionDetails')

    // Show the raster plot.
    // url = `/api/session/${pid}/raster`;
    // document.getElementById('rasterPlot').src = url;

    url = `/api/session/${pid}/psychometric`;
    document.getElementById('psychometricPlot').src = url;

        url = `/api/session/${pid}/clusters`;
    document.getElementById('clusterGoodBadPlot').src = url;


    // Set the trial selector.
    var s = document.getElementById('trialSelector');
    $('#trialSelector option').remove();
    for (var i = 0; i < details["N trials"]; i++) {
        s.options[s.options.length] = new Option(`trial #${i}`, i);
    }

    // Set the cluster selector.
    var s = document.getElementById('clusterSelector');
    $('#clusterSelector option').remove();
    for (var cluster_id of cluster_ids) {
        s.options[s.options.length] = new Option(`cluster #${cluster_id}`, cluster_id);
    }

    // Update the other plots.
    selectTrial(pid, 0);
    // Need to make sure first cluster is a good one, otherwise get error
    selectCluster(pid, cluster_ids[0]);
}



async function selectTrial(pid, tid) {
    // Show the trial raster plot.
    var url = `/api/session/${pid}/trial_raster/${tid}`;
    document.getElementById('trialRasterPlot').src = url;

    var url = `/api/session/${pid}/raster/trial/${tid}`;
    document.getElementById('rasterPlot').src = url;

    // Show information about trials in table
    var url = `/api/session/${pid}/trial_details/${tid}`;
    var r = await fetch(url).then();
    var details = await r.json();

    tablefromjson(details, 'trialDetails')

}



async function selectCluster(pid, cid) {
    console.log(cid)
    var url = `/api/session/${pid}/cluster/${cid}`;
    document.getElementById('clusterPlot').src = url;

    var url = `/api/session/${pid}/cluster_response/${cid}`;
    document.getElementById('clusterResponsePlot').src = url;

    var url = `/api/session/${pid}/cluster_properties/${cid}`;
    document.getElementById('clusterPropertiesPlot').src = url;

    // Show information about cluster in table
    var url = `/api/session/${pid}/cluster_details/${cid}`;
    var r = await fetch(url).then();
    var details = await r.json();

    tablefromjson(details, 'clusterDetails')

}



function setupDropdowns() {

    // Session selector.
    document.getElementById('sessionSelector').onchange = async function (e) {
        var pid = e.target.value;
        if(!pid) return;
        await selectSession(pid);
    }

    // Trial selector.
    document.getElementById('trialSelector').onchange = function (e) {
        var tid = e.target.value;
        if(!tid) return;
        selectTrial(CTX.pid, tid);
    }

    // Cluster selector.
    document.getElementById('clusterSelector').onchange = function (e) {
        var cid = e.target.value;
        console.log(cid)
        if(!cid) return;
        selectCluster(CTX.pid, cid);
    }

    // Initial selection.
    document.getElementById('sessionSelector').selectedIndex = 0;
    var pid = document.getElementById('sessionSelector').value;
    selectSession(pid);
};



function setupInputs() {

    // document.getElementById('clusterInput').onchange = function (e) {
    //     var cl = parseInt(e.target.value);
    // };

};



function setupButtons() {
    // document.getElementById('resetButton').onclick = function (e) {
    //     console.log("reset params");
    //     resetParams();
    // }
};





/*************************************************************************************************/
/*  Params browser persistence                                                                   */
/*************************************************************************************************/

function loadParams() {
    window.params = JSON.parse(localStorage.params || "{}");
    if (isEmpty(window.params)) {
        window.params = DEFAULT_PARAMS;
    }
};



function saveParams() {
    localStorage.params = JSON.stringify(window.params);
};



function resetParams() {
    window.params = DEFAULT_PARAMS;
    saveParams();
    setupSliders();
    setupDropdowns();
};



function setupPersistence() {
    loadParams();
    window.onbeforeunload = saveParams;
};



/*************************************************************************************************/
/*  Entry point                                                                                  */
/*************************************************************************************************/

function load() {
    setupPersistence();
    setupSliders();
    setupDropdowns();
    setupInputs();
    setupButtons();
};



$(document).ready(function () {
    load();
});
