
/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

const DEFAULT_PARAMS = {
};
var CTX = {
    pid: null,
    tid: 0,
};
var myGameInstance = null;


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
    for (let key in json) {
        var row = `<tr>
                    <th>${key}</th>
                    <td>${json[key]}</td>
                   </tr>`
        table_data += row
    }
    table_data += `</table>`

    document.getElementById(elementID).innerHTML = table_data
}



function isOnMobile() {
    let check = false;
    (function (a) { if (/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a) || /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0, 4))) check = true; })(navigator.userAgent || navigator.vendor || window.opera);
    return check;
};



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

function loadUnity() {

    // Disable Unity widget on smartphones.
    if (isOnMobile()) return;

    var canvas = document.querySelector("#unity-canvas");
    createUnityInstance(canvas, {
        dataUrl: "static/Build/IBLMini-webgl.data.gz",
        frameworkUrl: "static/Build/IBLMini-webgl.framework.js.gz",
        codeUrl: "static/Build/IBLMini-webgl.wasm.gz",
        streamingAssetsUrl: "StreamingAssets",
        companyName: "Daniel Birman @ UW",
        productName: "IBLMini",
        productVersion: "0.1.1",
        // matchWebGLToCanvasSize: false, // Uncomment this to separately control WebGL canvas render size and DOM element size.
        // devicePixelRatio: 1, // Uncomment this to override low DPI rendering on high DPI displays.
    }).then((unityInstance) => {
        window.myGameInstance = unityInstance;
    });
}



function setupSliders() {

    // Alpha slider

    // initSlider('sliderAlpha', window.params.alpha_range, window.params.alpha_lims);

    // onSliderChange('sliderAlpha', function (min, max) {
    //     window.params.alpha_range = [min, max];
    //     updateParamsData();
    // });

};



function selectPID(pid) {
    // UNITY callback
    document.getElementById('sessionSelector').value = pid;
    selectSession(pid);
}



async function selectSession(pid) {
    CTX.pid = pid;

    if (myGameInstance)
        myGameInstance.SendMessage("main", "HighlightProbe", pid);

    // Show the session details.
    var url = `/api/session/${pid}/details`;
    var r = await fetch(url);
    var details = await r.json();
    // Pop the cluster ids into a new variable
    var cluster_ids = details["cluster_ids"];
    var acronyms = details["acronyms"];
    var colors = details["colors"];
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
    for (var i = 0; i < cluster_ids.length; i++) {
        var cluster_id = cluster_ids[i];
        var acronym = acronyms[i];
        var color = colors[i];
        var opt = new Option(`${acronym} — #${cluster_id}`, cluster_id);
        var r = color[0];
        var g = color[1];
        var b = color[2];
        opt.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
        s.options[s.options.length] = opt;
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
    console.log(`select cluster #${cid}`);
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
        if (!pid) return;
        await selectSession(pid);
    }

    // Trial selector.
    document.getElementById('trialSelector').onchange = function (e) {
        var tid = e.target.value;
        if (!tid) return;
        selectTrial(CTX.pid, tid);
    }

    // Cluster selector.
    document.getElementById('clusterSelector').onchange = function (e) {
        var cid = e.target.value;
        if (!cid) return;
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
    loadUnity();
    setupPersistence();
    setupSliders();
    setupDropdowns();
    setupInputs();
    setupButtons();
};



$(document).ready(function () {
    load();
});
