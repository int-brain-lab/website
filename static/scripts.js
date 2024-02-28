
/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

// Passing data from Flask to Javascript

const ENABLE_UNITY = true;   // disable for debugging

const regexExp = /^[0-9A-F]{8}-[0-9A-F]{4}-[4][0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}$/i;
const SESSION_SEARCH_PLACEHOLDER = `examples: primary motor ; region:VISa6a ; pid:decc8d40, eid:f88d4dd4 ; lab:churchlandlab ; subject:NYU-`;
var unitySession = null; // unity instance for the session selector
var autoCompleteJS = null;
var isLoading = false;



/*************************************************************************************************/
/*  Utils                                                                                        */
/*************************************************************************************************/

function clamp(num, min, max) { return Math.min(Math.max(num, min), max); };



function binarySearch(arr, val) {
    let start = 0;
    let end = arr.length - 1;

    while (start <= end) {
        let mid = Math.floor((start + end) / 2);

        if (arr[mid] === val) {
            return mid;
        }

        if (val < arr[mid]) {
            end = mid - 1;
        } else {
            start = mid + 1;
        }
    }
    return -1;
};



function findStartIdx(sortedArray, target) {
    let start = 0,
        end = sortedArray.length - 1

    while (start <= end) {
        const mid = Math.floor((start + end) / 2)
        if (sortedArray[mid] < target) start = mid + 1
        else end = mid - 1
    }

    return start
};



function isEmpty(obj) {
    // https://stackoverflow.com/a/679937/1595060
    return Object.keys(obj).length === 0;
};



function isValidUUID(pid) {
    return regexExp.test(pid);
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



// // Display an array buffer.
// function show(arrbuf) {
//     const blob = new Blob([arrbuf]);
//     const url = URL.createObjectURL(blob);
//     const img = document.getElementById('imgRaster');
//     let w = img.offsetWidth;

//     var t0 = px2time(0);
//     var t1 = px2time(w);
//     var t = .5 * (t0 + t1);

//     Plotly.update('imgRaster', {}, {
//         "images[0].source": url,
//         "xaxis.ticktext": [t0.toFixed(3), t.toFixed(3), t1.toFixed(3)],
//     });

//     setLineOffset();
// };



function fillVerticalTable(json, elementID) {

    var table_data = `<table>`
    for (let key in json) {

        // NOTE: skip internal fields
        if (key.startsWith("_") || key == "ID") {
            continue;
        }

        var row = `<tr>
                    <th>${key}</th>
                    <td>${json[key]}</td>
                   </tr>`
        table_data += row
    }
    table_data += `</table>`

    document.getElementById(elementID).innerHTML = table_data
};



function fillHorizontalTable(json, elementID) {

    var table_data = `<table>`
    table_data += `<tr>`
    for (let key in json) {

        // NOTE: skip internal fields
        if (key.startsWith("_")) {
            continue;
        }

        var row = `<th>${key}</th>`
        table_data += row
    }
    table_data += `</tr>`
    table_data += `<tr>`
    for (let key in json) {

        // NOTE: skip internal fields
        if (key.startsWith("_")) {
            continue;
        }

        var row = `<td>${json[key]}</td>`
        table_data += row
    }
    table_data += `</tr>`
    table_data += `</table>`

    document.getElementById(elementID).innerHTML = table_data
};



function isOnMobile() {
    let check = false;
    (function (a) { if (/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(a) || /1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(a.substr(0, 4))) check = true; })(navigator.userAgent || navigator.vendor || window.opera);
    return check;
};



function showImage(id, url, unityCalled = false) {
    var loading = document.getElementById(id + "Loading");
    loading.style.visibility = "visible";

    var tmpImg = new Image();
    tmpImg.onload = function () {
        document.getElementById(id).src = tmpImg.src;
        loading.style.visibility = "hidden";
        delete tmpImg;
    }
    tmpImg.src = url;

    // document.getElementById(id).src = url;
};



function onlyUnique(value, index, self) {
    return self.indexOf(value) === index;
};



// This commented function does not take the count of each value in the result.
// function getUnique(arr) {
//     return arr.filter(onlyUnique);
// }

function getUnique(array) {
    var frequency = {}, value;

    // compute frequencies of each value
    for (var i = 0; i < array.length; i++) {
        value = array[i];
        if (value in frequency) {
            frequency[value]++;
        }
        else {
            frequency[value] = 1;
        }
    }

    // make array from the frequency object to de-duplicate
    var uniques = [];
    for (value in frequency) {
        uniques.push(value);
    }

    // sort the uniques array in descending order by frequency
    function compareFrequency(a, b) {
        return frequency[b] - frequency[a];
    }

    return uniques.sort(compareFrequency);
};


function filter_by_good(_, index) {
    return this[index] === true;
};


/*************************************************************************************************/
/*  Share button                                                                                 */
/*************************************************************************************************/

function getUrl() {
    let url = new URL(window.location);
    let params = url.searchParams;

    params.set("dset", CTX.dset);
    params.set("pid", CTX.pid);
    params.set("tid", CTX.tid);
    params.set("cid", CTX.cid);

    return url.toString();
}



function setupShare() {
    let share = document.getElementById("share");
    share.addEventListener("click", function (e) {
        navigator.clipboard.writeText(getUrl());
        share.children[0].innerHTML = "copied!";
        setTimeout(() => { share.children[0].innerHTML = "share"; }, 1500);
    });
};



/*************************************************************************************************/
/*  Unity callback functions                                                                     */
/*************************************************************************************************/

/// UNITY loaded callback event, update the current highlighted probe
function unityLoaded() {
    if (unitySession)
        unitySession.SendMessage("main", "HighlightProbe", CTX.pid);
}



// UNITY callback
function selectPID(pid) {
    selectSession(pid);
    autoCompleteJS.setQuery(pid);
};



/*************************************************************************************************/
/*  Setup session selection                                                                      */
/*************************************************************************************************/

function setupUnitySession() {
    // Disable Unity widget on smartphones.
    if (isOnMobile() || !ENABLE_UNITY) return;

    // Session selector widget.
    createUnityInstance(document.querySelector("#unity-canvas"), {
        dataUrl: "static/Build/IBLMini-webgl.data.gz",
        frameworkUrl: "static/Build/IBLMini-webgl.framework.js.gz",
        codeUrl: "static/Build/IBLMini-webgl.wasm.gz",
        streamingAssetsUrl: "StreamingAssets",
        companyName: "Daniel Birman @ UW",
        productName: "IBLMini",
        productVersion: "0.2.0",
        // matchWebGLToCanvasSize: false, // Uncomment this to separately control WebGL canvas render size and DOM element size.
        // devicePixelRatio: 1, // Uncomment this to override low DPI rendering on high DPI displays.
    }).then((unityInstance) => {
        window.unitySession = unityInstance;
    });
};



/*************************************************************************************************/
/*  Setup trial selection                                                                        */
/*************************************************************************************************/

function setupTrialDropdown(trial_ids, trial_id = -1) {
    // Set the trial selector.
    var s = document.getElementById('trialSelector');
    $('#trialSelector option').remove();
    var option = null;
    for (var i = 0; i < trial_ids.length; i++) {
        var tid = trial_ids[i]
        option = new Option(`trial #${tid.toString().padStart(3, "0")}`, tid);
        if (((trial_id == -1) && (i == 0)) || (tid == trial_id))
            option.selected = true;
        s.options[s.options.length] = option;
    }
};



function setupTrialCallback() {
    // Trial selector.
    document.getElementById('trialSelector').onchange = function (e) {
        var tid = e.target.value;
        if (!tid) return;
        selectTrial(CTX.pid, tid);
    }

    document.getElementById('trialPlot').onclick = clickTrial;
}



/*************************************************************************************************/
/*  Setup cluster selection                                                                     */
/*************************************************************************************************/

function setupClusterDropdown(cluster_ids, acronyms, colors, cluster_id = -1) {
    // Set the cluster selector.
    var s = document.getElementById('clusterSelector');
    $('#clusterSelector option').remove();
    for (var i = 0; i < cluster_ids.length; i++) {
        var cid = cluster_ids[i];
        var acronym = acronyms[i];
        var color = colors[i];
        option = new Option(`${acronym} — #${cid}`, cid);
        var r = color[0];
        var g = color[1];
        var b = color[2];
        option.style.backgroundColor = `rgb(${r}, ${g}, ${b})`;
        if (((cluster_id == -1) && (i == 0)) || (cid == cluster_id))
            option.selected = true;
        s.options[s.options.length] = option;
    }
}



function setupClusterCallback() {
    // Cluster selector.
    document.getElementById('clusterSelector').onchange = function (e) {
        var cid = e.target.value;
        if (!cid) return;
        selectCluster(CTX.pid, cid);
    }
};



function setupClusterClick() {
    const canvas = document.getElementById('clusterPlot');
    canvas.addEventListener('click', function (e) {
        // console.log("click cluster", e);
        onClusterClick(canvas, e);
    }, true);
};



/*************************************************************************************************/
/*  Session Filtering                                                                            */
/*************************************************************************************************/

function contains(query_, arr, exact = false) {
    if (!exact)
        return arr.some(x => x.includes(query_));
    else
        return arr.some(x => x == query_);
}

function filterQuery(query_, Lab, Subject, pid, eid, acronyms, regions, _good_ids) {
    Lab = Lab.toLowerCase();
    Subject = Subject.toLowerCase();
    pid = pid.toLowerCase();
    eid = eid.toLowerCase();

    // For a valid UUID: return yes if the query matches the current session's pid or eid.
    if (isValidUUID(query_)) {
        return pid.includes(query_) || eid.includes(query_);
    }

    // By default (no colon ":"), search the region names.
    if (!query_.includes(":")) {
        return regions.some(name => name.includes(query_));
    }

    // Otherwise, we assume the query is of the form:
    [field, value] = query_.split(":");

    // if (field == "region") return contains(query_, regions);
    if (field == "region") return contains(value, acronyms, exact = true);
    if (field == "pid") return pid.includes(value);
    if (field == "eid") return eid.includes(value);
    if (field == "lab") return Lab.includes(value);
    if (field == "subject") return Subject.includes(value);

    return false;
};



function getUniqueAcronyms(_acronyms, _good_ids) {
    // Remove duplicates in acronyms.
    _acronyms = _acronyms.map(a => a.toLowerCase());

    // Keep good acronyms.
    let acronyms = _acronyms.filter(filter_by_good, _good_ids);

    return Array.from(new Set(acronyms));
}

function acronymsToNames(acronyms) {
    return acronyms.map(acronym => ALLEN_REGIONS[acronym].toLowerCase());
}

function getSessionList() {

    let sessions = FLASK_CTX.SESSIONS;

    let out = sessions.filter(function (
        { Lab, Subject, pid, eid, _acronyms, _good_ids, dset_bwm, dset_rs }) {

        // Is the query a UUID?
        if (isValidUUID(query_)) {
            // If 1 session is already selected, show all of them.
            if (query_ == CTX.pid) return true;
            // Otherwise, show the corresponding session.
            return query_ == pid;
        }

        // Region acronyms and names.
        let acronyms = getUniqueAcronyms(_acronyms, _good_ids);
        let regions = acronymsToNames(acronyms);

        // Filter on each term (space-separated).
        var res = true;
        for (let q of query_.split(/(\s+)/)) {
            res &= filterQuery(q, Lab, Subject, pid, eid, acronyms, regions, _good_ids);
        }

        // Dataset selection
        if (CTX.dset == 'bwm')
            res &= dset_bwm;
        if (CTX.dset == 'rs')
            res &= dset_rs;

        return res;
    });

    // Update the mini brain viewer with the kept sessions.
    let pids = out.map(({ pid }) => pid);
    miniBrainActivatePIDs(pids);

    return out;
}



/*************************************************************************************************/
/*  Session selection                                                                            */
/*************************************************************************************************/

function onDatasetChanged(ev) {
    let dset = null;
    if (ev.target.id == "dset-1") dset = "bwm";
    else if (ev.target.id == "dset-2") dset = "rs";
    else { console.log("unknown dset name " + dset); return; }
    CTX.dset = dset;
    autoCompleteJS.refresh();
}

function setupDataset() {
    document.getElementById('dset-1').onclick = onDatasetChanged;
    document.getElementById('dset-2').onclick = onDatasetChanged;

    if (CTX.dset == "bwm") document.getElementById('dset-1').checked = true;
    if (CTX.dset == "rs") document.getElementById('dset-2').checked = true;
}

function loadAutoComplete() {
    autoCompleteJS = autocomplete({
        container: '#sessionSelector',
        placeholder: SESSION_SEARCH_PLACEHOLDER,
        openOnFocus: true,
        initialState: { query: CTX.pid },
        onStateChange({ state }) {
            var pid = state.query;

            // We only proceed if a new valid UUID has been selected.
            if (state.isOpen) return;
            if (!pid) return;
            if (pid == CTX.pid) return;
            if (!isValidUUID(pid)) return;
            // CTX.pid = pid;

            selectSession(pid);
        },
        getSources({ query }) {
            query_ = query.toLowerCase();
            return [
                {
                    sourceId: 'sessions',
                    getItemInputValue: ({ item }) => item.pid,
                    getItems() {
                        return getSessionList();
                    },
                    templates: {
                        item({ item, html }) {
                            var good_idx = item['_good_ids'];
                            var acronyms = item["_acronyms"].filter(filter_by_good, good_idx);

                            acronyms = getUnique(acronyms);
                            acronyms = acronyms.filter(item => item !== "void");

                            var n = acronyms.length;
                            var M = 5;
                            acronyms = acronyms.slice(0, M);
                            acronyms = acronyms.join(", ");
                            if (n > M)
                                acronyms += "...";
                            return html`
                            <div class="item-container">
                            <div class="item item-lab">${item.Lab}</div>
                            <div class="item item-subject">${item.Subject}</div>
                            <div class="item item-date">${item['Recording date']}</div>
                            <div class="item item-acronyms">${acronyms}</div>
                            <div class="item item-ID">${item.pid}</div>
                            </div>`
                                ;
                        },
                        noResults() {
                            return 'No results.';
                        },
                    },
                },
            ];
        },
    });
};



function updateSessionPlot(pid) {
    showImage('sessionPlot', `/api/session/${pid}/session_plot`);
};



function updateBehaviourPlot(pid) {
    showImage('behaviourPlot', `/api/session/${pid}/behaviour_plot`);
};



async function selectSession(pid) {
    if (isLoading) return;
    if (!pid) return;
    isLoading = true;
    console.log("select session " + pid);

    if (unitySession)
        unitySession.SendMessage("main", "HighlightProbe", pid);

    // Show the session details.
    var url = `/api/session/${pid}/details`;
    var r = await fetch(url);
    var details = await r.json();

    // Pop the cluster ids into a new variable

    // NOTE: these fields start with a leading _ so will be ignored by tablefromjson
    // which controls which fields are displayed in the session details box.
    var trial_ids = details['_trial_ids']
    var good_idx = details["_good_ids"];

    var cluster_ids = details["_cluster_ids"].filter(filter_by_good, good_idx);
    var acronyms = details["_acronyms"].filter(filter_by_good, good_idx);
    var colors = details["_colors"].filter(filter_by_good, good_idx);

    CTX.dur = details["_duration"];
    CTX.trial_ids = trial_ids;
    CTX.trial_onsets = details["_trial_onsets"];
    CTX.trial_offsets = details["_trial_offsets"];

    // Make table with session details.
    fillVerticalTable(details, 'sessionDetails')

    // Show the session overview plot.
    updateSessionPlot(pid);

    // Show the behaviour overview plot.
    updateBehaviourPlot(pid);

    // Show the trial plot.
    updateTrialPlot(pid);

    // Setup the trial selector.
    var trial_id = trial_ids[0];
    if (CTX.pid == pid && CTX.tid)
        trial_id = CTX.tid;
    setupTrialDropdown(trial_ids, trial_id);

    // Setup the cluster selector.
    setupClusterDropdown(cluster_ids, acronyms, colors, cluster_id = CTX.cid);

    // Update the other plots.
    selectTrial(pid, trial_id);

    // Need to make sure first cluster is a good one, otherwise get error
    var cluster_id = null;
    if ((CTX.pid == pid) && (CTX.cid && CTX.cid >= 0))
        cluster_id = CTX.cid;
    else if (cluster_ids)
        cluster_id = cluster_ids[0];
    if ((cluster_id !== null) && (cluster_ids.includes(cluster_id)))
        selectCluster(pid, cluster_id);

    CTX.pid = pid;
    isLoading = false;
};



/*************************************************************************************************/
/*  Unity mini brain                                                                             */
/*************************************************************************************************/

function miniBrainActivatePIDs(pidList) {
    // takes as input a list of PIDs and activates these
    if (unitySession) {
        unitySession.SendMessage("main", "DeactivateAllProbes");
        for (pid of pidList) {
            unitySession.SendMessage("main", "ActivateProbe", pid);
        }
    }
}



/*************************************************************************************************/
/*  Trials                                                                                       */
/*************************************************************************************************/

function updateTrialPlot(pid) {
    showImage('trialEventPlot', `/api/session/${pid}/trial_event_plot`);
};



function clickTrial(event) {
    // png is 1200x500
    // left panel:  x: 80-383,   y: 60-420
    // right panel: x: 399-1004, y: 60-420
    var img = document.getElementById("trialPlot");

    var w = img.width;
    var h = img.height;
    var c = w / 1200.0;
    var x0 = 80 * c;
    var x1 = 383 * c;
    var y0 = 60 * c;
    var y1 = 420 * c;

    var rect = img.getBoundingClientRect();
    var x = (event.clientX - rect.left) - x0;
    var y = Math.abs((event.clientY - rect.bottom)) - y0;

    // Limit the click to the left panel.
    if (x >= x1 - x0) return;

    x = x / (x1 - x0);
    y = y / (y1 - y0);

    x = clamp(x, 0, 1);
    y = clamp(y, 0, 1);

    var t = x * CTX.dur;
    console.log("select trial at time " + t);

    var tidx = findStartIdx(CTX.trial_onsets, t);
    tidx = clamp(tidx, 0, CTX.trial_ids.length - 1);
    var tid = CTX.trial_ids[tidx];
    $('#trialSelector').val(tid);
    selectTrial(CTX.pid, tid);
};



async function selectTrial(pid, tid, unityCalled = false) {
    CTX.tid = tid;

    // Show the trial raster plot.
    var url = `/api/session/${pid}/trial_plot/${tid}`;
    showImage('trialPlot', url, unityCalled);

    // Show information about trials in table
    var url = `/api/session/${pid}/trial_details/${tid}`;
    var r = await fetch(url).then();
    var details = await r.json();

    // Fill the trial details table.
    fillHorizontalTable(details, 'trialDetails')
};



function arrowButton(name, dir) {
    var select = document.getElementById(name);
    if (dir > 0)
        select.selectedIndex++;
    else
        select.selectedIndex--;

    if (select.selectedIndex < 0)
        select.selectedIndex = 0;

    select.dispatchEvent(new Event('change'));

};



/*************************************************************************************************/
/*  Cluster legends                                                                              */
/*************************************************************************************************/

function addPanelLetter(legend_name, fig, letter, coords, legend) {
    let plot_legend = document.getElementById(legend_name);
    const div = document.createElement("div");
    div.appendChild(document.createTextNode(letter));
    div.classList.add('panel-letter');
    fig.parentNode.appendChild(div);

    let [xmin, ymin, xmax, ymax] = coords;
    div.style.top = (ymin * 100 - 4) + "%";
    div.style.left = (xmin * 100 - 1) + "%";
    div.style.height = ((ymax - ymin) * 100) + "%";
    div.style.width = ((xmax - xmin) * 100) + "%";

    div.addEventListener("mouseover", function (e) {
        plot_legend.innerHTML = `<strong>Legend of panel ${letter}</strong>: ${legend}`;
    });

    // HACK: ensure the click events is propagated from the transparent overlayed div containing
    // the figure letter, to the figure below. Used for the cluster click in figure 5.
    div.addEventListener("click", function (e) {
        let elements = document.elementsFromPoint(e.clientX, e.clientY);
        var ev = new MouseEvent('click', {
            'view': window,
            'bubbles': true,
            'cancelable': true,
            'clientX': e.clientX,
            'clientY': e.clientY
        });

        elements[1].dispatchEvent(ev);
    }, true);
}



function setupLegends(plot_id, legend_id, key) {
    let plot = document.getElementById(plot_id);

    let legends = FLASK_CTX.LEGENDS;
    for (let letter in legends[key]) {
        let panel = legends[key][letter];
        addPanelLetter(legend_id, plot, letter, panel["coords"], panel["legend"]);
    }

}



function setupAllLegends() {
    setupLegends('sessionPlot', 'sessionPlotLegend', 'figure1');
    setupLegends('behaviourPlot', 'behaviourPlotLegend', 'figure2');
    setupLegends('trialPlot', 'trialPlotLegend', 'figure3');
    setupLegends('trialEventPlot', 'trialEventPlotLegend', 'figure4');
    setupLegends('clusterPlot', 'clusterPlotLegend', 'figure5');
}



/*************************************************************************************************/
/*  Cluster selection                                                                            */
/*************************************************************************************************/

async function onClusterClick(canvas, event) {
    const rect = canvas.getBoundingClientRect()
    const x = (event.clientX - rect.left) / rect.width
    const y = Math.abs((event.clientY - rect.bottom)) / rect.height
    var url = `/api/session/${CTX.pid}/cluster_plot_from_xy/${CTX.cid}/${x.toFixed(3)}_${y.toFixed(3)}/`;
    var r = await fetch(url);
    var details = await r.json();

    var new_cluster_idx = details["cluster_idx"];

    if (new_cluster_idx !== CTX.cid)
        selectCluster(CTX.pid, details["cluster_idx"]);
    var select = document.getElementById(`clusterSelector`);
    select.selectedIndex = details["idx"];
};



async function selectCluster(pid, cid) {
    console.log(`select cluster #${cid}`);
    CTX.cid = cid;

    var url = `/api/session/${pid}/cluster_plot/${cid}`;
    showImage('clusterPlot', url);

    // Show information about cluster in table
    var url = `/api/session/${pid}/cluster_details/${cid}`;
    var r = await fetch(url).then();
    var details = await r.json();

    fillHorizontalTable(details, 'clusterDetails')

};



/*************************************************************************************************/
/*  Entry point                                                                                  */
/*************************************************************************************************/

function load() {
    setupShare();

    setupDataset();
    loadAutoComplete();

    setupUnitySession();

    setupAllLegends();
    setupClusterClick()
    setupTrialCallback();
    setupClusterCallback();

    // Initial selection.
    selectSession(CTX.pid);
};



$(document).ready(function () {
    load();
});
