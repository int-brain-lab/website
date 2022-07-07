
/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

const DEFAULT_PARAMS = {
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



function setupDropdowns() {

    document.getElementById('sessionSelector').onchange = async function (e) {
        var session = e.target.value;
        if(!session) return;
        var url = `/api/session/${session}/details`;

        var r = await fetch(url);
        document.getElementById('sessionDetails').innerHTML = await r.text();
    }
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
