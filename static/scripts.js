
/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

// Passing data from Flask to Javascript

const ENABLE_UNITY = false;   // disable for debugging

const regexExp = /^[0-9A-F]{8}-[0-9A-F]{4}-[4][0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}$/i;
const SESSION_SEARCH_PLACEHOLDER = `examples: region:VISa6a ; eid:f88d4dd4 ; lab:churchlandlab ; subject:NYU-`;
var unitySession = null; // unity instance for the session selector
var unityTrial = null; // unity instance for the trial viewer
var autoCompleteJS = null;
var isLoading = false;
const ALLEN_REGIONS = {
    "void": "void", "root": "root (left)", "grey": "basic cell groups and regions (left)", "ch": "cerebrum (left)", "ctx": "cerebral cortex (left)", "ctxpl": "cortical plate (left)", "isocortex": "isocortex (left)", "frp": "frontal pole cerebral cortex (left)", "frp1": "frontal pole layer 1 (left)", "frp2/3": "frontal pole layer 2/3 (left)", "frp5": "frontal pole layer 5 (left)", "frp6a": "frontal pole layer 6a (left)", "frp6b": "frontal pole layer 6b (left)", "mo": "somatomotor areas (left)", "mo1": "somatomotor areas layer 1 (left)", "mo2/3": "somatomotor areas layer 2/3 (left)", "mo5": "somatomotor areas layer 5 (left)", "mo6a": "somatomotor areas layer 6a (left)", "mo6b": "somatomotor areas layer 6b (left)", "mop": "primary motor area (left)", "mop1": "primary motor area layer 1 (left)", "mop2/3": "primary motor area layer 2/3 (left)", "mop5": "primary motor area layer 5 (left)", "mop6a": "primary motor area layer 6a (left)", "mop6b": "primary motor area layer 6b (left)", "mos": "secondary motor area (left)", "mos1": "secondary motor area layer 1 (left)", "mos2/3": "secondary motor area layer 2/3 (left)", "mos5": "secondary motor area layer 5 (left)", "mos6a": "secondary motor area layer 6a (left)", "mos6b": "secondary motor area layer 6b (left)", "ss": "somatosensory areas (left)", "ss1": "somatosensory areas layer 1 (left)", "ss2/3": "somatosensory areas layer 2/3 (left)", "ss4": "somatosensory areas layer 4 (left)", "ss5": "somatosensory areas layer 5 (left)", "ss6a": "somatosensory areas layer 6a (left)", "ss6b": "somatosensory areas layer 6b (left)", "ssp": "primary somatosensory area (left)", "ssp1": "primary somatosensory area layer 1 (left)", "ssp2/3": "primary somatosensory area layer 2/3 (left)", "ssp4": "primary somatosensory area layer 4 (left)", "ssp5": "primary somatosensory area layer 5 (left)", "ssp6a": "primary somatosensory area layer 6a (left)", "ssp6b": "primary somatosensory area layer 6b (left)", "ssp-n": "primary somatosensory area nose (left)", "ssp-n1": "primary somatosensory area nose layer 1 (left)", "ssp-n2/3": "primary somatosensory area nose layer 2/3 (left)", "ssp-n4": "primary somatosensory area nose layer 4 (left)", "ssp-n5": "primary somatosensory area nose layer 5 (left)", "ssp-n6a": "primary somatosensory area nose layer 6a (left)", "ssp-n6b": "primary somatosensory area nose layer 6b (left)", "ssp-bfd": "primary somatosensory area barrel field (left)", "ssp-bfd1": "primary somatosensory area barrel field layer 1 (left)", "ssp-bfd2/3": "primary somatosensory area barrel field layer 2/3 (left)", "ssp-bfd4": "primary somatosensory area barrel field layer 4 (left)", "ssp-bfd5": "primary somatosensory area barrel field layer 5 (left)", "ssp-bfd6a": "primary somatosensory area barrel field layer 6a (left)", "ssp-bfd6b": "primary somatosensory area barrel field layer 6b (left)", "visrll": "rostrolateral lateral visual area (left)", "visrll1": "rostrolateral lateral visual area layer 1 (left)", "visrll2/3": "rostrolateral lateral visual area layer 2/3 (left)", "visrll4": "rostrolateral lateral visual area layer 4 (left)", "visrll5": "rostrolateral lateral visual arealayer 5 (left)", "visrll6a": "rostrolateral lateral visual area layer 6a (left)", "visrll6b": "rostrolateral lateral visual area layer 6b (left)", "ssp-ll": "primary somatosensory area lower limb (left)", "ssp-ll1": "primary somatosensory area lower limb layer 1 (left)", "ssp-ll2/3": "primary somatosensory area lower limb layer 2/3 (left)", "ssp-ll4": "primary somatosensory area lower limb layer 4 (left)", "ssp-ll5": "primary somatosensory area lower limb layer 5 (left)", "ssp-ll6a": "primary somatosensory area lower limb layer 6a (left)", "ssp-ll6b": "primary somatosensory area lower limb layer 6b (left)", "ssp-m": "primary somatosensory area mouth (left)", "ssp-m1": "primary somatosensory area mouth layer 1 (left)", "ssp-m2/3": "primary somatosensory area mouth layer 2/3 (left)", "ssp-m4": "primary somatosensory area mouth layer 4 (left)", "ssp-m5": "primary somatosensory area mouth layer 5 (left)", "ssp-m6a": "primary somatosensory area mouth layer 6a (left)", "ssp-m6b": "primary somatosensory area mouth layer 6b (left)", "ssp-ul": "primary somatosensory area upper limb (left)", "ssp-ul1": "primary somatosensory area upper limb layer 1 (left)", "ssp-ul2/3": "primary somatosensory area upper limb layer 2/3 (left)", "ssp-ul4": "primary somatosensory area upper limb layer 4 (left)", "ssp-ul5": "primary somatosensory area upper limb layer 5 (left)", "ssp-ul6a": "primary somatosensory area upper limb layer 6a (left)", "ssp-ul6b": "primary somatosensory area upper limb layer 6b (left)", "ssp-tr": "primary somatosensory area trunk (left)", "ssp-tr1": "primary somatosensory area trunk layer 1 (left)", "ssp-tr2/3": "primary somatosensory area trunk layer 2/3 (left)", "ssp-tr4": "primary somatosensory area trunk layer 4 (left)", "ssp-tr5": "primary somatosensory area trunk layer 5 (left)", "ssp-tr6a": "primary somatosensory area trunk layer 6a (left)", "ssp-tr6b": "primary somatosensory area trunk layer 6b (left)", "ssp-un": "primary somatosensory area unassigned (left)", "ssp-un1": "primary somatosensory area unassigned layer 1 (left)", "ssp-un2/3": "primary somatosensory area unassigned layer 2/3 (left)", "ssp-un4": "primary somatosensory area unassigned layer 4 (left)", "ssp-un5": "primary somatosensory area unassigned layer 5 (left)", "ssp-un6a": "primary somatosensory area unassigned layer 6a (left)", "ssp-un6b": "primary somatosensory area unassigned layer 6b (left)", "sss": "supplemental somatosensory area (left)", "sss1": "supplemental somatosensory area layer 1 (left)", "sss2/3": "supplemental somatosensory area layer 2/3 (left)", "sss4": "supplemental somatosensory area layer 4 (left)", "sss5": "supplemental somatosensory area layer 5 (left)", "sss6a": "supplemental somatosensory area layer 6a (left)", "sss6b": "supplemental somatosensory area layer 6b (left)", "gu": "gustatory areas (left)", "gu1": "gustatory areas layer 1 (left)", "gu2/3": "gustatory areas layer 2/3 (left)", "gu4": "gustatory areas layer 4 (left)", "gu5": "gustatory areas layer 5 (left)", "gu6a": "gustatory areas layer 6a (left)", "gu6b": "gustatory areas layer 6b (left)", "visc": "visceral area (left)", "visc1": "visceral area layer 1 (left)", "visc2/3": "visceral area layer 2/3 (left)", "visc4": "visceral area layer 4 (left)", "visc5": "visceral area layer 5 (left)", "visc6a": "visceral area layer 6a (left)", "visc6b": "visceral area layer 6b (left)", "aud": "auditory areas (left)", "audd": "dorsal auditory area (left)", "audd1": "dorsal auditory area layer 1 (left)", "audd2/3": "dorsal auditory area layer 2/3 (left)", "audd4": "dorsal auditory area layer 4 (left)", "audd5": "dorsal auditory area layer 5 (left)", "audd6a": "dorsal auditory area layer 6a (left)", "audd6b": "dorsal auditory area layer 6b (left)", "vislla": "laterolateral anterior visual area (left)", "vislla1": "laterolateral anterior visual area layer 1 (left)", "vislla2/3": "laterolateral anterior visual area layer 2/3 (left)", "vislla4": "laterolateral anterior visual area layer 4 (left)", "vislla5": "laterolateral anterior visual arealayer 5 (left)", "vislla6a": "laterolateral anterior visual area layer 6a (left)", "vislla6b": "laterolateral anterior visual area layer 6b (left)", "audp": "primary auditory area (left)", "audp1": "primary auditory area layer 1 (left)", "audp2/3": "primary auditory area layer 2/3 (left)", "audp4": "primary auditory area layer 4 (left)", "audp5": "primary auditory area layer 5 (left)", "audp6a": "primary auditory area layer 6a (left)", "audp6b": "primary auditory area layer 6b (left)", "audpo": "posterior auditory area (left)", "audpo1": "posterior auditory area layer 1 (left)", "audpo2/3": "posterior auditory area layer 2/3 (left)", "audpo4": "posterior auditory area layer 4 (left)", "audpo5": "posterior auditory area layer 5 (left)", "audpo6a": "posterior auditory area layer 6a (left)", "audpo6b": "posterior auditory area layer 6b (left)", "audv": "ventral auditory area (left)", "audv1": "ventral auditory area layer 1 (left)", "audv2/3": "ventral auditory area layer 2/3 (left)", "audv4": "ventral auditory area layer 4 (left)", "audv5": "ventral auditory area layer 5 (left)", "audv6a": "ventral auditory area layer 6a (left)", "audv6b": "ventral auditory area layer 6b (left)", "vis": "visual areas (left)", "vis1": "visual areas layer 1 (left)", "vis2/3": "visual areas layer 2/3 (left)", "vis4": "visual areas layer 4 (left)", "vis5": "visual areas layer 5 (left)", "vis6a": "visual areas layer 6a (left)", "vis6b": "visual areas layer 6b (left)", "visal": "anterolateral visual area (left)", "visal1": "anterolateral visual area layer 1 (left)", "visal2/3": "anterolateral visual area layer 2/3 (left)", "visal4": "anterolateral visual area layer 4 (left)", "visal5": "anterolateral visual area layer 5 (left)", "visal6a": "anterolateral visual area layer 6a (left)", "visal6b": "anterolateral visual area layer 6b (left)", "visam": "anteromedial visual area (left)", "visam1": "anteromedial visual area layer 1 (left)", "visam2/3": "anteromedial visual area layer 2/3 (left)", "visam4": "anteromedial visual area layer 4 (left)", "visam5": "anteromedial visual area layer 5 (left)", "visam6a": "anteromedial visual area layer 6a (left)", "visam6b": "anteromedial visual area layer 6b (left)", "visl": "lateral visual area (left)", "visl1": "lateral visual area layer 1 (left)", "visl2/3": "lateral visual area layer 2/3 (left)", "visl4": "lateral visual area layer 4 (left)", "visl5": "lateral visual area layer 5 (left)", "visl6a": "lateral visual area layer 6a (left)", "visl6b": "lateral visual area layer 6b (left)", "visp": "primary visual area (left)", "visp1": "primary visual area layer 1 (left)", "visp2/3": "primary visual area layer 2/3 (left)", "visp4": "primary visual area layer 4 (left)", "visp5": "primary visual area layer 5 (left)", "visp6a": "primary visual area layer 6a (left)", "visp6b": "primary visual area layer 6b (left)", "vispl": "posterolateral visual area (left)", "vispl1": "posterolateral visual area layer 1 (left)", "vispl2/3": "posterolateral visual area layer 2/3 (left)", "vispl4": "posterolateral visual area layer 4 (left)", "vispl5": "posterolateral visual area layer 5 (left)", "vispl6a": "posterolateral visual area layer 6a (left)", "vispl6b": "posterolateral visual area layer 6b (left)", "vispm": "posteromedial visual area (left)", "vispm1": "posteromedial visual area layer 1 (left)", "vispm2/3": "posteromedial visual area layer 2/3 (left)", "vispm4": "posteromedial visual area layer 4 (left)", "vispm5": "posteromedial visual area layer 5 (left)", "vispm6a": "posteromedial visual area layer 6a (left)", "vispm6b": "posteromedial visual area layer 6b (left)", "visli": "laterointermediate area (left)", "visli1": "laterointermediate area layer 1 (left)", "visli2/3": "laterointermediate area layer 2/3 (left)", "visli4": "laterointermediate area layer 4 (left)", "visli5": "laterointermediate area layer 5 (left)", "visli6a": "laterointermediate area layer 6a (left)", "visli6b": "laterointermediate area layer 6b (left)", "vispor": "postrhinal area (left)", "vispor1": "postrhinal area layer 1 (left)", "vispor2/3": "postrhinal area layer 2/3 (left)", "vispor4": "postrhinal area layer 4 (left)", "vispor5": "postrhinal area layer 5 (left)", "vispor6a": "postrhinal area layer 6a (left)", "vispor6b": "postrhinal area layer 6b (left)", "aca": "anterior cingulate area (left)", "aca1": "anterior cingulate area layer 1 (left)", "aca2/3": "anterior cingulate area layer 2/3 (left)", "aca5": "anterior cingulate area layer 5 (left)", "aca6a": "anterior cingulate area layer 6a (left)", "aca6b": "anterior cingulate area layer 6b (left)", "acad": "anterior cingulate area dorsal part (left)", "acad1": "anterior cingulate area dorsal part layer 1 (left)", "acad2/3": "anterior cingulate area dorsal part layer 2/3 (left)", "acad5": "anterior cingulate area dorsal part layer 5 (left)", "acad6a": "anterior cingulate area dorsal part layer 6a (left)", "acad6b": "anterior cingulate area dorsal part layer 6b (left)", "acav": "anterior cingulate area ventral part (left)", "acav1": "anterior cingulate area ventral part layer 1 (left)", "acav2/3": "anterior cingulate area ventral part layer 2/3 (left)", "acav5": "anterior cingulate area ventral part layer 5 (left)", "acav6a": "anterior cingulate area ventral part 6a (left)", "acav6b": "anterior cingulate area ventral part 6b (left)", "pl": "prelimbic area (left)", "pl1": "prelimbic area layer 1 (left)", "pl2": "prelimbic area layer 2 (left)", "pl2/3": "prelimbic area layer 2/3 (left)", "pl5": "prelimbic area layer 5 (left)", "pl6a": "prelimbic area layer 6a (left)", "pl6b": "prelimbic area layer 6b (left)", "ila": "infralimbic area (left)", "ila1": "infralimbic area layer 1 (left)", "ila2": "infralimbic area layer 2 (left)", "ila2/3": "infralimbic area layer 2/3 (left)", "ila5": "infralimbic area layer 5 (left)", "ila6a": "infralimbic area layer 6a (left)", "ila6b": "infralimbic area layer 6b (left)", "orb": "orbital area (left)", "orb1": "orbital area layer 1 (left)", "orb2/3": "orbital area layer 2/3 (left)", "orb5": "orbital area layer 5 (left)", "orb6a": "orbital area layer 6a (left)", "orb6b": "orbital area layer 6b (left)", "orbl": "orbital area lateral part (left)", "orbl1": "orbital area lateral part layer 1 (left)", "orbl2/3": "orbital area lateral part layer 2/3 (left)", "orbl5": "orbital area lateral part layer 5 (left)", "orbl6a": "orbital area lateral part layer 6a (left)", "orbl6b": "orbital area lateral part layer 6b (left)", "orbm": "orbital area medial part (left)", "orbm1": "orbital area medial part layer 1 (left)", "orbm2": "orbital area medial part layer 2 (left)", "orbm2/3": "orbital area medial part layer 2/3 (left)", "orbm5": "orbital area medial part layer 5 (left)", "orbm6a": "orbital area medial part layer 6a (left)", "orbm6b": "orbital area medial part layer 6b (left)", "orbv": "orbital area ventral part (left)", "orbvl": "orbital area ventrolateral part (left)", "orbvl1": "orbital area ventrolateral part layer 1 (left)", "orbvl2/3": "orbital area ventrolateral part layer 2/3 (left)", "orbvl5": "orbital area ventrolateral part layer 5 (left)", "orbvl6a": "orbital area ventrolateral part layer 6a (left)", "orbvl6b": "orbital area ventrolateral part layer 6b (left)", "ai": "agranular insular area (left)", "aid": "agranular insular area dorsal part (left)", "aid1": "agranular insular area dorsal part layer 1 (left)", "aid2/3": "agranular insular area dorsal part layer 2/3 (left)", "aid5": "agranular insular area dorsal part layer 5 (left)", "aid6a": "agranular insular area dorsal part layer 6a (left)", "aid6b": "agranular insular area dorsal part layer 6b (left)", "aip": "agranular insular area posterior part (left)", "aip1": "agranular insular area posterior part layer 1 (left)", "aip2/3": "agranular insular area posterior part layer 2/3 (left)", "aip5": "agranular insular area posterior part layer 5 (left)", "aip6a": "agranular insular area posterior part layer 6a (left)", "aip6b": "agranular insular area posterior part layer 6b (left)", "aiv": "agranular insular area ventral part (left)", "aiv1": "agranular insular area ventral part layer 1 (left)", "aiv2/3": "agranular insular area ventral part layer 2/3 (left)", "aiv5": "agranular insular area ventral part layer 5 (left)", "aiv6a": "agranular insular area ventral part layer 6a (left)", "aiv6b": "agranular insular area ventral part layer 6b (left)", "rsp": "retrosplenial area (left)", "rspagl": "retrosplenial area lateral agranular part (left)", "rspagl1": "retrosplenial area lateral agranular part layer 1 (left)", "rspagl2/3": "retrosplenial area lateral agranular part layer 2/3 (left)", "rspagl5": "retrosplenial area lateral agranular part layer 5 (left)", "rspagl6a": "retrosplenial area lateral agranular part layer 6a (left)", "rspagl6b": "retrosplenial area lateral agranular part layer 6b (left)", "vismma": "mediomedial anterior visual area (left)", "vismma1": "mediomedial anterior visual area layer 1 (left)", "vismma2/3": "mediomedial anterior visual area layer 2/3 (left)", "vismma4": "mediomedial anterior visual area layer 4 (left)", "vismma5": "mediomedial anterior visual arealayer 5 (left)", "vismma6a": "mediomedial anterior visual area layer 6a (left)", "vismma6b": "mediomedial anterior visual area layer 6b (left)", "vismmp": "mediomedial posterior visual area (left)", "vismmp1": "mediomedial posterior visual area layer 1 (left)", "vismmp2/3": "mediomedial posterior visual area layer 2/3 (left)", "vismmp4": "mediomedial posterior visual area layer 4 (left)", "vismmp5": "mediomedial posterior visual arealayer 5 (left)", "vismmp6a": "mediomedial posterior visual area layer 6a (left)", "vismmp6b": "mediomedial posterior visual area layer 6b (left)", "vism": "medial visual area (left)", "vism1": "medial visual area layer 1 (left)", "vism2/3": "medial visual area layer 2/3 (left)", "vism4": "medial visual area layer 4 (left)", "vism5": "medial visual arealayer 5 (left)", "vism6a": "medial visual area layer 6a (left)", "vism6b": "medial visual area layer 6b (left)", "rspd": "retrosplenial area dorsal part (left)", "rspd1": "retrosplenial area dorsal part layer 1 (left)", "rspd2/3": "retrosplenial area dorsal part layer 2/3 (left)", "rspd4": "retrosplenial area dorsal part layer 4 (left)", "rspd5": "retrosplenial area dorsal part layer 5 (left)", "rspd6a": "retrosplenial area dorsal part layer 6a (left)", "rspd6b": "retrosplenial area dorsal part layer 6b (left)", "rspv": "retrosplenial area ventral part (left)", "rspv1": "retrosplenial area ventral part layer 1 (left)", "rspv2": "retrosplenial area ventral part layer 2 (left)", "rspv2/3": "retrosplenial area ventral part layer 2/3 (left)", "rspv5": "retrosplenial area ventral part layer 5 (left)", "rspv6a": "retrosplenial area ventral part layer 6a (left)", "rspv6b": "retrosplenial area ventral part layer 6b (left)", "ptlp": "posterior parietal association areas (left)", "ptlp1": "posterior parietal association areas layer 1 (left)", "ptlp2/3": "posterior parietal association areas layer 2/3 (left)", "ptlp4": "posterior parietal association areas layer 4 (left)", "ptlp5": "posterior parietal association areas layer 5 (left)", "ptlp6a": "posterior parietal association areas layer 6a (left)", "ptlp6b": "posterior parietal association areas layer 6b (left)", "visa": "anterior area (left)", "visa1": "anterior area layer 1 (left)", "visa2/3": "anterior area layer 2/3 (left)", "visa4": "anterior area layer 4 (left)", "visa5": "anterior area layer 5 (left)", "visa6a": "anterior area layer 6a (left)", "visa6b": "anterior area layer 6b (left)", "visrl": "rostrolateral visual area (left)", "visrl1": "rostrolateral area layer 1 (left)", "visrl2/3": "rostrolateral area layer 2/3 (left)", "visrl4": "rostrolateral area layer 4 (left)", "visrl5": "rostrolateral area layer 5 (left)", "visrl6a": "rostrolateral area layer 6a (left)", "visrl6b": "rostrolateral area layer 6b (left)", "tea": "temporal association areas (left)", "tea1": "temporal association areas layer 1 (left)", "tea2/3": "temporal association areas layer 2/3 (left)", "tea4": "temporal association areas layer 4 (left)", "tea5": "temporal association areas layer 5 (left)", "tea6a": "temporal association areas layer 6a (left)", "tea6b": "temporal association areas layer 6b (left)", "peri": "perirhinal area (left)", "peri1": "perirhinal area layer 1 (left)", "peri2/3": "perirhinal area layer 2/3 (left)", "peri5": "perirhinal area layer 5 (left)", "peri6a": "perirhinal area layer 6a (left)", "peri6b": "perirhinal area layer 6b (left)", "ect": "ectorhinal area (left)", "ect1": "ectorhinal area/layer 1 (left)", "ect2/3": "ectorhinal area/layer 2/3 (left)", "ect5": "ectorhinal area/layer 5 (left)", "ect6a": "ectorhinal area/layer 6a (left)", "ect6b": "ectorhinal area/layer 6b (left)", "olf": "olfactory areas (left)", "mob": "main olfactory bulb (left)", "mobgl": "main olfactory bulb glomerular layer (left)", "mobgr": "main olfactory bulb granule layer (left)", "mobipl": "main olfactory bulb inner plexiform layer (left)", "mobmi": "main olfactory bulb mitral layer (left)", "mobopl": "main olfactory bulb outer plexiform layer (left)", "aob": "accessory olfactory bulb (left)", "aobgl": "accessory olfactory bulb glomerular layer (left)", "aobgr": "accessory olfactory bulb granular layer (left)", "aobmi": "accessory olfactory bulb mitral layer (left)", "aon": "anterior olfactory nucleus (left)", "aond": "anterior olfactory nucleus dorsal part (left)", "aone": "anterior olfactory nucleus external part (left)", "aonl": "anterior olfactory nucleus lateral part (left)", "aonm": "anterior olfactory nucleus medial part (left)", "aonpv": "anterior olfactory nucleus posteroventral part (left)", "aon1": "anterior olfactory nucleus layer 1 (left)", "aon2": "anterior olfactory nucleus layer 2 (left)", "tt": "taenia tecta (left)", "ttd": "taenia tecta dorsal part (left)", "ttd1-4": "taenia tecta dorsal part layers 1-4 (left)", "ttd1": "taenia tecta dorsal part layer 1 (left)", "ttd2": "taenia tecta dorsal part layer 2 (left)", "ttd3": "taenia tecta dorsal part layer 3 (left)", "ttd4": "taenia tecta dorsal part layer 4 (left)", "ttv": "taenia tecta ventral part (left)", "ttv1-3": "taenia tecta ventral part layers 1-3 (left)", "ttv1": "taenia tecta ventral part layer 1 (left)", "ttv2": "taenia tecta ventral part layer 2 (left)", "ttv3": "taenia tecta ventral part layer 3 (left)", "dp": "dorsal peduncular area (left)", "dp1": "dorsal peduncular area layer 1 (left)", "dp2": "dorsal peduncular area layer 2 (left)", "dp2/3": "dorsal peduncular area layer 2/3 (left)", "dp5": "dorsal peduncular area layer 5 (left)", "dp6a": "dorsal peduncular area layer 6a (left)", "pir": "piriform area (left)", "pir1-3": "piriform area layers 1-3 (left)", "pir1": "piriform area molecular layer (left)", "pir2": "piriform area pyramidal layer (left)", "pir3": "piriform area polymorph layer (left)", "nlot": "nucleus of the lateral olfactory tract (left)", "nlot1-3": "nucleus of the lateral olfactory tract layers 1-3 (left)", "nlot1": "nucleus of the lateral olfactory tract molecular layer (left)", "nlot2": "nucleus of the lateral olfactory tract pyramidal layer (left)", "nlot3": "nucleus of the lateral olfactory tract layer 3 (left)", "coa": "cortical amygdalar area (left)", "coaa": "cortical amygdalar area anterior part (left)", "coaa1": "cortical amygdalar area anterior part layer 1 (left)", "coaa2": "cortical amygdalar area anterior part layer 2 (left)", "coaa3": "cortical amygdalar area anterior part layer 3 (left)", "coap": "cortical amygdalar area posterior part (left)", "coapl": "cortical amygdalar area posterior part lateral zone (left)", "coapl1-2": "cortical amygdalar area posterior part lateral zone layers 1-2 (left)", "coapl1-3": "cortical amygdalar area posterior part lateral zone layers 1-3 (left)", "coapl1": "cortical amygdalar area posterior part lateral zone layer 1 (left)", "coapl2": "cortical amygdalar area posterior part lateral zone layer 2 (left)", "coapl3": "cortical amygdalar area posterior part lateral zone layer 3 (left)", "coapm": "cortical amygdalar area posterior part medial zone (left)", "coapm1-2": "cortical amygdalar area posterior part medial zone layers 1-2 (left)", "coapm1-3": "cortical amygdalar area posterior part medial zone layers 1-3 (left)", "coapm1": "cortical amygdalar area posterior part medial zone layer 1 (left)", "coapm2": "cortical amygdalar area posterior part medial zone layer 2 (left)", "coapm3": "cortical amygdalar area posterior part medial zone layer 3 (left)", "paa": "piriform-amygdalar area (left)", "paa1-3": "piriform-amygdalar area layers 1-3 (left)", "paa1": "piriform-amygdalar area molecular layer (left)", "paa2": "piriform-amygdalar area pyramidal layer (left)", "paa3": "piriform-amygdalar area polymorph layer (left)", "tr": "postpiriform transition area (left)", "tr1-3": "postpiriform transition area layers 1-3 (left)", "tr1": "postpiriform transition area layers 1 (left)", "tr2": "postpiriform transition area layers 2 (left)", "tr3": "postpiriform transition area layers 3 (left)", "hpf": "hippocampal formation (left)", "hip": "hippocampal region (left)", "ca": "ammon\'s horn (left)", "ca1": "field ca1 (left)", "ca1slm": "field ca1 stratum lacunosum-moleculare (left)", "ca1so": "field ca1 stratum oriens (left)", "ca1sp": "field ca1 pyramidal layer (left)", "ca1sr": "field ca1 stratum radiatum (left)", "ca2": "field ca2 (left)", "ca2slm": "field ca2 stratum lacunosum-moleculare (left)", "ca2so": "field ca2 stratum oriens (left)", "ca2sp": "field ca2 pyramidal layer (left)", "ca2sr": "field ca2 stratum radiatum (left)", "ca3": "field ca3 (left)", "ca3slm": "field ca3 stratum lacunosum-moleculare (left)", "ca3slu": "field ca3 stratum lucidum (left)", "ca3so": "field ca3 stratum oriens (left)", "ca3sp": "field ca3 pyramidal layer (left)", "ca3sr": "field ca3 stratum radiatum (left)", "dg": "dentate gyrus (left)", "dg-mo": "dentate gyrus molecular layer (left)", "dg-po": "dentate gyrus polymorph layer (left)", "dg-sg": "dentate gyrus granule cell layer (left)", "dg-sgz": "dentate gyrus subgranular zone (left)", "dgcr": "dentate gyrus crest (left)", "dgcr-mo": "dentate gyrus crest molecular layer (left)", "dgcr-po": "dentate gyrus crest polymorph layer (left)", "dgcr-sg": "dentate gyrus crest granule cell layer (left)", "dglb": "dentate gyrus lateral blade (left)", "dglb-mo": "dentate gyrus lateral blade molecular layer (left)", "dglb-po": "dentate gyrus lateral blade polymorph layer (left)", "dglb-sg": "dentate gyrus lateral blade granule cell layer (left)", "dgmb": "dentate gyrus medial blade (left)", "dgmb-mo": "dentate gyrus medial blade molecular layer (left)", "dgmb-po": "dentate gyrus medial blade polymorph layer (left)", "dgmb-sg": "dentate gyrus medial blade granule cell layer (left)", "fc": "fasciola cinerea (left)", "ig": "induseum griseum (left)", "rhp": "retrohippocampal region (left)", "ent": "entorhinal area (left)", "entl": "entorhinal area lateral part (left)", "entl1": "entorhinal area lateral part layer 1 (left)", "entl2": "entorhinal area lateral part layer 2 (left)", "entl2/3": "entorhinal area lateral part layer 2/3 (left)", "entl2a": "entorhinal area lateral part layer 2a (left)", "entl2b": "entorhinal area lateral part layer 2b (left)", "entl3": "entorhinal area lateral part layer 3 (left)", "entl4": "entorhinal area lateral part layer 4 (left)", "entl4/5": "entorhinal area lateral part layer 4/5 (left)", "entl5": "entorhinal area lateral part layer 5 (left)", "entl5/6": "entorhinal area lateral part layer 5/6 (left)", "entl6a": "entorhinal area lateral part layer 6a (left)", "entl6b": "entorhinal area lateral part layer 6b (left)", "entm": "entorhinal area medial part dorsal zone (left)", "entm1": "entorhinal area medial part dorsal zone layer 1 (left)", "entm2": "entorhinal area medial part dorsal zone layer 2 (left)", "entm2a": "entorhinal area medial part dorsal zone layer 2a (left)", "entm2b": "entorhinal area medial part dorsal zone layer 2b (left)", "entm3": "entorhinal area medial part dorsal zone layer 3 (left)", "entm4": "entorhinal area medial part dorsal zone layer 4 (left)", "entm5": "entorhinal area medial part dorsal zone layer 5 (left)", "entm5/6": "entorhinal area medial part dorsal zone layer 5/6 (left)", "entm6": "entorhinal area medial part dorsal zone layer 6 (left)", "entmv": "entorhinal area medial part ventral zone (left)", "entmv1": "entorhinal area medial part ventral zone layer 1 (left)", "entmv2": "entorhinal area medial part ventral zone layer 2 (left)", "entmv3": "entorhinal area medial part ventral zone layer 3 (left)", "entmv4": "entorhinal area medial part ventral zone layer 4 (left)", "entmv5/6": "entorhinal area medial part ventral zone layer 5/6 (left)", "par": "parasubiculum (left)", "par1": "parasubiculum layer 1 (left)", "par2": "parasubiculum layer 2 (left)", "par3": "parasubiculum layer 3 (left)", "post": "postsubiculum (left)", "post1": "postsubiculum layer 1 (left)", "post2": "postsubiculum layer 2 (left)", "post3": "postsubiculum layer 3 (left)", "pre": "presubiculum (left)", "pre1": "presubiculum layer 1 (left)", "pre2": "presubiculum layer 2 (left)", "pre3": "presubiculum layer 3 (left)", "sub": "subiculum (left)", "subd": "subiculum dorsal part (left)", "subd-m": "subiculum dorsal part molecular layer (left)", "subd-sp": "subiculum dorsal part pyramidal layer (left)", "subd-sr": "subiculum dorsal part stratum radiatum (left)", "subv": "subiculum ventral part (left)", "subv-m": "subiculum ventral part molecular layer (left)", "subv-sp": "subiculum ventral part pyramidal layer (left)", "subv-sr": "subiculum ventral part stratum radiatum (left)", "pros": "prosubiculum (left)", "prosd": "prosubiculum dorsal part (left)", "prosd-m": "prosubiculum dorsal part molecular layer (left)", "prosd-sp": "prosubiculum dorsal part pyramidal layer (left)", "prosd-sr": "prosubiculum dorsal part stratum radiatum (left)", "prosv": "prosubiculum ventral part (left)", "prosv-m": "prosubiculum ventral part molecular layer (left)", "prosv-sp": "prosubiculum ventral part pyramidal layer (left)", "prosv-sr": "prosubiculum ventral part stratum radiatum (left)", "hata": "hippocampo-amygdalar transition area (left)", "apr": "area prostriata (left)", "ctxsp": "cortical subplate (left)", "6b": "layer 6b isocortex (left)", "cla": "claustrum (left)", "ep": "endopiriform nucleus (left)", "epd": "endopiriform nucleus dorsal part (left)", "epv": "endopiriform nucleus ventral part (left)", "la": "lateral amygdalar nucleus (left)", "bla": "basolateral amygdalar nucleus (left)", "blaa": "basolateral amygdalar nucleus anterior part (left)", "blap": "basolateral amygdalar nucleus posterior part (left)", "blav": "basolateral amygdalar nucleus ventral part (left)", "bma": "basomedial amygdalar nucleus (left)", "bmaa": "basomedial amygdalar nucleus anterior part (left)", "bmap": "basomedial amygdalar nucleus posterior part (left)", "pa": "posterior amygdalar nucleus (left)", "cnu": "cerebral nuclei (left)", "str": "striatum (left)", "strd": "striatum dorsal region (left)", "cp": "caudoputamen (left)", "strv": "striatum ventral region (left)", "acb": "nucleus accumbens (left)", "fs": "fundus of striatum (left)", "ot": "olfactory tubercle (left)", "isl": "islands of calleja (left)", "islm": "major island of calleja (left)", "ot1-3": "olfactory tubercle layers 1-3 (left)", "ot1": "olfactory tubercle molecular layer (left)", "ot2": "olfactory tubercle pyramidal layer (left)", "ot3": "olfactory tubercle polymorph layer (left)", "lss": "lateral strip of striatum (left)", "lsx": "lateral septal complex (left)", "ls": "lateral septal nucleus (left)", "lsc": "lateral septal nucleus caudal (caudodorsal) part (left)", "lsr": "lateral septal nucleus rostral (rostroventral) part (left)", "lsv": "lateral septal nucleus ventral part (left)", "sf": "septofimbrial nucleus (left)", "sh": "septohippocampal nucleus (left)", "samy": "striatum-like amygdalar nuclei (left)", "aaa": "anterior amygdalar area (left)", "ba": "bed nucleus of the accessory olfactory tract (left)", "cea": "central amygdalar nucleus (left)", "ceac": "central amygdalar nucleus capsular part (left)", "ceal": "central amygdalar nucleus lateral part (left)", "ceam": "central amygdalar nucleus medial part (left)", "ia": "intercalated amygdalar nucleus (left)", "mea": "medial amygdalar nucleus (left)", "meaad": "medial amygdalar nucleus anterodorsal part (left)", "meaav": "medial amygdalar nucleus anteroventral part (left)", "meapd": "medial amygdalar nucleus posterodorsal part (left)", "meapd-a": "medial amygdalar nucleus posterodorsal part sublayer a (left)", "meapd-b": "medial amygdalar nucleus posterodorsal part sublayer b (left)", "meapd-c": "medial amygdalar nucleus posterodorsal part sublayer c (left)", "meapv": "medial amygdalar nucleus posteroventral part (left)", "pal": "pallidum (left)", "pald": "pallidum dorsal region (left)", "gpe": "globus pallidus external segment (left)", "gpi": "globus pallidus internal segment (left)", "palv": "pallidum ventral region (left)", "si": "substantia innominata (left)", "ma": "magnocellular nucleus (left)", "palm": "pallidum medial region (left)", "msc": "medial septal complex (left)", "ms": "medial septal nucleus (left)", "ndb": "diagonal band nucleus (left)", "trs": "triangular nucleus of septum (left)", "palc": "pallidum caudal region (left)", "bst": "bed nuclei of the stria terminalis (left)", "bsta": "bed nuclei of the stria terminalis anterior division (left)", "bstal": "bed nuclei of the stria terminalis anterior division anterolateral area (left)", "bstam": "bed nuclei of the stria terminalis anterior division anteromedial area (left)", "bstdm": "bed nuclei of the stria terminalis anterior division dorsomedial nucleus (left)", "bstfu": "bed nuclei of the stria terminalis anterior division fusiform nucleus (left)", "bstju": "bed nuclei of the stria terminalis anterior division juxtacapsular nucleus (left)", "bstmg": "bed nuclei of the stria terminalis anterior division magnocellular nucleus (left)", "bstov": "bed nuclei of the stria terminalis anterior division oval nucleus (left)", "bstrh": "bed nuclei of the stria terminalis anterior division rhomboid nucleus (left)", "bstv": "bed nuclei of the stria terminalis anterior division ventral nucleus (left)", "bstp": "bed nuclei of the stria terminalis posterior division (left)", "bstd": "bed nuclei of the stria terminalis posterior division dorsal nucleus (left)", "bstpr": "bed nuclei of the stria terminalis posterior division principal nucleus (left)", "bstif": "bed nuclei of the stria terminalis posterior division interfascicular nucleus (left)", "bsttr": "bed nuclei of the stria terminalis posterior division transverse nucleus (left)", "bstse": "bed nuclei of the stria terminalis posterior division strial extension (left)", "bac": "bed nucleus of the anterior commissure (left)", "bs": "brain stem (left)", "ib": "interbrain (left)", "th": "thalamus (left)", "dorsm": "thalamus sensory-motor cortex related (left)", "vent": "ventral group of the dorsal thalamus (left)", "val": "ventral anterior-lateral complex of the thalamus (left)", "vm": "ventral medial nucleus of the thalamus (left)", "vp": "ventral posterior complex of the thalamus (left)", "vpl": "ventral posterolateral nucleus of the thalamus (left)", "vplpc": "ventral posterolateral nucleus of the thalamus parvicellular part (left)", "vpm": "ventral posteromedial nucleus of the thalamus (left)", "vpmpc": "ventral posteromedial nucleus of the thalamus parvicellular part (left)", "pot": "posterior triangular thalamic nucleus (left)", "spf": "subparafascicular nucleus (left)", "spfm": "subparafascicular nucleus magnocellular part (left)", "spfp": "subparafascicular nucleus parvicellular part (left)", "spa": "subparafascicular area (left)", "pp": "peripeduncular nucleus (left)", "gend": "geniculate group dorsal thalamus (left)", "mg": "medial geniculate complex (left)", "mgd": "medial geniculate complex dorsal part (left)", "mgv": "medial geniculate complex ventral part (left)", "mgm": "medial geniculate complex medial part (left)", "lgd": "dorsal part of the lateral geniculate complex (left)", "lgd-sh": "dorsal part of the lateral geniculate complex shell (left)", "lgd-co": "dorsal part of the lateral geniculate complex core (left)", "lgd-ip": "dorsal part of the lateral geniculate complex ipsilateral zone (left)", "dorpm": "thalamus polymodal association cortex related (left)", "lat": "lateral group of the dorsal thalamus (left)", "lp": "lateral posterior nucleus of the thalamus (left)", "po": "posterior complex of the thalamus (left)", "pol": "posterior limiting nucleus of the thalamus (left)", "sgn": "suprageniculate nucleus (left)", "eth": "ethmoid nucleus of the thalamus (left)", "reth": "retroethmoid nucleus (left)", "atn": "anterior group of the dorsal thalamus (left)", "av": "anteroventral nucleus of thalamus (left)", "am": "anteromedial nucleus (left)", "amd": "anteromedial nucleus dorsal part (left)", "amv": "anteromedial nucleus ventral part (left)", "ad": "anterodorsal nucleus (left)", "iam": "interanteromedial nucleus of the thalamus (left)", "iad": "interanterodorsal nucleus of the thalamus (left)", "ld": "lateral dorsal nucleus of thalamus (left)", "med": "medial group of the dorsal thalamus (left)", "imd": "intermediodorsal nucleus of the thalamus (left)", "md": "mediodorsal nucleus of thalamus (left)", "mdc": "mediodorsal nucleus of the thalamus central part (left)", "mdl": "mediodorsal nucleus of the thalamus lateral part (left)", "mdm": "mediodorsal nucleus of the thalamus medial part (left)", "smt": "submedial nucleus of the thalamus (left)", "pr": "perireunensis nucleus (left)", "mtn": "midline group of the dorsal thalamus (left)", "pvt": "paraventricular nucleus of the thalamus (left)", "pt": "parataenial nucleus (left)", "re": "nucleus of reuniens (left)", "xi": "xiphoid thalamic nucleus (left)", "ilm": "intralaminar nuclei of the dorsal thalamus (left)", "rh": "rhomboid nucleus (left)", "cm": "cranial nerves (left)", "pcn": "paracentral nucleus (left)", "cl": "central lateral nucleus of the thalamus (left)", "pf": "parafascicular nucleus (left)", "pil": "posterior intralaminar thalamic nucleus (left)", "rt": "reticular nucleus of the thalamus (left)", "genv": "geniculate group ventral thalamus (left)", "igl": "intergeniculate leaflet of the lateral geniculate complex (left)", "intg": "intermediate geniculate nucleus (left)", "lgv": "ventral part of the lateral geniculate complex (left)", "lgvl": "ventral part of the lateral geniculate complex lateral zone (left)", "lgvm": "ventral part of the lateral geniculate complex medial zone (left)", "subg": "subgeniculate nucleus (left)", "epi": "epithalamus (left)", "mh": "medial habenula (left)", "lh": "lateral habenula (left)", "pin": "pineal body (left)", "hy": "hypothalamus (left)", "pvz": "periventricular zone (left)", "so": "supraoptic nucleus (left)", "aso": "accessory supraoptic group (left)", "nc": "nucleus circularis (left)", "pvh": "paraventricular hypothalamic nucleus (left)", "pvhm": "paraventricular hypothalamic nucleus magnocellular division (left)", "pvham": "paraventricular hypothalamic nucleus magnocellular division anterior magnocellular part (left)", "pvhmm": "paraventricular hypothalamic nucleus magnocellular division medial magnocellular part (left)", "pvhpm": "paraventricular hypothalamic nucleus magnocellular division posterior magnocellular part (left)", "pvhpml": "paraventricular hypothalamic nucleus magnocellular division posterior magnocellular part lateral zone (left)", "pvhpmm": "paraventricular hypothalamic nucleus magnocellular division posterior magnocellular part medial zone (left)", "pvhp": "paraventricular hypothalamic nucleus parvicellular division (left)", "pvhap": "paraventricular hypothalamic nucleus parvicellular division anterior parvicellular part (left)", "pvhmpd": "paraventricular hypothalamic nucleus parvicellular division medial parvicellular part dorsal zone (left)", "pvhpv": "paraventricular hypothalamic nucleus parvicellular division periventricular part (left)", "pva": "periventricular hypothalamic nucleus anterior part (left)", "pvi": "periventricular hypothalamic nucleus intermediate part (left)", "arh": "arcuate hypothalamic nucleus (left)", "pvr": "periventricular region (left)", "adp": "anterodorsal preoptic nucleus (left)", "aha": "anterior hypothalamic area (left)", "avp": "anteroventral preoptic nucleus (left)", "avpv": "anteroventral periventricular nucleus (left)", "dmh": "dorsomedial nucleus of the hypothalamus (left)", "dmha": "dorsomedial nucleus of the hypothalamus anterior part (left)", "dmhp": "dorsomedial nucleus of the hypothalamus posterior part (left)", "dmhv": "dorsomedial nucleus of the hypothalamus ventral part (left)", "mepo": "median preoptic nucleus (left)", "mpo": "medial preoptic area (left)", "ov": "vascular organ of the lamina terminalis (left)", "pd": "posterodorsal preoptic nucleus (left)", "ps": "parastrial nucleus (left)", "psch": "suprachiasmatic preoptic nucleus (left)", "pvp": "periventricular hypothalamic nucleus posterior part (left)", "pvpo": "periventricular hypothalamic nucleus preoptic part (left)", "sbpv": "subparaventricular zone (left)", "sch": "suprachiasmatic nucleus (left)", "sfo": "subfornical organ (left)", "vmpo": "ventromedial preoptic nucleus (left)", "vlpo": "ventrolateral preoptic nucleus (left)", "mez": "hypothalamic medial zone (left)", "ahn": "anterior hypothalamic nucleus (left)", "ahna": "anterior hypothalamic nucleus anterior part (left)", "ahnc": "anterior hypothalamic nucleus central part (left)", "ahnd": "anterior hypothalamic nucleus dorsal part (left)", "ahnp": "anterior hypothalamic nucleus posterior part (left)", "mbo": "mammillary body (left)", "lm": "lateral mammillary nucleus (left)", "mm": "medial mammillary nucleus (left)", "mmme": "medial mammillary nucleus median part (left)", "mml": "medial mammillary nucleus lateral part (left)", "mmm": "medial mammillary nucleus medial part (left)", "mmp": "medial mammillary nucleus posterior part (left)", "mmd": "medial mammillary nucleus dorsal part (left)", "sum": "supramammillary nucleus (left)", "suml": "supramammillary nucleus lateral part (left)", "summ": "supramammillary nucleus medial part (left)", "tm": "tuberomammillary nucleus (left)", "tmd": "tuberomammillary nucleus dorsal part (left)", "tmv": "tuberomammillary nucleus ventral part (left)", "mpn": "medial preoptic nucleus (left)", "mpnc": "medial preoptic nucleus central part (left)", "mpnl": "medial preoptic nucleus lateral part (left)", "mpnm": "medial preoptic nucleus medial part (left)", "pmd": "dorsal premammillary nucleus (left)", "pmv": "ventral premammillary nucleus (left)", "pvhd": "paraventricular hypothalamic nucleus descending division (left)", "pvhdp": "paraventricular hypothalamic nucleus descending division dorsal parvicellular part (left)", "pvhf": "paraventricular hypothalamic nucleus descending division forniceal part (left)", "pvhlp": "paraventricular hypothalamic nucleus descending division lateral parvicellular part (left)", "pvhmpv": "paraventricular hypothalamic nucleus descending division medial parvicellular part ventral zone (left)", "vmh": "ventromedial hypothalamic nucleus (left)", "vmha": "ventromedial hypothalamic nucleus anterior part (left)", "vmhc": "ventromedial hypothalamic nucleus central part (left)", "vmhdm": "ventromedial hypothalamic nucleus dorsomedial part (left)", "vmhvl": "ventromedial hypothalamic nucleus ventrolateral part (left)", "ph": "posterior hypothalamic nucleus (left)", "lz": "hypothalamic lateral zone (left)", "lha": "lateral hypothalamic area (left)", "lpo": "lateral preoptic area (left)", "pst": "preparasubthalamic nucleus (left)", "pstn": "parasubthalamic nucleus (left)", "pef": "perifornical nucleus (left)", "rch": "retrochiasmatic area (left)", "stn": "subthalamic nucleus (left)", "tu": "tuberal nucleus (left)", "zi": "zona incerta (left)", "a13": "dopaminergic a13 group (left)", "ff": "fields of forel (left)", "me": "median eminence (left)", "mb": "midbrain (left)", "mbsen": "midbrain sensory related (left)", "scs": "superior colliculus sensory related (left)", "scop": "superior colliculus optic layer (left)", "scsg": "superior colliculus superficial gray layer (left)", "sczo": "superior colliculus zonal layer (left)", "ic": "inferior colliculus (left)", "icc": "inferior colliculus central nucleus (left)", "icd": "inferior colliculus dorsal nucleus (left)", "ice": "inferior colliculus external nucleus (left)", "nb": "nucleus of the brachium of the inferior colliculus (left)", "sag": "nucleus sagulum (left)", "pbg": "parabigeminal nucleus (left)", "mev": "midbrain trigeminal nucleus (left)", "sco": "subcommissural organ (left)", "mbmot": "midbrain motor related (left)", "snr": "substantia nigra reticular part (left)", "vta": "ventral tegmental area (left)", "pn": "paranigral nucleus (left)", "rr": "midbrain reticular nucleus retrorubral area (left)", "mrn": "midbrain reticular nucleus (left)", "mrnm": "midbrain reticular nucleus magnocellular part (left)", "mrnmg": "midbrain reticular nucleus magnocellular part general (left)", "mrnp": "midbrain reticular nucleus parvicellular part (left)", "scm": "superior colliculus motor related (left)", "scdg": "superior colliculus motor related deep gray layer (left)", "scdw": "superior colliculus motor related deep white layer (left)", "sciw": "superior colliculus motor related intermediate white layer (left)", "scig": "superior colliculus motor related intermediate gray layer (left)", "scig-a": "superior colliculus motor related intermediate gray layer sublayer a (left)", "scig-b": "superior colliculus motor related intermediate gray layer sublayer b (left)", "scig-c": "superior colliculus motor related intermediate gray layer sublayer c (left)", "pag": "periaqueductal gray (left)", "prc": "precommissural nucleus (left)", "inc": "interstitial nucleus of cajal (left)", "nd": "nucleus of darkschewitsch (left)", "su3": "supraoculomotor periaqueductal gray (left)", "prt": "pretectal region (left)", "apn": "anterior pretectal nucleus (left)", "mpt": "medial pretectal area (left)", "not": "nucleus of the optic tract (left)", "npc": "nucleus of the posterior commissure (left)", "op": "olivary pretectal nucleus (left)", "ppt": "posterior pretectal nucleus (left)", "rpf": "retroparafascicular nucleus (left)", "inco": "intercollicular nucleus (left)", "cun": "cuneiform nucleus (left)", "rn": "red nucleus (left)", "iii": "oculomotor nucleus (left)", "ma3": "medial accesory oculomotor nucleus (left)", "ew": "edinger-westphal nucleus (left)", "iv": "trochlear nucleus (left)", "pa4": "paratrochlear nucleus (left)", "vtn": "ventral tegmental nucleus (left)", "at": "anterior tegmental nucleus (left)", "lt": "lateral terminal nucleus of the accessory optic tract (left)", "dt": "dorsal terminal nucleus of the accessory optic tract (left)", "mt": "medial terminal nucleus of the accessory optic tract (left)", "snl": "substantia nigra lateral part (left)", "mbsta": "midbrain behavioral state related (left)", "snc": "substantia nigra compact part (left)", "ppn": "pedunculopontine nucleus (left)", "ramb": "midbrain raphe nuclei (left)", "if": "interfascicular nucleus raphe (left)", "ipn": "interpeduncular nucleus (left)", "ipr": "interpeduncular nucleus rostral (left)", "ipc": "interpeduncular nucleus caudal (left)", "ipa": "interpeduncular nucleus apical (left)", "ipl": "interpeduncular nucleus lateral (left)", "ipi": "interpeduncular nucleus intermediate (left)", "ipdm": "interpeduncular nucleus dorsomedial (left)", "ipdl": "interpeduncular nucleus dorsolateral (left)", "iprl": "interpeduncular nucleus rostrolateral (left)", "rl": "rostral linear nucleus raphe (left)", "cli": "central linear nucleus raphe (left)", "dr": "dorsal nucleus raphe (left)", "hb": "hindbrain (left)", "p": "pons (left)", "p-sen": "pons sensory related (left)", "nll": "nucleus of the lateral lemniscus (left)", "nlld": "nucleus of the lateral lemniscus dorsal part (left)", "nllh": "nucleus of the lateral lemniscus horizontal part (left)", "nllv": "nucleus of the lateral lemniscus ventral part (left)", "psv": "principal sensory nucleus of the trigeminal (left)", "pb": "parabrachial nucleus (left)", "kf": "koelliker-fuse subnucleus (left)", "pbl": "parabrachial nucleus lateral division (left)", "pblc": "parabrachial nucleus lateral division central lateral part (left)", "pbld": "parabrachial nucleus lateral division dorsal lateral part (left)", "pble": "parabrachial nucleus lateral division external lateral part (left)", "pbls": "parabrachial nucleus lateral division superior lateral part (left)", "pblv": "parabrachial nucleus lateral division ventral lateral part (left)", "pbm": "parabrachial nucleus medial division (left)", "pbme": "parabrachial nucleus medial division external medial part (left)", "pbmm": "parabrachial nucleus medial division medial medial part (left)", "pbmv": "parabrachial nucleus medial division ventral medial part (left)", "soc": "superior olivary complex (left)", "por": "superior olivary complex periolivary region (left)", "socm": "superior olivary complex medial part (left)", "socl": "superior olivary complex lateral part (left)", "p-mot": "pons motor related (left)", "b": "barrington\'s nucleus (left)", "dtn": "dorsal tegmental nucleus (left)", "ltn": "lateral tegmental nucleus (left)", "pdtg": "posterodorsal tegmental nucleus (left)", "pcg": "pontine central gray (left)", "pg": "pontine gray (left)", "prnc": "pontine reticular nucleus caudal part (left)", "prnv": "pontine reticular nucleus ventral part (left)", "sg": "supragenual nucleus (left)", "ssn": "superior salivatory nucleus (left)", "sut": "supratrigeminal nucleus (left)", "trn": "tegmental reticular nucleus (left)", "v": "motor nucleus of trigeminal (left)", "p5": "peritrigeminal zone (left)", "acs5": "accessory trigeminal nucleus (left)", "pc5": "parvicellular motor 5 nucleus (left)", "i5": "intertrigeminal nucleus (left)", "p-sat": "pons behavioral state related (left)", "cs": "superior central nucleus raphe (left)", "csl": "superior central nucleus raphe lateral part (left)", "csm": "superior central nucleus raphe medial part (left)", "lc": "locus ceruleus (left)", "ldt": "laterodorsal tegmental nucleus (left)", "ni": "nucleus incertus (left)", "prnr": "pontine reticular nucleus (left)", "rpo": "nucleus raphe pontis (left)", "slc": "subceruleus nucleus (left)", "sld": "sublaterodorsal nucleus (left)", "my": "medulla (left)", "my-sen": "medulla sensory related (left)", "ap": "area postrema (left)", "cn": "cochlear nuclei (left)", "cnlam": "granular lamina of the cochlear nuclei (left)", "cnspg": "cochlear nucleus subpedunclular granular region (left)", "dco": "dorsal cochlear nucleus (left)", "vco": "ventral cochlear nucleus (left)", "dcn": "dorsal column nuclei (left)", "cu": "cuneate nucleus (left)", "gr": "gracile nucleus (left)", "ecu": "external cuneate nucleus (left)", "ntb": "nucleus of the trapezoid body (left)", "nts": "nucleus of the solitary tract (left)", "ntsce": "nucleus of the solitary tract central part (left)", "ntsco": "nucleus of the solitary tract commissural part (left)", "ntsge": "nucleus of the solitary tract gelatinous part (left)", "ntsl": "nucleus of the solitary tract lateral part (left)", "ntsm": "nucleus of the solitary tract medial part (left)", "spvc": "spinal nucleus of the trigeminal caudal part (left)", "spvi": "spinal nucleus of the trigeminal interpolar part (left)", "spvo": "spinal nucleus of the trigeminal oral part (left)", "spvocdm": "spinal nucleus of the trigeminal oral part caudal dorsomedial part (left)", "spvomdmd": "spinal nucleus of the trigeminal oral part middle dorsomedial part dorsal zone (left)", "spvomdmv": "spinal nucleus of the trigeminal oral part middle dorsomedial part ventral zone (left)", "spvordm": "spinal nucleus of the trigeminal oral part rostral dorsomedial part (left)", "spvovl": "spinal nucleus of the trigeminal oral part ventrolateral part (left)", "pa5": "paratrigeminal nucleus (left)", "z": "nucleus z (left)", "my-mot": "medulla motor related (left)", "vi": "abducens nucleus (left)", "acvi": "accessory abducens nucleus (left)", "vii": "facial motor nucleus (left)", "acvii": "accessory facial motor nucleus (left)", "ev": "efferent vestibular nucleus (left)", "amb": "nucleus ambiguus (left)", "ambd": "nucleus ambiguus dorsal division (left)", "ambv": "nucleus ambiguus ventral division (left)", "dmx": "dorsal motor nucleus of the vagus nerve (left)", "eco": "efferent cochlear group (left)", "grn": "gigantocellular reticular nucleus (left)", "icb": "infracerebellar nucleus (left)", "io": "inferior olivary complex (left)", "irn": "intermediate reticular nucleus (left)", "isn": "inferior salivatory nucleus (left)", "lin": "linear nucleus of the medulla (left)", "lrn": "lateral reticular nucleus (left)", "lrnm": "lateral reticular nucleus magnocellular part (left)", "lrnp": "lateral reticular nucleus parvicellular part (left)", "marn": "magnocellular reticular nucleus (left)", "mdrn": "medullary reticular nucleus (left)", "mdrnd": "medullary reticular nucleus dorsal part (left)", "mdrnv": "medullary reticular nucleus ventral part (left)", "parn": "parvicellular reticular nucleus (left)", "pas": "parasolitary nucleus (left)", "pgrn": "paragigantocellular reticular nucleus (left)", "pgrnd": "paragigantocellular reticular nucleus dorsal part (left)", "pgrnl": "paragigantocellular reticular nucleus lateral part (left)", "phy": "perihypoglossal nuclei (left)", "nis": "nucleus intercalatus (left)", "nr": "nucleus of roller (left)", "prp": "nucleus prepositus (left)", "pmr": "paramedian reticular nucleus (left)", "ppy": "parapyramidal nucleus (left)", "ppyd": "parapyramidal nucleus deep part (left)", "ppys": "parapyramidal nucleus superficial part (left)", "vnc": "vestibular nuclei (left)", "lav": "lateral vestibular nucleus (left)", "mv": "medial vestibular nucleus (left)", "spiv": "spinal vestibular nucleus (left)", "suv": "superior vestibular nucleus (left)", "x": "nucleus x (left)", "xii": "hypoglossal nucleus (left)", "y": "nucleus y (left)", "inv": "interstitial nucleus of the vestibular nerve (left)", "my-sat": "medulla behavioral state related (left)", "rm": "nucleus raphe magnus (left)", "rpa": "nucleus raphe pallidus (left)", "ro": "nucleus raphe obscurus (left)", "cb": "cerebellum (left)", "cbx": "cerebellar cortex (left)", "cbxmo": "cerebellar cortex molecular layer (left)", "cbxpu": "cerebellar cortex purkinje layer (left)", "cbxgr": "cerebellar cortex granular layer (left)", "verm": "vermal regions (left)", "ling": "lingula (i) (left)", "lingmo": "lingula (i) molecular layer (left)", "lingpu": "lingula (i) purkinje layer (left)", "linggr": "lingula (i) granular layer (left)", "cent": "central lobule (left)", "cent2": "lobule ii (left)", "cent2mo": "lobule ii molecular layer (left)", "cent2pu": "lobule ii purkinje layer (left)", "cent2gr": "lobule ii granular layer (left)", "cent3": "lobule iii (left)", "cent3mo": "lobule iii molecular layer (left)", "cent3pu": "lobule iii purkinje layer (left)", "cent3gr": "lobule iii granular layer (left)", "cul": "culmen (left)", "cul4": "lobule iv (left)", "cul4mo": "lobule iv molecular layer (left)", "cul4pu": "lobule iv purkinje layer (left)", "cul4gr": "lobule iv granular layer (left)", "cul5": "lobule v (left)", "cul5mo": "lobule v molecular layer (left)", "cul5pu": "lobule v purkinje layer (left)", "cul5gr": "lobule v granular layer (left)", "cul4 5": "lobules iv-v (left)", "cul4 5mo": "lobules iv-v molecular layer (left)", "cul4 5pu": "lobules iv-v purkinje layer (left)", "cul4 5gr": "lobules iv-v granular layer (left)", "dec": "declive (vi) (left)", "decmo": "declive (vi) molecular layer (left)", "decpu": "declive (vi) purkinje layer (left)", "decgr": "declive (vi) granular layer (left)", "fotu": "folium-tuber vermis (vii) (left)", "fotumo": "folium-tuber vermis (vii) molecular layer (left)", "fotupu": "folium-tuber vermis (vii) purkinje layer (left)", "fotugr": "folium-tuber vermis (vii) granular layer (left)", "pyr": "pyramus (viii) (left)", "pyrmo": "pyramus (viii) molecular layer (left)", "pyrpu": "pyramus (viii) purkinje layer (left)", "pyrgr": "pyramus (viii) granular layer (left)", "uvu": "uvula (ix) (left)", "uvumo": "uvula (ix) molecular layer (left)", "uvupu": "uvula (ix) purkinje layer (left)", "uvugr": "uvula (ix) granular layer (left)", "nod": "nodulus (x) (left)", "nodmo": "nodulus (x) molecular layer (left)", "nodpu": "nodulus (x) purkinje layer (left)", "nodgr": "nodulus (x) granular layer (left)", "hem": "hemispheric regions (left)", "sim": "simple lobule (left)", "simmo": "simple lobule molecular layer (left)", "simpu": "simple lobule purkinje layer (left)", "simgr": "simple lobule granular layer (left)", "an": "ansiform lobule (left)", "ancr1": "crus 1 (left)", "ancr1mo": "crus 1 molecular layer (left)", "ancr1pu": "crus 1 purkinje layer (left)", "ancr1gr": "crus 1 granular layer (left)", "ancr2": "crus 2 (left)", "ancr2mo": "crus 2 molecular layer (left)", "ancr2pu": "crus 2 purkinje layer (left)", "ancr2gr": "crus 2 granular layer (left)", "prm": "paramedian lobule (left)", "prmmo": "paramedian lobule molecular layer (left)", "prmpu": "paramedian lobule purkinje layer (left)", "prmgr": "paramedian lobule granular layer (left)", "copy": "copula pyramidis (left)", "copymo": "copula pyramidis molecular layer (left)", "copypu": "copula pyramidis purkinje layer (left)", "copygr": "copula pyramidis granular layer (left)", "pfl": "paraflocculus (left)", "pflmo": "paraflocculus molecular layer (left)", "pflpu": "paraflocculus purkinje layer (left)", "pflgr": "paraflocculus granular layer (left)", "fl": "flocculus (left)", "flmo": "flocculus molecular layer (left)", "flpu": "flocculus purkinje layer (left)", "flgr": "flocculus granular layer (left)", "cbn": "cerebellar nuclei (left)", "fn": "fastigial nucleus (left)", "ip": "interposed nucleus (left)", "dn": "dentate nucleus (left)", "vecb": "vestibulocerebellar nucleus (left)", "fiber tracts": "fiber tracts (left)", "tn": "terminal nerve (left)", "von": "vomeronasal nerve (left)", "in": "olfactory nerve (left)", "onl": "olfactory nerve layer of main olfactory bulb (left)", "lotg": "lateral olfactory tract general (left)", "lot": "lateral olfactory tract body (left)", "lotd": "dorsal limb (left)", "aolt": "accessory olfactory tract (left)", "aco": "anterior commissure olfactory limb (left)", "iin": "optic nerve (left)", "aot": "accessory optic tract (left)", "bsc": "brachium of the superior colliculus (left)", "csc": "superior colliculus commissure (left)", "och": "optic chiasm (left)", "opt": "optic tract (left)", "ttp": "tectothalamic pathway (left)", "iiin": "oculomotor nerve (left)", "mlf": "medial longitudinal fascicle (left)", "pc": "posterior commissure (left)", "ivn": "trochlear nerve (left)", "ivd": "trochlear nerve decussation (left)", "vin": "abducens nerve (left)", "vn": "trigeminal nerve (left)", "mov": "motor root of the trigeminal nerve (left)", "sv": "sensory root of the trigeminal nerve (left)", "mtv": "midbrain tract of the trigeminal nerve (left)", "sptv": "spinal tract of the trigeminal nerve (left)", "viin": "facial nerve (left)", "iviin": "intermediate nerve (left)", "gviin": "genu of the facial nerve (left)", "viiin": "vestibulocochlear nerve (left)", "cvb": "efferent cochleovestibular bundle (left)", "vviiin": "vestibular nerve (left)", "cviiin": "cochlear nerve (left)", "tb": "trapezoid body (left)", "ias": "intermediate acoustic stria (left)", "das": "dorsal acoustic stria (left)", "ll": "lateral lemniscus (left)", "cic": "inferior colliculus commissure (left)", "bic": "brachium of the inferior colliculus (left)", "ixn": "glossopharyngeal nerve (left)", "xn": "vagus nerve (left)", "ts": "solitary tract (left)", "xin": "accessory spinal nerve (left)", "xiin": "hypoglossal nerve (left)", "vrt": "ventral roots (left)", "drt": "dorsal roots (left)", "cett": "cervicothalamic tract (left)", "dl": "dorsolateral fascicle (left)", "dcm": "dorsal commissure of the spinal cord (left)", "vc": "ventral commissure of the spinal cord (left)", "fpr": "fasciculus proprius (left)", "dc": "dorsal column (left)", "cuf": "cuneate fascicle (left)", "grf": "gracile fascicle (left)", "iaf": "internal arcuate fibers (left)", "ml": "medial lemniscus (left)", "sst": "spinothalamic tract (left)", "sttl": "lateral spinothalamic tract (left)", "sttv": "ventral spinothalamic tract (left)", "scrt": "spinocervical tract (left)", "sop": "spino-olivary pathway (left)", "srp": "spinoreticular pathway (left)", "svp": "spinovestibular pathway (left)", "stp": "spinotectal pathway (left)", "shp": "spinohypothalamic pathway (left)", "step": "spinotelenchephalic pathway (left)", "hht": "hypothalamohypophysial tract (left)", "cbf": "cerebellum related fiber tracts (left)", "cbc": "cerebellar commissure (left)", "cbp": "cerebellar peduncles (left)", "scp": "superior cerebelar peduncles (left)", "dscp": "superior cerebellar peduncle decussation (left)", "sct": "spinocerebellar tract (left)", "uf": "uncinate fascicle (left)", "sctv": "ventral spinocerebellar tract (left)", "mcp": "middle cerebellar peduncle (left)", "icp": "inferior cerebellar peduncle (left)", "sctd": "dorsal spinocerebellar tract (left)", "cct": "cuneocerebellar tract (left)", "jrb": "juxtarestiform body (left)", "bct": "bulbocerebellar tract (left)", "oct": "olivocerebellar tract (left)", "rct": "reticulocerebellar tract (left)", "tct": "trigeminocerebellar tract (left)", "arb": "arbor vitae (left)", "scwm": "supra-callosal cerebral white matter (left)", "lfbs": "lateral forebrain bundle system (left)", "cc": "corpus callosum (left)", "fa": "corpus callosum anterior forceps (left)", "ec": "external capsule (left)", "ee": "corpus callosum extreme capsule (left)", "ccg": "genu of corpus callosum (left)", "fp": "corpus callosum posterior forceps (left)", "ccr": "corpus callosum rostrum (left)", "ccb": "corpus callosum body (left)", "ccs": "corpus callosum splenium (left)", "cst": "corticospinal tract (left)", "int": "internal capsule (left)", "cpd": "cerebal peduncle (left)", "cte": "corticotectal tract (left)", "crt": "corticorubral tract (left)", "cpt": "corticopontine tract (left)", "cbt": "corticobulbar tract (left)", "py": "pyramid (left)", "pyd": "pyramidal decussation (left)", "cstc": "corticospinal tract crossed (left)", "cstu": "corticospinal tract uncrossed (left)", "lfbst": "thalamus related (left)", "em": "external medullary lamina of the thalamus (left)", "im": "internal medullary lamina of the thalamus (left)", "mtc": "middle thalamic commissure (left)", "tp": "thalamic peduncles (left)", "or": "optic radiation (left)", "ar": "auditory radiation (left)", "eps": "extrapyramidal fiber systems (left)", "epsc": "cerebral nuclei related (left)", "pap": "pallidothalamic pathway (left)", "nst": "nigrostriatal tract (left)", "ntt": "nigrothalamic fibers (left)", "ptf": "pallidotegmental fascicle (left)", "snp": "striatonigral pathway (left)", "stf": "subthalamic fascicle (left)", "tsp": "tectospinal pathway (left)", "tspd": "direct tectospinal pathway (left)", "dtd": "doral tegmental decussation (left)", "tspc": "crossed tectospinal pathway (left)", "rust": "rubrospinal tract (left)", "vtd": "ventral tegmental decussation (left)", "rrt": "rubroreticular tract (left)", "ctb": "central tegmental bundle (left)", "rst": "retriculospinal tract (left)", "rstl": "retriculospinal tract lateral part (left)", "rstm": "retriculospinal tract medial part (left)", "vsp": "vestibulospinal pathway (left)", "mfbs": "medial forebrain bundle system (left)", "mfbc": "cerebrum related (left)", "amc": "amygdalar capsule (left)", "apd": "ansa peduncularis (left)", "act": "anterior commissure temporal limb (left)", "cing": "cingulum bundle (left)", "fxs": "fornix system (left)", "alv": "alveus (left)", "df": "dorsal fornix (left)", "fi": "fimbria (left)", "fxprg": "precommissural fornix general (left)", "db": "precommissural fornix diagonal band (left)", "fxpo": "postcommissural fornix (left)", "mct": "medial corticohypothalamic tract (left)", "fx": "columns of the fornix (left)", "hc": "hippocampal commissures (left)", "dhc": "dorsal hippocampal commissure (left)", "vhc": "ventral hippocampal commissure (left)", "per": "perforant path (left)", "ab": "angular path (left)", "lab": "longitudinal association bundle (left)", "st": "stria terminalis (left)", "stc": "commissural branch of stria terminalis (left)", "mfsbshy": "hypothalamus related (left)", "mfb": "medial forebrain bundle (left)", "vlt": "ventrolateral hypothalamic tract (left)", "poc": "preoptic commissure (left)", "sup": "supraoptic commissures (left)", "supa": "supraoptic commissures anterior (left)", "supd": "supraoptic commissures dorsal (left)", "supv": "supraoptic commissures ventral (left)", "pmx": "premammillary commissure (left)", "smd": "supramammillary decussation (left)", "php": "propriohypothalamic pathways (left)", "phpd": "propriohypothalamic pathways dorsal (left)", "phpl": "propriohypothalamic pathways lateral (left)", "phpm": "propriohypothalamic pathways medial (left)", "phpv": "propriohypothalamic pathways ventral (left)", "pvbh": "periventricular bundle of the hypothalamus (left)", "mfbsma": "mammillary related (left)", "pm": "principal mammillary tract (left)", "mtt": "mammillothalamic tract (left)", "mtg": "mammillotegmental tract (left)", "mp": "mammillary peduncle (left)", "mfbst": "dorsal thalamus related (left)", "pvbt": "periventricular bundle of the thalamus (left)", "mfbse": "epithalamus related (left)", "sm": "stria medullaris (left)", "fr": "fasciculus retroflexus (left)", "hbc": "habenular commissure (left)", "pis": "pineal stalk (left)", "mfbsm": "midbrain related (left)", "dlf": "dorsal longitudinal fascicle (left)", "dtt": "dorsal tegmental tract (left)", "vs": "ventricular systems (left)", "vl": "lateral ventricle (left)", "rc": "rhinocele (left)", "sez": "subependymal zone (left)", "chpl": "choroid plexus (left)", "chfl": "choroid fissure (left)", "ivf": "interventricular foramen (left)", "v3": "third ventricle (left)", "aq": "cerebral aqueduct (left)", "v4": "fourth ventricle (left)", "v4r": "lateral recess (left)", "c": "central canal spinal cord/medulla (left)", "grv": "grooves (left)", "grv of ctx": "grooves of the cerebral cortex (left)", "eg": "endorhinal groove (left)", "hf": "hippocampal fissure (left)", "rf": "rhinal fissure (left)", "ri": "rhinal incisure (left)", "grv of cbx": "grooves of the cerebellar cortex (left)", "pce": "precentral fissure (left)", "pcf": "preculminate fissure (left)", "pri": "primary fissure (left)", "psf": "posterior superior fissure (left)", "ppf": "prepyramidal fissure (left)", "sec": "secondary fissure (left)", "plf": "posterolateral fissure (left)", "nf": "nodular fissure (left)", "sif": "simple fissure (left)", "icf": "intercrural fissure (left)", "apf": "ansoparamedian fissure (left)", "ipf": "interpeduncular fossa (left)", "pms": "paramedian sulcus (left)", "pfs": "parafloccular sulcus (left)", "retina": "retina (left)"
};


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



function isValidUUID(eid) {
    return regexExp.test(eid);
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
    if (unityCalled && unityTrial)
        unityTrial.SendMessage("main", "Stop");

    var tmpImg = new Image();
    tmpImg.onload = function () {
        document.getElementById(id).src = tmpImg.src;
        loading.style.visibility = "hidden";
        delete tmpImg;
        if (unityCalled && unityTrial)
            unityTrial.SendMessage("main", "Play");
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
    params.set("eid", CTX.eid);
    params.set("tid", CTX.tid);
    params.set("rid", CTX.rid);
    params.set("preprocess", CTX.preprocess);
    params.set("qc", CTX.qc);

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



function setupQC() {
    let qc = document.getElementById("qc-checkbox");
    qc.checked = CTX.qc == 1 ? true : false;
    qc.addEventListener("change", function (e) {
        CTX.qc = qc.checked ? 1 : 0;
        let url = getUrl();
        window.location.href = url;
    });
};



/*************************************************************************************************/
/*  Unity callback functions                                                                     */
/*************************************************************************************************/

/// UNITY loaded callback event, update the current highlighted probe
function unityLoaded() {
    if (unitySession)
        unitySession.SendMessage("main", "HighlightProbe", CTX.eid);
}



// UNITY callback
function selectEID(eid) {
    selectSession(eid);
    autoCompleteJS.setQuery(eid);
};

function unityUpdateQC() {
    if (unityTrial)
        window.unityTrial.SendMessage("FullPanels", "SwitchLayout", Number(CTX.qc));
}


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

function setupUnityTrial() {
    // Disable Unity widget on smartphones.
    if (isOnMobile() || !ENABLE_UNITY) return;

    // Trial viewer.
    createUnityInstance(document.querySelector("#unity-canvas-trial"), {
        dataUrl: "static/TrialViewerBuild/TrialViewer.data.gz",
        frameworkUrl: "static/TrialViewerBuild/TrialViewer.framework.js.gz",
        codeUrl: "static/TrialViewerBuild/TrialViewer.wasm.gz",
        streamingAssetsUrl: "StreamingAssets",
        companyName: "Daniel Birman @ UW",
        productName: "TrialViewer",
        productVersion: "0.1.0",
        // matchWebGLToCanvasSize: false, // Uncomment this to separately control WebGL canvas render size and DOM element size.
        // devicePixelRatio: 1, // Uncomment this to override low DPI rendering on high DPI displays.
    }).then((unityInstance) => {
        window.unityTrial = unityInstance;
        window.unityTrial.SendMessage("main", "SetSession", CTX.eid);
        window.unityTrial.SendMessage("main", "SetTrial", Number(CTX.tid));
        window.unityTrial.SendMessage("FullPanels", "SwitchLayout", Number(CTX.qc));
        unityUpdateQC();
    });
};



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
        selectTrial(CTX.eid, tid);
    }

    document.getElementById('trialPlot').onclick = clickTrial;
}



/*************************************************************************************************/
/*  Setup roi selection                                                                     */
/*************************************************************************************************/

function setupRoiDropdown(roi_ids, roi_id = -1) {
    // Set the cluster selector.
    var s = document.getElementById('roiSelector');
    $('#roiSelector option').remove();
    for (var i = 0; i < roi_ids.length; i++) {
        var rid = roi_ids[i];
        option = new Option(`roi ${rid}`, rid);
        if (((roi_id == -1) && (i == 0)) || (rid == roi_id))
            option.selected = true;
        s.options[s.options.length] = option;
    }
}


function setupRoiCallback() {
    // Cluster selector.
    document.getElementById('roiSelector').onchange = function (e) {
        var rid = e.target.value;
        if (!rid) return;
        selectRoi(CTX.eid, rid, CTX.preprocess);
    }
};

/*************************************************************************************************/
/*  Setup preprocess selection                                                                     */
/*************************************************************************************************/

function setupPreprocessDropdown(preprocess_ids, preprocess_id = -1) {
    // Set the cluster selector.
    var s = document.getElementById('preprocessSelector');
    $('#preprocessSelector option').remove();
    for (var i = 0; i < preprocess_ids.length; i++) {
        var preid = preprocess_ids[i];
        option = new Option(`${preid}`, preid);
        if (((preprocess_id == -1) && (preprocess == 0)) || (preid == preprocess_id))
            option.selected = true;
        s.options[s.options.length] = option;
    }
}


function setupPreprocessCallback() {
    // Cluster selector.
    document.getElementById('preprocessSelector').onchange = function (e) {
        var preprocess = e.target.value;
        if (!preprocess) return;
        selectPreprocess(CTX.eid, CTX.rid, preprocess);
    }
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

//function filterQuery(query_, Lab, Subject, pid, eid, acronyms, regions, _good_ids) {
function filterQuery(query_, Lab, Subject, eid, acronyms) {
    Lab = Lab.toLowerCase();
    Subject = Subject.toLowerCase();
    eid = eid.toLowerCase();

    // For a valid UUID: return yes if the query matches the current session's pid or eid.
    if (isValidUUID(query_)) {
        return eid.includes(query_);
    }

    // By default (no colon ":"), return all options.
    if (!query_.includes(":")) {
        return true
    }

    // Otherwise, we assume the query is of the form:
    [field, value] = query_.split(":");

    let _acronyms = Array.from(new Set(acronyms.map(a => a.toLowerCase())));

    // if (field == "region") return contains(query_, regions);
    if (field == "region") return contains(value, _acronyms, exact = true);
    if (field == "eid") return eid.includes(value);
    if (field == "lab") return Lab.includes(value);
    if (field == "subject") return Subject.includes(value);

    return false;
};



function getUniqueAcronyms(_acronyms, _good_ids) {
    // Remove duplicates in acronyms.
    _acronyms = _acronyms.map(a => a.toLowerCase());

    // Keep good acronyms if not QC mode.
    let acronyms = CTX.qc ? _acronyms : _acronyms.filter(filter_by_good, _good_ids);

    return Array.from(new Set(acronyms));
}

function acronymsToNames(acronyms) {
    return acronyms.map(acronym => ALLEN_REGIONS[acronym].toLowerCase());
}

function getSessionList() {

    let sessions = FLASK_CTX.SESSIONS;

    let out = sessions.filter(function (
        { Lab, Subject, eid, _acronyms}) {

        // Is the query a UUID?
        if (isValidUUID(query_)) {
            // If 1 session is already selected, show all of them.
            if (query_ == CTX.eid) return true;
            // Otherwise, show the corresponding session.
            return query_ == eid;
        }

        // Region acronyms and names.
        let acronyms = getUnique(_acronyms);

        // Filter on each term (space-separated).
        var res = true;
        for (let q of query_.split(/(\s+)/)) {
            res &= filterQuery(q, Lab, Subject, eid, acronyms);
        }

        return res;
    });

    // Update the mini brain viewer with the kept sessions.
    let eids = out.map(({ eid }) => eid);
    miniBrainActivateEIDs(eids);

    return out;
}



/*************************************************************************************************/
/*  Session selection                                                                            */
/*************************************************************************************************/

function loadAutoComplete() {
    autoCompleteJS = autocomplete({
        container: '#sessionSelector',
        placeholder: SESSION_SEARCH_PLACEHOLDER,
        openOnFocus: true,
        initialState: { query: CTX.eid },
        onStateChange({ state }) {
            var eid = state.query;

            // We only proceed if a new valid UUID has been selected.
            if (state.isOpen) return;
            if (!eid) return;
            if (eid == CTX.eid) return;
            if (!isValidUUID(eid)) return;
            //CTX.eid = eid;

            //selectSession(eid);
        },
        getSources({ query }) {
            query_ = query.toLowerCase();
            return [
                {
                    sourceId: 'sessions',
                    getItemInputValue: ({ item }) => item.eid,
                    getItems() {
                        return getSessionList();
                    },
                    templates: {
                        item({ item, html }) {
                            var acronyms = item['_acronyms']

                            acronyms = getUnique(acronyms);

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
                            <div class="item item-acronyms">${acronyms}</div>
                            <div class="item item-date">${item['Recording date']}</div>
                            <div class="item item-ID">${item.eid}</div>
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



function updateSessionPlot(eid, rid, preprocess) {
    showImage('sessionPlot', `/api/session/${eid}/session_plot/${rid}/${preprocess}`);
};



function updateBehaviourPlot(eid) {
    showImage('behaviourPlot', `/api/session/${eid}/behaviour_plot`);
};



async function selectSession(eid) {
    if (isLoading) return;
    if (!eid) return;
    isLoading = true;
    console.log("select session " + eid);

    if (unitySession)
        unitySession.SendMessage("main", "HighlightProbe", eid);

    if (unityTrial)
        unityTrial.SendMessage("main", "SetSession", eid);

    // Show the session details.
    var url = `/api/session/${eid}/details`;
    var r = await fetch(url);
    var details = await r.json();

    // Pop the cluster ids into a new variable

    // NOTE: these fields start with a leading _ so will be ignored by tablefromjson
    // which controls which fields are displayed in the session details box.
    var trial_ids = details['_trial_ids']
    var roi_ids = details['_roi_ids']
    var preprocess_ids = ['calcium', 'isosbestic_control', 'photobleach']

    CTX.dur = details["_duration"];
    CTX.trial_ids = trial_ids;
    CTX.trial_onsets = details["_trial_onsets"];
    CTX.trial_offsets = details["_trial_offsets"];

    // Make table with session details.
    fillVerticalTable(details, 'sessionDetails')


    // Setup the trial selector.
//    var trial_id = 0;
//    if (CTX.eid == eid && CTX.tid)
//        trial_id = CTX.tid;
//    setupTrialDropdown(trial_ids, trial_id);

    // Setup the roi selector.
    var roi_id = 0;
    if (CTX.eid == eid && CTX.rid)
        roi_id = CTX.rid;
    setupRoiDropdown(roi_ids, roi_id);

    // Setup the preprocess selector
    var preprocess_id = 'calcium';
    if (CTX.eid == eid && CTX.preprocess)
        preprocess_id = CTX.preprocess;
    setupPreprocessDropdown(preprocess_ids, preprocess_id);

    // Set the Roi and update plots
    selectRoi(eid, roi_id, preprocess_id)

    // Set the preprocess and update plots
    selectPreprocess(eid, roi_id, preprocess_id)

    // Show the behaviour overview plot.
    updateBehaviourPlot(eid);

    // Set the trial and update plots
    // selectTrial(eid, CTX.tid);

    CTX.eid = eid;
    isLoading = false;
};



/*************************************************************************************************/
/*  Unity mini brain                                                                             */
/*************************************************************************************************/

function miniBrainActivateEIDs(eidList) {
    // takes as input a list of PIDs and activates these
    if (unitySession) {
        unitySession.SendMessage("main", "DeactivateAllProbes");
        for (eid of eidList) {
            unitySession.SendMessage("main", "ActivateProbe", eid);
        }
    }
}



/*************************************************************************************************/
/*  Trial viewer                                                                                 */
/*************************************************************************************************/

function trialViewerLoaded() {
    if (unityTrial) {
        unityTrial.SendMessage("main", "SetSession", CTX.eid);
    }
}



function trialViewerDataLoaded() {
    // callback when the data finishes loading, expects to be sent back the current trial #
    if (unityTrial) {
        unityTrial.SendMessage("main", "SetTrial", CTX.tid);
    }
}



function updateTrialTime(time) {
    // png is 1200x500
    // left panel:  x: 80-540,   y: 60-420
    // right panel: x: 399-1004, y: 60-420
    // takes a float time and renders a red vertical line on the trial plot showing the current position
    var img = document.getElementById("trialPlot");

    var trial_id = CTX.tid;
    // note: this will work as long as trial_onsets/offsets contain all trials, including nan
    // ones, such that we can index them by trial_id.
    var t0 = CTX.trial_onsets[trial_id];
    var t1 = CTX.trial_offsets[trial_id];

    var perc = clamp((time - t0) / (t1 - t0), 0, 1);

    var w = img.width;
    // var h = img.height;
    var c = w / 1200.0;
    var x0 = 399 * c;
    var x1 = 1004 * c;
    var y0 = 60 * c;
    var y1 = 420 * c;

    var line = document.getElementById("trialTime");
    line.style.display = "block";
    line.style.left = x0 + perc * (x1 - x0) + "px";
    line.style.top = y0 + "px";
    line.style.height = (y1 - y0) + "px";

    // console.log(line.style.left);
};



// UNITY callback
function changeTrial(trialNum) {
    $('#trialSelector').val(trialNum);
    // var s = document.getElementById('trialSelector');
    // s.options[trialNum].selected = true;

    // trialNum will be the trial to jump to
    selectTrial(CTX.eid, trialNum, true);
};



function updateTrialPlot(eid, rid, preprocess) {
    showImage('trialRasterPlot', `/api/session/${eid}/trial_raster_plot/${rid}/${preprocess}`);
    showImage('trialPsthPlot', `/api/session/${eid}/trial_psth_plot/${rid}/${preprocess}`);
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
    selectTrial(CTX.eid, tid);
};



async function selectTrial(eid, tid, unityCalled = false) {
    CTX.tid = tid;

    if (unityTrial && !unityCalled)
        unityTrial.SendMessage("main", "SetTrial", Number(tid));

    // Show the trial raster plot.
    var url = `/api/session/${eid}/trial_plot/${tid}`;
    showImage('trialPlot', url, unityCalled);

    // Show information about trials in table
    var url = `/api/session/${eid}/trial_details/${tid}`;
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
    // // Show information about trials in table
    // var url = `/api/figures/details`;
    // var r = await fetch(url).then();
    // var details = await r.json();
    for (let letter in legends[key]) {
        // console.log(letter);
        // let [xmin, ymin, xmax, ymax] = details['cluster'][panel];
        // console.log(xmin, ymin, xmax, ymax);
        let panel = legends[key][letter];
        addPanelLetter(legend_id, plot, letter, panel["coords"], panel["legend"]);
    }

}



function setupAllLegends() {
    setupLegends('sessionPlot', 'sessionPlotLegend', 'figure1');
    setupLegends('behaviourPlot', 'behaviourPlotLegend', 'figure2');
    setupLegends('trialPlot', 'trialPlotLegend', 'figure3');
    setupLegends('trialRasterPlot', 'trialRasterPlotLegend', 'figure4');
}

/*************************************************************************************************/
/*  Roi selection                                                                            */
/*************************************************************************************************/

async function selectRoi(eid, rid, preprocess) {
    console.log(`select roi #${rid}`);
    CTX.rid = rid;

    // TODO eventually use updateSessionPlot
    var url = `/api/session/${eid}/session_plot/${rid}/${preprocess}`;
    showImage('sessionPlot', url);

    updateTrialPlot(eid, rid, preprocess)

    // Show information about trials in table
    var url = `/api/session/${eid}/roi_details/${rid}`;
    var r = await fetch(url).then();
    var details = await r.json();

    // Fill the trial details table.
    fillHorizontalTable(details, 'roiDetails')

};


/*************************************************************************************************/
/*  Preprocessing selection                                                                            */
/*************************************************************************************************/

async function selectPreprocess(eid, rid, preprocess) {
    console.log(`select preprocess ${preprocess}`);
    CTX.preprocess = preprocess;

    var url = `/api/session/${eid}/session_plot/${rid}/${preprocess}`;
    showImage('sessionPlot', url);

    updateTrialPlot(eid, rid, preprocess);


};

/*************************************************************************************************/
/*  Entry point                                                                                  */
/*************************************************************************************************/

function load() {
    setupShare();
    setupQC();

    loadAutoComplete();

    // setupUnitySession();
    // setupUnityTrial();

    // Remove for now until legends are settled
    //setupAllLegends();

    setupRoiCallback();
    setupPreprocessCallback();
    // setupTrialCallback();

    // Initial selection.
    selectSession(CTX.eid);
};



$(document).ready(function () {
    load();
});
