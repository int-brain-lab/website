
/*************************************************************************************************/
/*  Constants                                                                                    */
/*************************************************************************************************/

// Passing data from Flask to Javascript

const ENABLE_UNITY = true;   // disable for debugging

const regexExp = /^[0-9A-F]{8}-[0-9A-F]{4}-[4][0-9A-F]{3}-[89AB][0-9A-F]{3}-[0-9A-F]{12}$/i;
var unitySession = null; // unity instance for the session selector
var unityTrial = null; // unity instance for the trial viewer
var autoCompleteJS = null;
var isLoading = false;
var ALLEN_ACRONYMS = [
    "6b",
    "a13",
    "aaa",
    "ab",
    "aca",
    "aca1",
    "aca2/3",
    "aca5",
    "aca6a",
    "aca6b",
    "acad",
    "acad1",
    "acad2/3",
    "acad5",
    "acad6a",
    "acad6b",
    "acav",
    "acav1",
    "acav2/3",
    "acav5",
    "acav6a",
    "acav6b",
    "acb",
    "aco",
    "acs5",
    "act",
    "acvi",
    "acvii",
    "ad",
    "adp",
    "aha",
    "ahn",
    "ahna",
    "ahnc",
    "ahnd",
    "ahnp",
    "ai",
    "aid",
    "aid1",
    "aid2/3",
    "aid5",
    "aid6a",
    "aid6b",
    "aip",
    "aip1",
    "aip2/3",
    "aip5",
    "aip6a",
    "aip6b",
    "aiv",
    "aiv1",
    "aiv2/3",
    "aiv5",
    "aiv6a",
    "aiv6b",
    "alv",
    "am",
    "amb",
    "ambd",
    "ambv",
    "amc",
    "amd",
    "amv",
    "an",
    "ancr1",
    "ancr1gr",
    "ancr1mo",
    "ancr1pu",
    "ancr2",
    "ancr2gr",
    "ancr2mo",
    "ancr2pu",
    "aob",
    "aobgl",
    "aobgr",
    "aobmi",
    "aolt",
    "aon",
    "aon1",
    "aon2",
    "aond",
    "aone",
    "aonl",
    "aonm",
    "aonpv",
    "aot",
    "ap",
    "apd",
    "apf",
    "apn",
    "apr",
    "aq",
    "ar",
    "arb",
    "arh",
    "aso",
    "at",
    "atn",
    "aud",
    "audd",
    "audd1",
    "audd2/3",
    "audd4",
    "audd5",
    "audd6a",
    "audd6b",
    "audp",
    "audp1",
    "audp2/3",
    "audp4",
    "audp5",
    "audp6a",
    "audp6b",
    "audpo",
    "audpo1",
    "audpo2/3",
    "audpo4",
    "audpo5",
    "audpo6a",
    "audpo6b",
    "audv",
    "audv1",
    "audv2/3",
    "audv4",
    "audv5",
    "audv6a",
    "audv6b",
    "av",
    "avp",
    "avpv",
    "b",
    "ba",
    "bac",
    "bct",
    "bic",
    "bla",
    "blaa",
    "blap",
    "blav",
    "bma",
    "bmaa",
    "bmap",
    "bs",
    "bsc",
    "bst",
    "bsta",
    "bstal",
    "bstam",
    "bstd",
    "bstdm",
    "bstfu",
    "bstif",
    "bstju",
    "bstmg",
    "bstov",
    "bstp",
    "bstpr",
    "bstrh",
    "bstse",
    "bsttr",
    "bstv",
    "c",
    "ca",
    "ca1",
    "ca1slm",
    "ca1so",
    "ca1sp",
    "ca1sr",
    "ca2",
    "ca2slm",
    "ca2so",
    "ca2sp",
    "ca2sr",
    "ca3",
    "ca3slm",
    "ca3slu",
    "ca3so",
    "ca3sp",
    "ca3sr",
    "cb",
    "cbc",
    "cbf",
    "cbn",
    "cbp",
    "cbt",
    "cbx",
    "cbxgr",
    "cbxmo",
    "cbxpu",
    "cc",
    "ccb",
    "ccg",
    "ccr",
    "ccs",
    "cct",
    "cea",
    "ceac",
    "ceal",
    "ceam",
    "cent",
    "cent2",
    "cent2gr",
    "cent2mo",
    "cent2pu",
    "cent3",
    "cent3gr",
    "cent3mo",
    "cent3pu",
    "cett",
    "ch",
    "chfl",
    "chpl",
    "cic",
    "cing",
    "cl",
    "cla",
    "cli",
    "cm",
    "cm",
    "cn",
    "cnlam",
    "cnspg",
    "cnu",
    "coa",
    "coaa",
    "coaa1",
    "coaa2",
    "coaa3",
    "coap",
    "coapl",
    "coapl1",
    "coapl1-2",
    "coapl1-3",
    "coapl2",
    "coapl3",
    "coapm",
    "coapm1",
    "coapm1-2",
    "coapm1-3",
    "coapm2",
    "coapm3",
    "copy",
    "copygr",
    "copymo",
    "copypu",
    "cp",
    "cpd",
    "cpt",
    "crt",
    "cs",
    "csc",
    "csl",
    "csm",
    "cst",
    "cstc",
    "cstu",
    "ctb",
    "cte",
    "ctx",
    "ctxpl",
    "ctxsp",
    "cu",
    "cuf",
    "cul",
    "cul4",
    "cul4 5",
    "cul4 5gr",
    "cul4 5mo",
    "cul4 5pu",
    "cul4gr",
    "cul4mo",
    "cul4pu",
    "cul5",
    "cul5gr",
    "cul5mo",
    "cul5pu",
    "cun",
    "cvb",
    "cviiin",
    "das",
    "db",
    "dc",
    "dcm",
    "dcn",
    "dco",
    "dec",
    "decgr",
    "decmo",
    "decpu",
    "df",
    "dg",
    "dg-mo",
    "dg-po",
    "dg-sg",
    "dg-sgz",
    "dgcr",
    "dgcr-mo",
    "dgcr-po",
    "dgcr-sg",
    "dglb",
    "dglb-mo",
    "dglb-po",
    "dglb-sg",
    "dgmb",
    "dgmb-mo",
    "dgmb-po",
    "dgmb-sg",
    "dhc",
    "dl",
    "dlf",
    "dmh",
    "dmha",
    "dmhp",
    "dmhv",
    "dmx",
    "dn",
    "dorpm",
    "dorsm",
    "dp",
    "dp1",
    "dp2",
    "dp2/3",
    "dp5",
    "dp6a",
    "dr",
    "drt",
    "dscp",
    "dt",
    "dtd",
    "dtn",
    "dtt",
    "ec",
    "eco",
    "ect",
    "ect1",
    "ect2/3",
    "ect5",
    "ect6a",
    "ect6b",
    "ecu",
    "ee",
    "eg",
    "em",
    "ent",
    "entl",
    "entl1",
    "entl2",
    "entl2/3",
    "entl2a",
    "entl2b",
    "entl3",
    "entl4",
    "entl4/5",
    "entl5",
    "entl5/6",
    "entl6a",
    "entl6b",
    "entm",
    "entm1",
    "entm2",
    "entm2a",
    "entm2b",
    "entm3",
    "entm4",
    "entm5",
    "entm5/6",
    "entm6",
    "entmv",
    "entmv1",
    "entmv2",
    "entmv3",
    "entmv4",
    "entmv5/6",
    "ep",
    "epd",
    "epi",
    "eps",
    "epsc",
    "epv",
    "eth",
    "ev",
    "ew",
    "fa",
    "fc",
    "ff",
    "fi",
    "fiber tracts",
    "fl",
    "flgr",
    "flmo",
    "flpu",
    "fn",
    "fotu",
    "fotugr",
    "fotumo",
    "fotupu",
    "fp",
    "fpr",
    "fr",
    "frp",
    "frp1",
    "frp2/3",
    "frp5",
    "frp6a",
    "frp6b",
    "fs",
    "fx",
    "fxpo",
    "fxprg",
    "fxs",
    "gend",
    "genv",
    "gpe",
    "gpi",
    "gr",
    "grey",
    "grf",
    "grn",
    "grv",
    "grv of cbx",
    "grv of ctx",
    "gu",
    "gu1",
    "gu2/3",
    "gu4",
    "gu5",
    "gu6a",
    "gu6b",
    "gviin",
    "hata",
    "hb",
    "hbc",
    "hc",
    "hem",
    "hf",
    "hht",
    "hip",
    "hpf",
    "hy",
    "i5",
    "ia",
    "iad",
    "iaf",
    "iam",
    "ias",
    "ib",
    "ic",
    "icb",
    "icc",
    "icd",
    "ice",
    "icf",
    "icp",
    "if",
    "ig",
    "igl",
    "iii",
    "iiin",
    "iin",
    "ila",
    "ila1",
    "ila2",
    "ila2/3",
    "ila5",
    "ila6a",
    "ila6b",
    "ilm",
    "im",
    "imd",
    "in",
    "inc",
    "inco",
    "int",
    "intg",
    "inv",
    "io",
    "ip",
    "ipa",
    "ipc",
    "ipdl",
    "ipdm",
    "ipf",
    "ipf",
    "ipi",
    "ipl",
    "ipn",
    "ipr",
    "iprl",
    "irn",
    "isl",
    "islm",
    "isn",
    "isocortex",
    "iv",
    "ivd",
    "ivf",
    "iviin",
    "ivn",
    "ixn",
    "jrb",
    "kf",
    "la",
    "lab",
    "lat",
    "lav",
    "lc",
    "ld",
    "ldt",
    "lfbs",
    "lfbst",
    "lgd",
    "lgd-co",
    "lgd-ip",
    "lgd-sh",
    "lgv",
    "lgvl",
    "lgvm",
    "lh",
    "lha",
    "lin",
    "ling",
    "linggr",
    "lingmo",
    "lingpu",
    "ll",
    "lm",
    "lot",
    "lotd",
    "lotg",
    "lp",
    "lpo",
    "lrn",
    "lrnm",
    "lrnp",
    "ls",
    "lsc",
    "lsr",
    "lss",
    "lsv",
    "lsx",
    "lt",
    "ltn",
    "lz",
    "ma",
    "ma3",
    "marn",
    "mb",
    "mbmot",
    "mbo",
    "mbsen",
    "mbsta",
    "mcp",
    "mct",
    "md",
    "mdc",
    "mdl",
    "mdm",
    "mdrn",
    "mdrnd",
    "mdrnv",
    "me",
    "mea",
    "meaad",
    "meaav",
    "meapd",
    "meapd-a",
    "meapd-b",
    "meapd-c",
    "meapv",
    "med",
    "mepo",
    "mev",
    "mez",
    "mfb",
    "mfbc",
    "mfbs",
    "mfbse",
    "mfbsm",
    "mfbsma",
    "mfbst",
    "mfsbshy",
    "mg",
    "mgd",
    "mgm",
    "mgv",
    "mh",
    "ml",
    "mlf",
    "mm",
    "mmd",
    "mml",
    "mmm",
    "mmme",
    "mmp",
    "mo",
    "mo1",
    "mo2/3",
    "mo5",
    "mo6a",
    "mo6b",
    "mob",
    "mobgl",
    "mobgr",
    "mobipl",
    "mobmi",
    "mobopl",
    "mop",
    "mop1",
    "mop2/3",
    "mop5",
    "mop6a",
    "mop6b",
    "mos",
    "mos1",
    "mos2/3",
    "mos5",
    "mos6a",
    "mos6b",
    "mov",
    "mp",
    "mpn",
    "mpnc",
    "mpnl",
    "mpnm",
    "mpo",
    "mpt",
    "mrn",
    "mrnm",
    "mrnmg",
    "mrnp",
    "ms",
    "msc",
    "mt",
    "mtc",
    "mtg",
    "mtn",
    "mtt",
    "mtv",
    "mv",
    "my",
    "my-mot",
    "my-sat",
    "my-sen",
    "nb",
    "nc",
    "nd",
    "ndb",
    "nf",
    "ni",
    "nis",
    "nll",
    "nlld",
    "nllh",
    "nllv",
    "nlot",
    "nlot1",
    "nlot1-3",
    "nlot2",
    "nlot3",
    "nod",
    "nodgr",
    "nodmo",
    "nodpu",
    "not",
    "npc",
    "nr",
    "nst",
    "ntb",
    "nts",
    "ntsce",
    "ntsco",
    "ntsge",
    "ntsl",
    "ntsm",
    "ntt",
    "och",
    "oct",
    "olf",
    "onl",
    "op",
    "opt",
    "or",
    "orb",
    "orb1",
    "orb2/3",
    "orb5",
    "orb6a",
    "orb6b",
    "orbl",
    "orbl1",
    "orbl2/3",
    "orbl5",
    "orbl6a",
    "orbl6b",
    "orbm",
    "orbm1",
    "orbm2",
    "orbm2/3",
    "orbm5",
    "orbm6a",
    "orbm6b",
    "orbv",
    "orbvl",
    "orbvl1",
    "orbvl2/3",
    "orbvl5",
    "orbvl6a",
    "orbvl6b",
    "ot",
    "ot1",
    "ot1-3",
    "ot2",
    "ot3",
    "ov",
    "p",
    "p-mot",
    "p-sat",
    "p-sen",
    "p5",
    "pa",
    "pa4",
    "pa5",
    "paa",
    "paa1",
    "paa1-3",
    "paa2",
    "paa3",
    "pag",
    "pal",
    "palc",
    "pald",
    "palm",
    "palv",
    "pap",
    "par",
    "par1",
    "par2",
    "par3",
    "parn",
    "pas",
    "pb",
    "pbg",
    "pbl",
    "pblc",
    "pbld",
    "pble",
    "pbls",
    "pblv",
    "pbm",
    "pbme",
    "pbmm",
    "pbmv",
    "pc",
    "pc5",
    "pce",
    "pcf",
    "pcg",
    "pcn",
    "pd",
    "pdtg",
    "pef",
    "per",
    "peri",
    "peri1",
    "peri2/3",
    "peri5",
    "peri6a",
    "peri6b",
    "pf",
    "pfl",
    "pflgr",
    "pflmo",
    "pflpu",
    "pfs",
    "pg",
    "pgrn",
    "pgrnd",
    "pgrnl",
    "ph",
    "php",
    "phpd",
    "phpl",
    "phpm",
    "phpv",
    "phy",
    "pil",
    "pin",
    "pir",
    "pir1",
    "pir1-3",
    "pir2",
    "pir3",
    "pis",
    "pl",
    "pl1",
    "pl2",
    "pl2/3",
    "pl5",
    "pl6a",
    "pl6b",
    "plf",
    "pm",
    "pmd",
    "pmr",
    "pms",
    "pmv",
    "pmx",
    "pn",
    "po",
    "poc",
    "pol",
    "por",
    "post",
    "post1",
    "post2",
    "post3",
    "pot",
    "pp",
    "ppf",
    "ppn",
    "ppt",
    "ppy",
    "ppyd",
    "ppys",
    "pr",
    "prc",
    "pre",
    "pre1",
    "pre2",
    "pre3",
    "pri",
    "prm",
    "prmgr",
    "prmmo",
    "prmpu",
    "prnc",
    "prnr",
    "prnv",
    "pros",
    "prosd",
    "prosd-m",
    "prosd-sp",
    "prosd-sr",
    "prosv",
    "prosv-m",
    "prosv-sp",
    "prosv-sr",
    "prp",
    "prt",
    "ps",
    "psch",
    "psf",
    "pst",
    "pstn",
    "psv",
    "pt",
    "ptf",
    "ptlp",
    "ptlp1",
    "ptlp2/3",
    "ptlp4",
    "ptlp5",
    "ptlp6a",
    "ptlp6b",
    "pva",
    "pvbh",
    "pvbt",
    "pvh",
    "pvham",
    "pvhap",
    "pvhd",
    "pvhdp",
    "pvhf",
    "pvhlp",
    "pvhm",
    "pvhmm",
    "pvhmpd",
    "pvhmpv",
    "pvhp",
    "pvhpm",
    "pvhpml",
    "pvhpmm",
    "pvhpv",
    "pvi",
    "pvp",
    "pvpo",
    "pvr",
    "pvt",
    "pvz",
    "py",
    "pyd",
    "pyr",
    "pyrgr",
    "pyrmo",
    "pyrpu",
    "ramb",
    "rc",
    "rch",
    "rct",
    "re",
    "reth",
    "retina",
    "rf",
    "rh",
    "rhp",
    "ri",
    "rl",
    "rm",
    "rn",
    "ro",
    "rpa",
    "rpf",
    "rpo",
    "rr",
    "rrt",
    "rsp",
    "rspagl",
    "rspagl1",
    "rspagl2/3",
    "rspagl5",
    "rspagl6a",
    "rspagl6b",
    "rspd",
    "rspd1",
    "rspd2/3",
    "rspd4",
    "rspd5",
    "rspd6a",
    "rspd6b",
    "rspv",
    "rspv1",
    "rspv2",
    "rspv2/3",
    "rspv5",
    "rspv6a",
    "rspv6b",
    "rst",
    "rstl",
    "rstm",
    "rt",
    "rust",
    "sag",
    "samy",
    "sbpv",
    "scdg",
    "scdw",
    "sch",
    "scig",
    "scig-a",
    "scig-b",
    "scig-c",
    "sciw",
    "scm",
    "sco",
    "scop",
    "scp",
    "scrt",
    "scs",
    "scsg",
    "sct",
    "sctd",
    "sctv",
    "scwm",
    "sczo",
    "sec",
    "sez",
    "sf",
    "sfo",
    "sg",
    "sgn",
    "sh",
    "shp",
    "si",
    "sif",
    "sim",
    "simgr",
    "simmo",
    "simpu",
    "slc",
    "sld",
    "sm",
    "smd",
    "smt",
    "snc",
    "snl",
    "snp",
    "snr",
    "so",
    "soc",
    "socl",
    "socm",
    "sop",
    "spa",
    "spf",
    "spfm",
    "spfp",
    "spiv",
    "sptv",
    "spvc",
    "spvi",
    "spvo",
    "spvocdm",
    "spvomdmd",
    "spvomdmv",
    "spvordm",
    "spvovl",
    "srp",
    "ss",
    "ss1",
    "ss2/3",
    "ss4",
    "ss5",
    "ss6a",
    "ss6b",
    "ssn",
    "ssp",
    "ssp-bfd",
    "ssp-bfd1",
    "ssp-bfd2/3",
    "ssp-bfd4",
    "ssp-bfd5",
    "ssp-bfd6a",
    "ssp-bfd6b",
    "ssp-ll",
    "ssp-ll1",
    "ssp-ll2/3",
    "ssp-ll4",
    "ssp-ll5",
    "ssp-ll6a",
    "ssp-ll6b",
    "ssp-m",
    "ssp-m1",
    "ssp-m2/3",
    "ssp-m4",
    "ssp-m5",
    "ssp-m6a",
    "ssp-m6b",
    "ssp-n",
    "ssp-n1",
    "ssp-n2/3",
    "ssp-n4",
    "ssp-n5",
    "ssp-n6a",
    "ssp-n6b",
    "ssp-tr",
    "ssp-tr1",
    "ssp-tr2/3",
    "ssp-tr4",
    "ssp-tr5",
    "ssp-tr6a",
    "ssp-tr6b",
    "ssp-ul",
    "ssp-ul1",
    "ssp-ul2/3",
    "ssp-ul4",
    "ssp-ul5",
    "ssp-ul6a",
    "ssp-ul6b",
    "ssp-un",
    "ssp-un1",
    "ssp-un2/3",
    "ssp-un4",
    "ssp-un5",
    "ssp-un6a",
    "ssp-un6b",
    "ssp1",
    "ssp2/3",
    "ssp4",
    "ssp5",
    "ssp6a",
    "ssp6b",
    "sss",
    "sss1",
    "sss2/3",
    "sss4",
    "sss5",
    "sss6a",
    "sss6b",
    "sst",
    "st",
    "stc",
    "step",
    "stf",
    "stn",
    "stp",
    "str",
    "strd",
    "strv",
    "sttl",
    "sttv",
    "su3",
    "sub",
    "subd",
    "subd-m",
    "subd-sp",
    "subd-sr",
    "subg",
    "subv",
    "subv-m",
    "subv-sp",
    "subv-sr",
    "sum",
    "suml",
    "summ",
    "sup",
    "supa",
    "supd",
    "supv",
    "sut",
    "suv",
    "sv",
    "svp",
    "tb",
    "tct",
    "tea",
    "tea1",
    "tea2/3",
    "tea4",
    "tea5",
    "tea6a",
    "tea6b",
    "th",
    "tm",
    "tmd",
    "tmv",
    "tn",
    "tp",
    "tr",
    "tr1",
    "tr1-3",
    "tr2",
    "tr3",
    "trn",
    "trs",
    "ts",
    "tsp",
    "tspc",
    "tspd",
    "tt",
    "ttd",
    "ttd1",
    "ttd1-4",
    "ttd2",
    "ttd3",
    "ttd4",
    "ttp",
    "ttv",
    "ttv1",
    "ttv1-3",
    "ttv2",
    "ttv3",
    "tu",
    "uf",
    "uvu",
    "uvugr",
    "uvumo",
    "uvupu",
    "v",
    "v3",
    "v4",
    "v4r",
    "val",
    "vc",
    "vco",
    "vecb",
    "vent",
    "verm",
    "vhc",
    "vi",
    "vii",
    "viiin",
    "viin",
    "vin",
    "vis",
    "vis1",
    "vis2/3",
    "vis4",
    "vis5",
    "vis6a",
    "vis6b",
    "visa",
    "visa1",
    "visa2/3",
    "visa4",
    "visa5",
    "visa6a",
    "visa6b",
    "visal",
    "visal1",
    "visal2/3",
    "visal4",
    "visal5",
    "visal6a",
    "visal6b",
    "visam",
    "visam1",
    "visam2/3",
    "visam4",
    "visam5",
    "visam6a",
    "visam6b",
    "visc",
    "visc1",
    "visc2/3",
    "visc4",
    "visc5",
    "visc6a",
    "visc6b",
    "visl",
    "visl1",
    "visl2/3",
    "visl4",
    "visl5",
    "visl6a",
    "visl6b",
    "visli",
    "visli1",
    "visli2/3",
    "visli4",
    "visli5",
    "visli6a",
    "visli6b",
    "vislla",
    "vislla1",
    "vislla2/3",
    "vislla4",
    "vislla5",
    "vislla6a",
    "vislla6b",
    "vism",
    "vism1",
    "vism2/3",
    "vism4",
    "vism5",
    "vism6a",
    "vism6b",
    "vismma",
    "vismma1",
    "vismma2/3",
    "vismma4",
    "vismma5",
    "vismma6a",
    "vismma6b",
    "vismmp",
    "vismmp1",
    "vismmp2/3",
    "vismmp4",
    "vismmp5",
    "vismmp6a",
    "vismmp6b",
    "visp",
    "visp1",
    "visp2/3",
    "visp4",
    "visp5",
    "visp6a",
    "visp6b",
    "vispl",
    "vispl1",
    "vispl2/3",
    "vispl4",
    "vispl5",
    "vispl6a",
    "vispl6b",
    "vispm",
    "vispm1",
    "vispm2/3",
    "vispm4",
    "vispm5",
    "vispm6a",
    "vispm6b",
    "vispor",
    "vispor1",
    "vispor2/3",
    "vispor4",
    "vispor5",
    "vispor6a",
    "vispor6b",
    "visrl",
    "visrl1",
    "visrl2/3",
    "visrl4",
    "visrl5",
    "visrl6a",
    "visrl6b",
    "visrll",
    "visrll1",
    "visrll2/3",
    "visrll4",
    "visrll5",
    "visrll6a",
    "visrll6b",
    "vl",
    "vlpo",
    "vlt",
    "vm",
    "vmh",
    "vmha",
    "vmhc",
    "vmhdm",
    "vmhvl",
    "vmpo",
    "vn",
    "vnc",
    "von",
    "vp",
    "vpl",
    "vplpc",
    "vpm",
    "vpmpc",
    "vrt",
    "vs",
    "vsp",
    "vta",
    "vtd",
    "vtn",
    "vviiin",
    "xi",
    "xii",
    "xiin",
    "xin",
    "xn",
    "zi",
];


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
    params.set("pid", CTX.pid);
    params.set("tid", CTX.tid);
    params.set("cid", CTX.cid);
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
        unitySession.SendMessage("main", "HighlightProbe", CTX.pid);
}



// UNITY callback
function selectPID(pid) {
    selectSession(pid);
    autoCompleteJS.setQuery(pid);
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
        window.unityTrial.SendMessage("main", "SetSession", CTX.pid);
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
        console.log("click cluster", e);
        onClusterClick(canvas, e);
    }, true);
};



/*************************************************************************************************/
/*  Session selection                                                                            */
/*************************************************************************************************/

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



function filterQuery(query_, Lab, Subject, ID, _acronyms, _regions, _good_ids) {
    // For a valid UUID: return it.
    if (isValidUUID(query_)) {
        return ID.toLowerCase().includes(query_);
    }

    // Exact Allen region match: exact match on the acronyms.
    if (ALLEN_ACRONYMS.includes(query_)) {
        // Keep good acronyms if not QC mode.
        let acronyms = CTX.qc ? _acronyms : _acronyms.filter(filter_by_good, _good_ids);
        let boo = acronyms.includes(query_);
        return boo;
    }

    // Keep good regions if not QC mode.
    _regions = _regions.split(", ");
    let regions = CTX.qc ? _regions : _regions.filter(filter_by_good, _good_ids);
    regions = regions.join(", ");

    // Search on lab, subject, regions.
    let out = Lab.toLowerCase().includes(query_) ||
        Subject.toLowerCase().includes(query_) ||
        regions.includes(query_);

    return out;
};



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
        placeholder: 'search for session',
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
                    getItemInputValue: ({ item }) => item.ID,
                    getItems() {
                        let sessions = FLASK_CTX.SESSIONS;

                        let out = sessions.filter(function (
                            { Lab, Subject, ID, _acronyms, _regions, _good_ids, dset_bwm, dset_rs }) {

                            // NOTE: remove duplicates in acronyms.
                            _acronyms = Array.from(new Set(_acronyms));
                            _acronyms = _acronyms.map(a => a.toLowerCase());

                            var res = true;

                            // If 1 session is already selected, show all of them.
                            if (!isValidUUID(query_) || query_ != CTX.pid) {
                                for (let q of query_.split(/(\s+)/)) {
                                    res &= filterQuery(q, Lab, Subject, ID, _acronyms, _regions, _good_ids);
                                }
                            }

                            // Dataset selection
                            if (CTX.dset == 'bwm')
                                res &= dset_bwm;
                            if (CTX.dset == 'rs')
                                res &= dset_rs;

                            return res;
                        });
                        let pids = out.map(({ ID }) => ID);
                        miniBrainActivatePIDs(pids);
                        return out;
                    },
                    templates: {
                        item({ item, html }) {
                            var good_idx = item['_good_ids'];
                            if (CTX.qc) {
                                var acronyms = item['_acronyms'];
                            } else {
                                var acronyms = item["_acronyms"].filter(filter_by_good, good_idx);
                            }

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
                            <div class="item item-ID">${item.ID}</div>
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

    if (unityTrial)
        unityTrial.SendMessage("main", "SetSession", pid);

    // Show the session details.
    var url = `/api/session/${pid}/details`;
    var r = await fetch(url);
    var details = await r.json();

    // Pop the cluster ids into a new variable

    // NOTE: these fields start with a leading _ so will be ignored by tablefromjson
    // which controls which fields are displayed in the session details box.
    var trial_ids = details['_trial_ids']
    var good_idx = details["_good_ids"];

    if (CTX.qc) {
        var cluster_ids = details["_cluster_ids"];
        var acronyms = details["_acronyms"];
        var colors = details["_colors"];
    }
    else {
        var cluster_ids = details["_cluster_ids"].filter(filter_by_good, good_idx);
        var acronyms = details["_acronyms"].filter(filter_by_good, good_idx);
        var colors = details["_colors"].filter(filter_by_good, good_idx);
    }


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
    var trial_id = 0;
    if (CTX.pid == pid && CTX.tid)
        trial_id = CTX.tid;
    setupTrialDropdown(trial_ids, trial_id);

    // Setup the cluster selector.
    setupClusterDropdown(cluster_ids, acronyms, colors, cluster_id = CTX.cid);

    // Update the other plots.
    selectTrial(pid, CTX.tid);

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
/*  Trial viewer                                                                                 */
/*************************************************************************************************/

function trialViewerLoaded() {
    if (unityTrial) {
        unityTrial.SendMessage("main", "SetSession", CTX.pid);
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
    selectTrial(CTX.pid, trialNum, true);
};



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

    if (unityTrial && !unityCalled)
        unityTrial.SendMessage("main", "SetTrial", Number(tid));

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
    setupLegends('trialEventPlot', 'trialEventPlotLegend', 'figure4');
    if (CTX.qc) {
        setupLegends('clusterPlot', 'clusterPlotLegend', 'figure5_qc');
    } else {
        setupLegends('clusterPlot', 'clusterPlotLegend', 'figure5');
    }
}



/*************************************************************************************************/
/*  Cluster selection                                                                            */
/*************************************************************************************************/

async function onClusterClick(canvas, event) {
    const rect = canvas.getBoundingClientRect()
    const x = (event.clientX - rect.left) / rect.width
    const y = Math.abs((event.clientY - rect.bottom)) / rect.height
    var url = `/api/session/${CTX.pid}/cluster_plot_from_xy/${CTX.cid}/${x}_${y}/${Number(CTX.qc)}`;
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

    if (CTX.qc) {
        var url = `/api/session/${pid}/cluster_qc_plot/${cid}`;
    } else {
        var url = `/api/session/${pid}/cluster_plot/${cid}`;
    }
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
    setupQC();

    setupDataset();
    loadAutoComplete();

    setupUnitySession();
    setupUnityTrial();

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
