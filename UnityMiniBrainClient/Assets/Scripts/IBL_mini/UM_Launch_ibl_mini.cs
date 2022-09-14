using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;

public class UM_Launch_ibl_mini : MonoBehaviour
{
    [DllImport("__Internal")]
    private static extern void SelectPID(string pid);
    [DllImport("__Internal")]
    private static extern void SelectCluster(int cluster);

    //[SerializeField] private CCFModelControl modelControl;
    [SerializeField] private BrainCameraController cameraController;
    [SerializeField] private UM_CameraController umCamera;

    [SerializeField] private GameObject probeLinePrefab;
    [SerializeField] private Transform probeParentT;
    [SerializeField] private TextAsset probeData;

    [SerializeField] private InfoText infoText;

    [SerializeField] private bool loadDefaults;

    private Vector3 center = new Vector3(5.7f, 4f, -6.6f);

    // Neuron materials
    [SerializeField] private Dictionary<string, Material> neuronMaterials;

    //private Dictionary<int, CCFTreeNode> visibleNodes;

    private bool ccfLoaded;

    [SerializeField] private List<Color> colors;
    [SerializeField] private Color defaultColor = Color.gray;

    [SerializeField] private List<Color> brainColors;
    [SerializeField] private List<GameObject> brainAreas;

    private Dictionary<string, GameObject> pid2probe;
    private Dictionary<GameObject, string> probe2pid;
    private Dictionary<GameObject, (int, string, string)> probeInfo;
    private string[] labs = {"angelakilab", "churchlandlab", "churchlandlab_ucla", "cortexlab",
       "danlab", "hoferlab", "mainenlab", "mrsicflogellab",
       "steinmetzlab", "wittenlab", "zadorlab" };
    private Dictionary<string, Color> labColors;

    private GameObject highlightedProbe;
    private GameObject hoveredProbe;

    private string serverTarget;

    private void Awake()
    {
        Debug.Log("v0.1.0");

        pid2probe = new Dictionary<string, GameObject>();
        probeInfo = new Dictionary<GameObject, (int, string, string)>();
        probe2pid = new Dictionary<GameObject, string>();

        //visibleNodes = new Dictionary<int, CCFTreeNode>();

        labColors = new Dictionary<string, Color>();
        for (int i = 0; i < labs.Length; i++)
            labColors.Add(labs[i], colors[i]);

        for (int i = 0; i < brainAreas.Count; i++)
        {
            brainAreas[i].GetComponentInChildren<Renderer>().material.color = brainColors[i];
        }

#if !UNITY_EDITOR && UNITY_WEBGL
        // disable WebGLInput.captureAllKeyboardInput so elements in web page can handle keyboard inputs
        WebGLInput.captureAllKeyboardInput = false;
#endif
    }

    // Start is called before the first frame update
    void Start()
    {        
        cameraController.SetBrainAxisAngles(new Vector3(0f, 45f, 135f));
        umCamera.SwitchCameraMode(false);

        LoadProbes();
    }

    // Update is called once per frame
    void Update()
    {
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;
        if (Physics.Raycast(ray, out hit))
        {

            //Select stage    
            if (hit.transform.gameObject.layer == 10)
            {
                GameObject target = hit.transform.gameObject;
                if (Input.GetMouseButtonDown(0))
                    SelectProbe(target);
                else
                {
                    if (target != null && target != hoveredProbe)
                    {
                        UnhoverProbe(hoveredProbe);
                        HoverProbe(hit.transform.gameObject);
                    }
                }
            }
            else
            {
                UnhoverProbe(hoveredProbe);
            }
        }
        else
        {
            UnhoverProbe(hoveredProbe);
        }
    }

    private void SetProbePositionAndAngles(Transform probeT, Vector3 pos, Vector3 angles)
    {
        // reset position and angles
        probeT.transform.localPosition = Vector3.zero;
        probeT.localRotation = Quaternion.identity;

        // then translate
        probeT.Translate(new Vector3(-pos.x / 1000f, -pos.z / 1000f, pos.y / 1000f));
        // rotate around azimuth first
        probeT.RotateAround(probeT.position, Vector3.up, -angles.x - 90f);
        // then elevation
        probeT.RotateAround(probeT.position, probeT.right, angles.y);
        // then spin
        probeT.RotateAround(probeT.position, probeT.up, angles.z);

        if (angles.z < 3840)
        {
            //resize the probe object
            Vector3 scale = probeT.GetChild(0).transform.localScale;
            scale.y = angles.z / 1000f;
            probeT.GetChild(0).transform.localScale = scale;
            probeT.Translate(probeT.up * -(scale.y / 2f));
        }
    }

    private void LoadProbes()
    {
        List<Dictionary<string,object>> data = CSVReader.ParseText(probeData.text);

        foreach (Dictionary<string,object> row in data)
        {
            GameObject newProbe = Instantiate(probeLinePrefab, probeParentT);
            string pid = (string)row["pid"];
            string lab = (string)row["lab"];
            string subject = (string)row["subject"];
            string date = (string)row["date"];
            string selectable = (string)row["selectable"];

            int labIdx = 0;
            for (int i = 0; i < labs.Length; i++)
                if (labs[i].Equals(lab))
                {
                    labIdx = i;
                    break;
                }

            Vector3 pos = new Vector3((float)row["ml_ccf_tip"], (float)row["ap_ccf_tip"], (float)row["dv_ccf_tip"]);
            Vector3 angles = new Vector3((float)row["phi"], (float)row["theta"], (float)row["depth"]);

            newProbe.GetComponentInChildren<BoxCollider>().enabled = false;

            pid2probe.Add(pid, newProbe);
            probe2pid.Add(newProbe, pid);
            probeInfo.Add(newProbe, (labIdx, subject, date));
            SetProbePositionAndAngles(newProbe.transform, pos, angles);

            if (selectable.Equals("TRUE"))
                ActivateProbe(pid);
        }
    }

    public void ActivateProbe(string pid)
    {
        Debug.Log("Activate: " + pid);
        pid2probe[pid].GetComponentInChildren<Renderer>().material.color = defaultColor;
        //pid2probe[pid].GetComponentInChildren<Renderer>().material.SetColor("_Color", colors[probeLabs[pid]]);
        pid2probe[pid].GetComponentInChildren<BoxCollider>().enabled = true;
        pid2probe[pid].GetComponentInChildren<BoxCollider>().isTrigger = true;
        // make it bigger
        pid2probe[pid].transform.localScale = new Vector3(5f, 1f, 5f);
    }

    public void DeactivateProbe(string pid)
    {
        pid2probe[pid].GetComponentInChildren<Renderer>().material.SetColor("Color", Color.white);
        pid2probe[pid].transform.localScale = Vector3.one;
    }

    public void HoverProbe(GameObject probe)
    {
        probe.GetComponent<ProbeComponent>().SetTrackActive(true);
        probe.GetComponent<ProbeComponent>().SetTrackHighlight(true);
        if (!highlightedProbe==probe)
            probe.GetComponent<Renderer>().material.color = Color.red;
        hoveredProbe = probe;
    }

    public void UnhoverProbe(GameObject probe)
    {
        if (probe != null)
        {
            probe.GetComponent<ProbeComponent>().SetTrackActive(false);
            probe.GetComponent<ProbeComponent>().SetTrackHighlight(false);
            if (!highlightedProbe==probe)
                probe.GetComponent<Renderer>().material.color = defaultColor;
        }
        hoveredProbe = null;
    }

    public void HighlightProbe(string pid) {
        Debug.Log("Highlighting pid: " + pid);
        HighlightProbeGO(pid2probe[pid]);
    }

    public void HighlightProbeGO(GameObject probe) {
        UnhighlightProbe();

        highlightedProbe = probe;

        // also set the lab information
        (int labIdx, string subj, string date) = probeInfo[probe.transform.parent.gameObject];
        string lab = labs[labIdx];
        infoText.SetText(lab, subj, date, labColors[lab]);

        probe.GetComponent<Renderer>().material.color = labColors[lab];
    }

    public void UnhighlightProbe() {
        if (highlightedProbe != null) {
            highlightedProbe.GetComponent<Renderer>().material.color = defaultColor;
        }
    }

    public void SelectProbe(GameObject probe)
    {
        string pid = probe2pid[probe.transform.parent.gameObject];
        HighlightProbeGO(probe);
#if !UNITY_EDITOR && UNITY_WEBGL
        SelectPID(pid);
#endif
        Debug.Log("Sent select message with payload: " + pid);
    }

    public void SelectCluster(GameObject cluster) {
        Debug.Log("not implemented");
    }
}
