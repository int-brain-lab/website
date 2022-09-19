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

    [SerializeField] private bool loadDefaults;

    // Neuron materials
    [SerializeField] private Dictionary<string, Material> neuronMaterials;

    [SerializeField] private List<Color> colors;
    [SerializeField] private Color defaultColor = Color.gray;

    [SerializeField] private List<Color> brainColors;
    [SerializeField] private List<GameObject> brainAreas;

    private Dictionary<string, GameObject> pid2probe;
    private string[] labs = {"angelakilab", "churchlandlab", "churchlandlab_ucla", "cortexlab",
       "danlab", "hoferlab", "mainenlab", "mrsicflogellab",
       "steinmetzlab", "wittenlab", "zadorlab" };
    private Dictionary<string, Color> labColors;

    private GameObject highlightedProbe;
    private GameObject hoveredProbe;

    private string serverTarget;

    bool IsMouseOverGameWindow { get { return !(0 > Input.mousePosition.x || 0 > Input.mousePosition.y || Screen.width < Input.mousePosition.x || Screen.height < Input.mousePosition.y); } }

    private void Awake()
    {
        Debug.Log("v0.1.0");

        pid2probe = new Dictionary<string, GameObject>();

        labColors = new Dictionary<string, Color>();
        for (int i = 0; i < labs.Length; i++)
            labColors.Add(labs[i], colors[i]);

        for (int i = 0; i < brainAreas.Count; i++)
        {
            brainAreas[i].GetComponentInChildren<Renderer>().material.color = brainColors[i];
        }

        LoadProbes();

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

    }

    // Update is called once per frame
    void Update()
    {
        if (IsMouseOverGameWindow)
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
                            UnhoverProbe();
                            HoverProbe(hit.transform.gameObject);
                        }
                    }
                }
                else
                {
                    UnhoverProbe();
                }
            }
            else
            {
                UnhoverProbe();
            }
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
            pid = pid.ToLower();
            string lab = (string)row["lab"];
            string subject = (string)row["subject"];
            string date = (string)row["date"];
            string selectable = (string)row["selectable"];

            Vector3 pos = new Vector3((float)row["ml_ccf_tip"], (float)row["ap_ccf_tip"], (float)row["dv_ccf_tip"]);
            Vector3 angles = new Vector3((float)row["phi"], (float)row["theta"], (float)row["depth"]);

            newProbe.GetComponentInChildren<BoxCollider>().enabled = false;

            pid2probe.Add(pid, newProbe);
            newProbe.GetComponentInChildren<ProbeComponent>().SetInfo(pid, lab, subject, date);

            SetProbePositionAndAngles(newProbe.transform, pos, angles);
            
            if (selectable.Equals("TRUE"))
                ActivateProbe(pid);
        }

        HighlightProbe("1841cf1f-725d-499e-ab8e-7f6fc8c308b6");
    }

    public void ActivateProbe(string pid)
    {
        GameObject probe = pid2probe[pid];
        Debug.Log("Activate: " + pid);
        probe.GetComponentInChildren<Renderer>().material.color = defaultColor;
        //pid2probe[pid].GetComponentInChildren<Renderer>().material.SetColor("_Color", colors[probeLabs[pid]]);
        probe.GetComponentInChildren<BoxCollider>().enabled = true;
        probe.GetComponentInChildren<BoxCollider>().isTrigger = true;
        // make it bigger
        probe.transform.localScale = new Vector3(5f, 1f, 5f);
    }

    public void DeactivateProbe(string pid)
    {
        pid2probe[pid].GetComponentInChildren<Renderer>().material.SetColor("Color", Color.white);
        pid2probe[pid].transform.localScale = Vector3.one;
    }

    public void HoverProbe(GameObject probe)
    {
        ProbeComponent probeComponent = probe.GetComponent<ProbeComponent>();

        (_, string lab, _, _) = probeComponent.GetInfo();

        if (!probeComponent.Highlighted)
        {
            probeComponent.SetTrackActive(true);
            probeComponent.SetTrackHighlight(Color.red);
            probe.GetComponent<Renderer>().material.color = Color.red;
        }
        hoveredProbe = probe;
    }

    public void UnhoverProbe()
    {
        if (hoveredProbe != null)
        {
            ProbeComponent probeComponent = hoveredProbe.GetComponent<ProbeComponent>();
            if (!probeComponent.Highlighted)
            {
                probeComponent.SetTrackActive(false);
                hoveredProbe.GetComponent<Renderer>().material.color = defaultColor;
            }
        }
        hoveredProbe = null;
    }

    public void HighlightProbe(string pid) {
        //pid = StripSpecialChars(pid);
        //Debug.Log(pid2probe);
        if (pid2probe.ContainsKey(pid))
            HighlightProbeGO(pid2probe[pid]);
        else
            Debug.Log(string.Format("{0} does not exist in pid list", pid));
    }

    private string StripSpecialChars(string input)
    {
        string[] chars = new string[] { ",", ".", "/", "!", "@", "#", "$", "%", "^", "&", "*", "'", "\"", ";", "_", "(", ")", ":", "|", "[", "]" };
        foreach (string checkStr in chars)
            if (input.Contains(checkStr))
                input.Replace(checkStr, "");
        return input;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="probe"></param>
    public void HighlightProbeGO(GameObject probe) {
        UnhighlightProbe();
        ProbeComponent probeComponent = probe.GetComponentInChildren<ProbeComponent>();

        highlightedProbe = probe;
        probeComponent.Highlighted = true;

        // also get the lab information
        (_, string lab, _, _) = probeComponent.GetInfo();

        probeComponent.SetTrackActive(true);
        probeComponent.SetTrackHighlight(labColors[lab]);
        probe.GetComponentInChildren<Renderer>().material.color = labColors[lab];
    }

    public void UnhighlightProbe()
    {
        if (highlightedProbe != null)
        {
            ProbeComponent probeComponent = highlightedProbe.GetComponentInChildren<ProbeComponent>();

            highlightedProbe.GetComponentInChildren<Renderer>().material.color = defaultColor;
            probeComponent.SetTrackActive(false);
            probeComponent.Highlighted = false; 
        }
        highlightedProbe = null;
    }

    public void SelectProbe(GameObject probe)
    {
        (string pid, _, _, _) = probe.GetComponent<ProbeComponent>().GetInfo();

#if !UNITY_EDITOR && UNITY_WEBGL
        SelectPID(pid);
#endif
        Debug.Log("Sent select message with payload: " + pid);
    }

    public void SelectCluster(GameObject cluster) {
        Debug.Log("not implemented");
    }
}
