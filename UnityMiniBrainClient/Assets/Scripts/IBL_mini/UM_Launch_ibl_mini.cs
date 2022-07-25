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
    private Dictionary<string, int> probeLabs;
    private string[] labs = {"angelakilab", "churchlandlab", "churchlandlab_ucla", "cortexlab",
       "danlab", "hoferlab", "mainenlab", "mrsicflogellab",
       "steinmetzlab", "wittenlab", "zadorlab" };
    private Dictionary<string, Color> labColors;

    private GameObject highlightedProbe;

    private string serverTarget;

    private void Awake()
    {
        pid2probe = new Dictionary<string, GameObject>();
        probeLabs = new Dictionary<string, int>();
        probe2pid = new Dictionary<GameObject, string>();

        //visibleNodes = new Dictionary<int, CCFTreeNode>();

        labColors = new Dictionary<string, Color>();
        for (int i = 0; i < labs.Length; i++)
            labColors.Add(labs[i], colors[i]);

        for (int i = 0; i < brainAreas.Count; i++)
        {
            brainAreas[i].GetComponentInChildren<Renderer>().material.SetColor("_Color", brainColors[i]);
        }
    }

    // Start is called before the first frame update
    void Start()
    {        
        //modelControl.SetBeryl(true);
        //modelControl.LateStart(loadDefaults);

        //if (loadDefaults)
        //    DelayedStart();

        cameraController.SetBrainAxisAngles(new Vector3(0f, 45f, 135f));
        umCamera.SwitchCameraMode(false);

        LoadProbes();
    }


    //private async void DelayedStart()
    //{
    //    //await modelControl.GetDefaultLoaded();
    //    //ccfLoaded = true;

    //    //foreach (CCFTreeNode node in modelControl.GetDefaultLoadedNodes())
    //    //{
    //    //    FixNodeTransformPosition(node);

    //    //    RegisterNode(node);
    //    //    node.SetNodeModelVisibility(true);
    //    //    node.SetShaderProperty("_Alpha", 0.15f);
    //    //}
    //}

    //public void FixNodeTransformPosition(CCFTreeNode node)
    //{
    //    // I don't know why we have to do this? For some reason when we load the node models their positions are all offset in space in a weird way... 
    //    node.GetNodeTransform().localPosition = Vector3.zero;
    //    node.GetNodeTransform().localRotation = Quaternion.identity;
    //    //node.RightGameObject().transform.localPosition = Vector3.forward * 11.4f;
    //}

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit))
            {

                //Select stage    
                if (hit.transform.gameObject.layer == 10)
                {
                    SelectProbe(hit.transform.gameObject);
                }
            }
        }
    }

    //public void RegisterNode(CCFTreeNode node)
    //{
    //    if (!visibleNodes.ContainsKey(node.ID))
    //        visibleNodes.Add(node.ID, node);
    //}

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
    }

    private void LoadProbes()
    {
        List<Dictionary<string,object>> data = CSVReader.ParseText(probeData.text);

        foreach (Dictionary<string,object> row in data)
        {
            GameObject newProbe = Instantiate(probeLinePrefab, probeParentT);
            string pid = (string)row["pid"];
            string lab = (string)row["lab"];

            int labIdx = 0;
            for (int i = 0; i < labs.Length; i++)
                if (labs[i].Equals(lab))
                {
                    labIdx = i;
                    break;
                }

            Vector3 pos = new Vector3((float)row["ml"], (float)row["ap"], (float)row["dv"]);
            Vector3 angles = new Vector3((float)row["phi"], (float)row["theta"], (float)row["depth"]);

            newProbe.GetComponentInChildren<BoxCollider>().enabled = false;

            pid2probe.Add(pid, newProbe);
            probe2pid.Add(newProbe, pid);
            probeLabs.Add(pid, labIdx);
            SetProbePositionAndAngles(newProbe.transform, pos, angles);
        }

        // activate a few probes for testing
        ActivateProbe("7d999a68-0215-4e45-8e6c-879c6ca2b771");
        ActivateProbe("3eb6e6e0-8a57-49d6-b7c9-f39d5834e682");
        ActivateProbe("18be19f9-6ca5-4fc8-9220-ba43c3e75905");
    }

    public void ActivateProbe(string pid)
    {
        pid2probe[pid].GetComponentInChildren<Renderer>().material.SetColor("_Color", defaultColor);
        //pid2probe[pid].GetComponentInChildren<Renderer>().material.SetColor("_Color", colors[probeLabs[pid]]);
        pid2probe[pid].GetComponentInChildren<BoxCollider>().enabled = true;
        pid2probe[pid].GetComponentInChildren<BoxCollider>().isTrigger = true;
        // make it bigger
        pid2probe[pid].transform.localScale = new Vector3(3f, 1f, 3f);
    }

    public void DeactivateProbe(string pid)
    {
        pid2probe[pid].GetComponentInChildren<Renderer>().material.SetColor("_Color", defaultColor);
        pid2probe[pid].transform.localScale = Vector3.one;
    }

    public void HighlightProbe(string pid) {
        Debug.Log("Highlighting pid: " + pid);
        HighlightProbeGO(pid2probe[pid]);
    }

    public void HighlightProbeGO(GameObject probe) {
        UnhighlightProbe();

        highlightedProbe = probe;

        probe.GetComponentInChildren<Renderer>().material.SetColor("_Color", Color.red);
    }

    public void UnhighlightProbe() {
        if (highlightedProbe != null) {
            highlightedProbe.GetComponentInChildren<Renderer>().material.SetColor("_Color", defaultColor);
        }
    }

    public void SwitchHighlightedProbeJS(int id)
    {

    }

    public void SelectProbe(GameObject probe)
    {
        string pid = probe2pid[probe.transform.parent.gameObject];
        HighlightProbeGO(probe);
#if UNITY_WEBGL
        SelectPID(pid);
#endif
        Debug.Log("Sent select message with payload: " + pid);
    }

    public void SelectCluster(GameObject cluster) {
        Debug.Log("not implemented");
    }
}
