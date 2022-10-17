using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEngine.Networking;
using UnityEngine.ResourceManagement.AsyncOperations;
using UnityEngine.UI;
using UnityEngine.Video;

public class TrialViewerManager : MonoBehaviour
{
    [DllImport("__Internal")]
    private static extern void UpdateTrialTime(float t0, float t1, float t);
    [DllImport("__Internal")]
    private static extern void ChangeTrial(int trialInc);
    [DllImport("__Internal")]
    private static extern void TrialViewerLoaded();

    #region exposed fields
    [SerializeField] private Button prevTrialButton;
    [SerializeField] private Button nextTrialButton;
    [SerializeField] private Button playButton;
    [SerializeField] private Button stopButton;

    [SerializeField] private GameObject _loadingScreen;

    [SerializeField] private Image infoImage;
    [SerializeField] private Color defaultColor;
    [SerializeField] private Sprite defaultSprite;
    [SerializeField] private Sprite goSprite;
    [SerializeField] private Sprite correctSprite;
    [SerializeField] private Sprite wrongSprite;

    [SerializeField] private VideoPlayer videoPlayer;

    [SerializeField] private DLCPoint cr_pawL;
    [SerializeField] private DLCPoint cr_pawR;
    [SerializeField] private DLCPoint cr_tongueEnd;
    [SerializeField] private DLCPoint cr_tubeTop;

    [SerializeField] private DLCPoint cl_pawL;
    [SerializeField] private DLCPoint cl_pawR;
    [SerializeField] private DLCPoint cl_tongueEnd;
    [SerializeField] private DLCPoint cl_tubeTop;

    [SerializeField] private DLCPoint body_tail;

    [SerializeField] private DLCPoint pupilTop;
    [SerializeField] private DLCPoint pupilRight;
    [SerializeField] private DLCPoint pupilBottom;
    [SerializeField] private DLCPoint pupilLeft;

    [SerializeField] private GaborStimulus stimulus;

    [SerializeField] private WheelComponent wheel;

    [SerializeField] private AudioManager audmanager;

    #endregion

    #region data
    //private List<(float time, int leftIdx, int bodyIdx,
    //            float cr_pawlx, float cr_pawly, float cr_pawrx, float cr_pawry,
    //            float cl_pawlx, float cl_pawly, float cl_pawrx, float cl_pawry,
    //            float wheel)> timestampData;
    private Dictionary<string, float[]> timestampData;

    private List<(int start, int stimOn, int feedback, bool right, float contrast, bool correct)> trialData;
    private (int start, int stimOn, int feedback, bool right, float contrast, bool correct) currentTrialData;
    private (int start, int stimOn, int feedback, bool right, float contrast, bool correct) nextTrialData;
    #endregion

    #region local vars
    private float time; // the current time referenced from 0 across the entire session
    private int trial;
    private bool playing;
    private Coroutine infoCoroutine;

    private AsyncOperationHandle catalogHandle;
    private Coroutine loadDataRoutine;
    #endregion

    #region trial vars
    private bool playedGo;
    private bool playedFeedback;
    private float sideFlip;
    private float initDeg;
    private float endDeg;
    #endregion

    private void Awake()
    {
        Debug.Log("(TrialViewer) v0.1.0");
        catalogHandle = Addressables.LoadContentCatalogAsync("https://viz.internationalbrainlab.org/WebGL/catalog_2022.10.17.05.35.13.json");

#if !UNITY_EDITOR && UNITY_WEBGL
        // disable WebGLInput.captureAllKeyboardInput so elements in web page can handle keyboard inputs
        WebGLInput.captureAllKeyboardInput = false;
#endif
        Addressables.WebRequestOverride = EditWebRequestURL;

        //StartCoroutine(LoadData("47be9ae4-290f-46ab-b047-952bc3a1a509"));
        //StartCoroutine(LoadData("decc8d40-cf74-4263-ae9d-a0cc68b47e86"));   
        loadDataRoutine = StartCoroutine(LoadData("7d999a68-0215-4e45-8e6c-879c6ca2b771"));

        // force load the remote content catalog

        Stop();
    }

    //Override the url of the WebRequest, the request passed to the method is what would be used as standard by Addressables.
    private void EditWebRequestURL(UnityWebRequest request)
    {
        if (request.url.Contains("http://"))
            request.url = request.url.Replace("http://", "https://");
    }

    #region data loading
    public IEnumerator LoadData(string pid)
    {
        _loadingScreen.SetActive(true);
        while (!catalogHandle.IsDone)
            yield return catalogHandle;

        // for now we ignore the PID and just load the referenced assets
        Debug.Log("Starting async load calls");
        string path = string.Format("Assets/AddressableAssets/{0}/{0}.trials.csv", pid);
        AsyncOperationHandle<TextAsset> trialHandle = Addressables.LoadAssetAsync<TextAsset>(path);

        Debug.Log("Passed initial load");
        // videos
        videoPlayer.url = string.Format("https://viz.internationalbrainlab.org/WebGL/{0}.mp4", pid);

        timestampData = new Dictionary<string, float[]>();
        string[] dataTypes = {"left_ts", "wheel", "tail_start_x",
                               "tail_start_y", "cl_nose_tip_x", "cl_nose_tip_y", "cl_paw_l_x",
                               "cl_paw_l_y", "cl_paw_r_x", "cl_paw_r_y", "cl_tube_top_x",
                               "cl_tube_top_y", "cl_tongue_end_l_x", "cl_tongue_end_l_y",
                               "cl_tongue_end_r_x", "cl_tongue_end_r_y", "cr_nose_tip_x",
                               "cr_nose_tip_y", "cr_paw_l_x", "cr_paw_l_y", "cr_paw_r_x", "cr_paw_r_y",
                               "cr_tube_top_x", "cr_tube_top_y", "cr_tongue_end_l_x",
                               "cr_tongue_end_l_y", "cr_tongue_end_r_x", "cr_tongue_end_r_y",
                               "pupil_right_r_x", "pupil_right_r_y", "pupil_left_r_x",
                               "pupil_left_r_y", "pupil_top_r_x", "pupil_top_r_y", "pupil_bottom_r_x",
                               "pupil_bottom_r_y"};

        Dictionary<string, AsyncOperationHandle<TextAsset>> dataHandles = new Dictionary<string, AsyncOperationHandle<TextAsset>>();


        foreach (string type in dataTypes)
        {
            Debug.Log("Loading: " + type);
            dataHandles.Add(type, Addressables.LoadAssetAsync<TextAsset>(string.Format("Assets/AddressableAssets/{0}/{0}.{1}.bytes", pid, type)));
        }

        foreach (KeyValuePair<string, AsyncOperationHandle<TextAsset>> kvp in dataHandles)
        {
            string type = kvp.Key;
            AsyncOperationHandle<TextAsset> dataHandle = kvp.Value;
            while (!dataHandle.IsDone)
                yield return dataHandle;

            int nBytes = dataHandle.Result.bytes.Length;
            Debug.Log(string.Format("Loading {0} with {1} bytes", type, nBytes));
            float[] data = new float[nBytes / 4];

            Buffer.BlockCopy(dataHandle.Result.bytes, 0, data, 0, nBytes);
            Debug.LogFormat("Found {0} floats", data.Length);

            timestampData[type] = data;
        }

        while (!trialHandle.IsDone)
            yield return trialHandle;

        // parse trial data
        trialData = CSVReader.ParseTrialData(trialHandle.Result.text);

        trial = 0;
        UpdateTrial();

        MoveToFrameAndPrepare(currentTrialData.start);

        cl_pawL.gameObject.SetActive(true);
        cl_pawR.gameObject.SetActive(true);
        cl_tongueEnd.gameObject.SetActive(true);
        cl_tubeTop.gameObject.SetActive(true);

        cr_pawL.gameObject.SetActive(true);
        cr_pawR.gameObject.SetActive(true);
        cr_tongueEnd.gameObject.SetActive(true);
        cr_tubeTop.gameObject.SetActive(true);

        body_tail.gameObject.SetActive(true);

        pupilBottom.gameObject.SetActive(true);
        pupilLeft.gameObject.SetActive(true);
        pupilRight.gameObject.SetActive(true);
        pupilTop.gameObject.SetActive(true);

        while (!videoPlayer.isPrepared)
            yield return null;


#if !UNITY_EDITOR && UNITY_WEBGL
        TrialViewerLoaded();
#endif

        Debug.Log("LOADED");
        _loadingScreen.SetActive(false);
    }

    #endregion 

    private void Update()
    {
        if (Input.GetMouseButtonDown(0) && !audmanager.gameObject.activeSelf)
            audmanager.gameObject.SetActive(true);

        if (playing)
        {
            int frameIdx = (int)videoPlayer.frame;

            // catch in case the video hasn"t finished loading
            if (frameIdx == -1)
                return;

            time = timestampData["left_ts"][frameIdx];

#if !UNITY_EDITOR && UNITY_WEBGL
            UpdateTrialTime(timestampData["left_ts"][currentTrialData.start], timestampData["left_ts"][nextTrialData.start], time);
#endif

            if (frameIdx >= nextTrialData.start)
            {
                trial++;
                UpdateTrial();

        #if !UNITY_EDITOR && UNITY_WEBGL
                ChangeTrial(trial);
        #endif
            }

            // wheel properties
            wheel.SetRotation(timestampData["wheel"][frameIdx]);


            // stimulus properties
            if (currentTrialData.correct)
            {
                if (frameIdx >= currentTrialData.stimOn && frameIdx <= currentTrialData.feedback)
                {
                    stimulus.gameObject.SetActive(true);

                    float deg = wheel.Degrees();
                    Mathf.InverseLerp(initDeg, endDeg, deg);
                    stimulus.SetPosition(sideFlip * Mathf.InverseLerp(endDeg, initDeg, deg));
                }
                else if (frameIdx >= currentTrialData.feedback && frameIdx <= nextTrialData.start)
                {
                    stimulus.gameObject.SetActive(true);
                    stimulus.SetPosition(0f);
                }
                else
                    stimulus.gameObject.SetActive(false);
            }
            else
            {
                if (frameIdx >= currentTrialData.stimOn && frameIdx <= currentTrialData.feedback)
                {
                    stimulus.gameObject.SetActive(true);

                    float deg = wheel.Degrees();
                    Mathf.InverseLerp(initDeg, endDeg, deg);
                    stimulus.SetPosition(1 + -sideFlip * Mathf.InverseLerp(endDeg, initDeg, deg));
                }
                else if (frameIdx >= currentTrialData.feedback && frameIdx <= nextTrialData.start)
                {
                    stimulus.gameObject.SetActive(true);
                    stimulus.SetPosition(2f);
                }
                else
                    stimulus.gameObject.SetActive(false);
            }

            if (frameIdx >= currentTrialData.stimOn && !playedGo)
            {
                // stim on
                playedGo = true;
                
                audmanager.PlayGoTone();

                infoImage.sprite = goSprite;
                infoImage.color = Color.yellow;

                if (infoCoroutine != null)
                    StopCoroutine(infoCoroutine);
                infoCoroutine = StartCoroutine(ClearSprite(0.1f));
            }

            if (frameIdx >= currentTrialData.feedback && !playedFeedback)
            {
                playedFeedback = true;
                if (currentTrialData.correct)
                {
                    infoImage.sprite = correctSprite;
                    infoImage.color = Color.blue;

                    if (infoCoroutine != null)
                        StopCoroutine(infoCoroutine);
                    infoCoroutine = StartCoroutine(ClearSprite(0.5f));
                }
                else
                {
                    audmanager.PlayWhiteNoise();
                    infoImage.sprite = wrongSprite;
                    infoImage.color = Color.red;

                    if (infoCoroutine != null)
                        StopCoroutine(infoCoroutine);
                    infoCoroutine = StartCoroutine(ClearSprite(0.5f));
                }
            }

            // Set DLC points
            SetDLC2Frame(frameIdx);
        }
    }

    public IEnumerator ClearSprite(float delay)
    {
        yield return new WaitForSeconds(delay);

        infoImage.sprite = defaultSprite;
        infoImage.color = defaultColor;
    }

    Coroutine nextFrameRoutine;

    public void UpdateTrial(bool forceFrame = false)
    {
        //reset trial variables
        playedGo = false;
        playedFeedback = false;

        currentTrialData = trialData[trial];
        nextTrialData = trialData[trial + 1];

#if UNITY_EDITOR
        Debug.Log(string.Format("Starting trial {0}: start {1} end {2} right {3} correct {4}", trial,
            currentTrialData.start, nextTrialData.start, currentTrialData.right, currentTrialData.correct));
#endif

        // set the stimulus properties
        stimulus.SetContrast(currentTrialData.contrast);
        sideFlip = currentTrialData.right ? 1 : -1;

        // set the wheel properties
        initDeg = wheel.CalculateDegrees(timestampData["wheel"][currentTrialData.stimOn]);
        endDeg = wheel.CalculateDegrees(timestampData["wheel"][currentTrialData.feedback]);

        if (!playing || forceFrame)
        {
            if (forceFrame)
            {
                if (nextFrameRoutine != null)
                    StopCoroutine(nextFrameRoutine);
                nextFrameRoutine = StartCoroutine(UpdateVideoFramesAndPlay());
            }
            else
            {
                // if we aren"t playing, move the videos to the correct frame
                MoveToFrameAndPrepare(currentTrialData.start);
            }
        }
    }

    private IEnumerator UpdateVideoFramesAndPlay()
    {
        Stop();

        MoveToFrameAndPrepare(currentTrialData.start);

        while (!videoPlayer.isPrepared)
            yield return null;

        Play();
    }

    private void MoveToFrameAndPrepare(int frame)
    {
        videoPlayer.frame = frame;

        videoPlayer.Prepare();

        SetDLC2Frame(frame);
    }

    private void SetDLC2Frame(int frame)
    {
        cr_pawL.SetPosition(timestampData["cr_paw_l_x"][frame], timestampData["cr_paw_l_y"][frame]);
        cr_pawR.SetPosition(timestampData["cr_paw_r_x"][frame], timestampData["cr_paw_r_y"][frame]);
        cr_tongueEnd.SetPosition(timestampData["cr_tongue_end_l_x"][frame], timestampData["cr_tongue_end_l_y"][frame]);
        cr_tubeTop.SetPosition(timestampData["cr_tube_top_x"][frame], timestampData["cr_tube_top_y"][frame]);

        cl_pawL.SetPosition(timestampData["cl_paw_l_x"][frame], timestampData["cl_paw_l_y"][frame]);
        cl_pawR.SetPosition(timestampData["cl_paw_r_x"][frame], timestampData["cl_paw_r_y"][frame]);
        cl_tongueEnd.SetPosition(timestampData["cl_tongue_end_l_x"][frame], timestampData["cl_tongue_end_l_y"][frame]);
        cl_tubeTop.SetPosition(timestampData["cl_tube_top_x"][frame], timestampData["cl_tube_top_y"][frame]);

        body_tail.SetPosition(timestampData["tail_start_x"][frame], timestampData["tail_start_y"][frame]);

        pupilTop.SetPosition(timestampData["pupil_top_r_x"][frame], timestampData["pupil_top_r_y"][frame]);
        pupilRight.SetPosition(timestampData["pupil_right_r_x"][frame], timestampData["pupil_right_r_y"][frame]);
        pupilBottom.SetPosition(timestampData["pupil_bottom_r_x"][frame], timestampData["pupil_bottom_r_y"][frame]);
        pupilLeft.SetPosition(timestampData["pupil_left_r_x"][frame], timestampData["pupil_left_r_y"][frame]);
    }

    #region webpage callbacks
    public void SetSession(string pid)
    {
        if (loadDataRoutine != null)
            StopCoroutine(loadDataRoutine);
        loadDataRoutine = StartCoroutine(LoadData(pid));
    }

    /// <summary>
    /// Callback from the webpage to tell us which trial to go to
    /// 
    /// Stops playback
    /// </summary>
    /// <param name="newTrial"></param>
    public void SetTrial(int newTrial)
    {
        Debug.Log(string.Format("(TrialViewer) Setting trial to {0}", newTrial));
        if (newTrial == 0)
            prevTrialButton.enabled = false;
        if (newTrial == trialData.Count)
            nextTrialButton.enabled = false;

        trial = newTrial;
        UpdateTrial(playing);
    }
    #endregion

    #region button controls

    public void PrevTrial()
    {
        trial -= 1;
        UpdateTrial(playing);

#if !UNITY_EDITOR && UNITY_WEBGL
        ChangeTrial(trial);
#endif
    }

    public void NextTrial()
    {
        trial += 1;
        UpdateTrial(playing);

#if !UNITY_EDITOR && UNITY_WEBGL
        ChangeTrial(trial);
#endif
    }

    public void Play()
    {
        playing = true;

        videoPlayer.Play();
    }

    private void PlayOnPrepared()
    {

    }

    public void Stop()
    {
        playing = false;

        videoPlayer.Stop();
    }
    #endregion
}
