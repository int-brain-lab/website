using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.AddressableAssets;
using UnityEngine.Networking;
using UnityEngine.ResourceManagement.AsyncOperations;
using UnityEngine.UI;
using UnityEngine.Video;

public class TrialViewerManager : MonoBehaviour
{
    [DllImport("__Internal")]
    private static extern void UpdateTrialTime(float t);
    [DllImport("__Internal")]
    private static extern void ChangeTrial(int trialInc);
    [DllImport("__Internal")]
    private static extern void TrialViewerLoaded();
    [DllImport("__Internal")]
    private static extern void DataLoaded();

    #region exposed fields
    [SerializeField] private Button prevTrialButton;
    [SerializeField] private Button nextTrialButton;
    [SerializeField] private Button playButton;
    [SerializeField] private Sprite playSprite;
    [SerializeField] private Sprite stopSprite;

    [SerializeField] private GameObject _loadingScreen;
    [SerializeField] private GameObject _waitingScreen;
    [SerializeField] private TMP_Text _trialText;

    [SerializeField] private Image infoImage;
    [SerializeField] private Color defaultColor;
    [SerializeField] private Sprite defaultSprite;
    [SerializeField] private Sprite goSprite;
    [SerializeField] private Sprite correctSprite;
    [SerializeField] private Sprite wrongSprite;

    [SerializeField] private VideoPlayer videoPlayer;

    [SerializeField] private GaborStimulus stimulus;

    [SerializeField] private WheelComponent wheel;

    [SerializeField] private AudioManager audmanager;

    [SerializeField] private TextAsset pid2eidText;
    private Dictionary<string, string> pid2eid;

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

    private bool waitingToLoad = false;
    private string waitingToLoadPid;
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
        Debug.Log("(TrialViewer) v1.0.0");
        catalogHandle = Addressables.LoadContentCatalogAsync("https://viz.internationalbrainlab.org/WebGL/catalog_2022.12.21.22.26.50.json");

#if !UNITY_EDITOR && UNITY_WEBGL
        // disable WebGLInput.captureAllKeyboardInput so elements in web page can handle keyboard inputs
        WebGLInput.captureAllKeyboardInput = false;
#endif
        Addressables.WebRequestOverride = EditWebRequestURL;

        Camera.main.transparencySortMode = TransparencySortMode.Orthographic;

        LoadPid2Eid();

#if !UNITY_EDITOR && UNITY_WEBGL
        TrialViewerLoaded();
#elif UNITY_EDITOR
        LoadData("59b7a543-8827-4157-a4f6-d5d0e50fdf39");
#endif
    }

    //Override the url of the WebRequest, the request passed to the method is what would be used as standard by Addressables.
    private void EditWebRequestURL(UnityWebRequest request)
    {
        if (request.url.Contains("http://"))
            request.url = request.url.Replace("http://", "https://");
    }

    #region data loading

    private void LoadPid2Eid()
    {
        //pid2eidText
        pid2eid = CSVReader.ParsePid2Eid(pid2eidText.text);
    }

    public void WaitToLoad(string pid)
    {
        waitingToLoad = true;
        waitingToLoadPid = pid;
        if (loadDataRoutine != null)
            StopCoroutine(loadDataRoutine);
        _loadingScreen.SetActive(false);
        _waitingScreen.SetActive(true);
    }

    public void LoadData(string pid)
    {
        Stop();
        loadDataRoutine = StartCoroutine(LoadDataHelper(pid));
    }

    public IEnumerator LoadDataHelper(string pid)
    {
        string eid = pid2eid[pid];

        waitingToLoad = false;

        _loadingScreen.SetActive(true);
        _waitingScreen.SetActive(false);

        while (!catalogHandle.IsDone)
            yield return catalogHandle;

        // for now we ignore the PID and just load the referenced assets
        Debug.Log("Starting async load calls");
        string path = string.Format("Assets/AddressableAssets/{0}/{0}.trials.csv", eid);
        AsyncOperationHandle<TextAsset> trialHandle = Addressables.LoadAssetAsync<TextAsset>(path);

        Debug.Log("Passed initial load");
        // videos
        videoPlayer.url = $"https://viz.internationalbrainlab.org/static/WebGL2/{eid}.mp4";

        timestampData = new Dictionary<string, float[]>();

        string[] dataTypes = {"left_ts", "wheel"};

        Dictionary<string, AsyncOperationHandle<TextAsset>> dataHandles = new Dictionary<string, AsyncOperationHandle<TextAsset>>();


        foreach (string type in dataTypes)
        {
            Debug.Log("Loading: " + type);
            dataHandles.Add(type, Addressables.LoadAssetAsync<TextAsset>(string.Format("Assets/AddressableAssets/{0}/{0}.{1}.bytes", eid, type)));
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

        while (!videoPlayer.isPrepared)
            yield return null;

        Debug.Log("LOADED");
        _loadingScreen.SetActive(false);

#if !UNITY_EDITOR && UNITY_WEBGL
        DataLoaded();
#endif
    }

    #endregion 

    private int _prevFrame;
    private int _frame;

    private void Update()
    {
        if (waitingToLoad)
        {
            if (Input.GetMouseButtonDown(0))
                LoadData(waitingToLoadPid);
            return;
        }

        if (Input.GetMouseButtonDown(0) && !audmanager.gameObject.activeSelf)
            audmanager.gameObject.SetActive(true);

        if (playing)
        {
            _frame = (int)videoPlayer.frame;

            // check whether we are on a repeated frame or if the video hasn't loaded
            if (_frame == _prevFrame || _frame < 0)
                return;
            _prevFrame = _frame;

            time = timestampData["left_ts"][_frame];

#if !UNITY_EDITOR && UNITY_WEBGL
            UpdateTrialTime(time);
#endif

            if (_frame >= nextTrialData.start)
            {
                trial++;
                UpdateTrial();

#if !UNITY_EDITOR && UNITY_WEBGL
                ChangeTrial(trial);
#endif
            }

            // wheel properties
            wheel.SetRotation(timestampData["wheel"][_frame]);


            // stimulus properties
            if (_frame >= currentTrialData.stimOn && _frame <= currentTrialData.feedback)
            {
                // show stimulus and set position
                if (currentTrialData.contrast > 0)
                {
                    stimulus.gameObject.SetActive(true);

                    if (currentTrialData.correct)
                        stimulus.SetPosition(sideFlip * Mathf.InverseLerp(endDeg, initDeg, wheel.Degrees()));
                    else
                    {
                        stimulus.SetPosition(sideFlip * (1 + Mathf.InverseLerp(initDeg, endDeg, wheel.Degrees())));
                    }
                }

                // check if we played the go cue
                if (!playedGo)
                {
                    playedGo = true;

                    audmanager.PlayGoTone();

                    infoImage.sprite = goSprite;
                    infoImage.color = Color.yellow;

                    if (infoCoroutine != null)
                        StopCoroutine(infoCoroutine);
                    infoCoroutine = StartCoroutine(ClearSprite(0.1f));
                }
            }
            else if (_frame >= currentTrialData.feedback && _frame <= nextTrialData.start)
            {
                if (currentTrialData.contrast > 0)
                {
                    stimulus.gameObject.SetActive(true);

                    if (currentTrialData.correct)
                        stimulus.SetPosition(0f);
                    else
                        stimulus.SetPosition(sideFlip * 2f);
                }

                // check if we displayed the feedback data
                if (!playedFeedback)
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
            }
            else
                stimulus.gameObject.SetActive(false);

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
        if (waitingToLoad)
            return;

        prevTrialButton.enabled = trial > 0;
        nextTrialButton.enabled = trial < trialData.Count;

        //reset trial variables
        playedGo = false;
        playedFeedback = false;

        currentTrialData = trialData[trial];
        nextTrialData = trialData[trial + 1];

        _trialText.text = string.Format("Trial {0}", trial);

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

    }

    #region webpage callbacks
    public void SetSession(string pid)
    {
        Debug.Log(string.Format("(TrailViewer) SetSession called with pid {0}",pid));
        Stop();
        WaitToLoad(pid);
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

        trial = newTrial;
        UpdateTrial(playing);
    }

    public void Play()
    {
        playing = true;
        playButton.image.sprite = stopSprite;

        videoPlayer.Play();
    }

    public void Stop()
    {
        playing = false;
        playButton.image.sprite = playSprite;

        videoPlayer.Pause();
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

    public void PlayButton()
    {
        if (playing)
            Stop();
        else
            Play();
    }
    #endregion

    public void SetSpeed(float speed)
    {
        videoPlayer.playbackSpeed = speed;
    }
}
