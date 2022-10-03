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
    private static extern void UpdateTrialTime(float time);
    [DllImport("__Internal")]
    private static extern void ChangeTrial(int trialInc);
    [DllImport("__Internal")]
    private static extern void TrialViewerLoaded();

    #region exposed fields
    [SerializeField] private Button prevTrialButton;
    [SerializeField] private Button nextTrialButton;
    [SerializeField] private Button playButton;
    [SerializeField] private Button stopButton;

    [SerializeField] private Image infoImage;
    [SerializeField] private Color defaultColor;
    [SerializeField] private Sprite defaultSprite;
    [SerializeField] private Sprite goSprite;
    [SerializeField] private Sprite correctSprite;
    [SerializeField] private Sprite wrongSprite;

    [SerializeField] private VideoPlayer leftVideoPlayer;
    [SerializeField] private VideoPlayer bodyVideoPlayer;
    [SerializeField] private VideoPlayer rightVideoPlayer;

    [SerializeField] private DLCPoint pawLcamR;
    [SerializeField] private DLCPoint pawRcamR;
    [SerializeField] private DLCPoint pawLcamL;
    [SerializeField] private DLCPoint pawRcamL;

    [SerializeField] private GaborStimulus stimulus;

    [SerializeField] private WheelComponent wheel;

    [SerializeField] private AssetReferenceT<TextAsset> timestampTextAsset;
    [SerializeField] private AssetReferenceT<TextAsset> trialTextAsset;

    [SerializeField] private AssetReferenceT<VideoClip> leftClip;
    [SerializeField] private AssetReferenceT<VideoClip> rightClip;
    [SerializeField] private AssetReferenceT<VideoClip> bodyClip;

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
#if !UNITY_EDITOR && UNITY_WEBGL
        // disable WebGLInput.captureAllKeyboardInput so elements in web page can handle keyboard inputs
        WebGLInput.captureAllKeyboardInput = false;
#endif
        Addressables.WebRequestOverride = EditWebRequestURL;

        LoadData("0802ced5-33a3-405e-8336-b65ebc5cb07c");

        Stop();
    }

    //Override the url of the WebRequest, the request passed to the method is what would be used as standard by Addressables.
    private void EditWebRequestURL(UnityWebRequest request)
    {
        if (request.url.Contains("http://"))
            request.url = request.url.Replace("http://", "https://");
        Debug.Log(request.url);
    }

    #region data loading
    public async void LoadData(string pid)
    {
        // for now we ignore the PID and just load the referenced assets
        Debug.Log("Starting async load calls");
        AsyncOperationHandle<TextAsset> timestampHandle = timestampTextAsset.LoadAssetAsync();
        AsyncOperationHandle<TextAsset> trialHandle = trialTextAsset.LoadAssetAsync();
        AsyncOperationHandle<VideoClip> leftHandle = leftClip.LoadAssetAsync();
        AsyncOperationHandle<VideoClip> rightHandle = rightClip.LoadAssetAsync();
        AsyncOperationHandle<VideoClip> bodyHandle = bodyClip.LoadAssetAsync();


        await Task.WhenAll(new Task[] { timestampHandle.Task, trialHandle.Task , leftHandle.Task, rightHandle.Task, bodyHandle.Task});

        Debug.Log("Passed initial load");
        // videos
        leftVideoPlayer.url = leftHandle.Result.originalPath;
        rightVideoPlayer.url = rightHandle.Result.originalPath;
        bodyVideoPlayer.url = bodyHandle.Result.originalPath;
        //leftVideoPlayer.clip = leftHandle.Result;
        //rightVideoPlayer.clip = rightHandle.Result;
        //bodyVideoPlayer.clip = bodyHandle.Result;

        leftVideoPlayer.Prepare();
        rightVideoPlayer.Prepare();
        bodyVideoPlayer.Prepare();

        // parse timestamp data
        //timestampData = CSVReader.ParseTimestampData(timestampHandle.Result.text);

        timestampData = new Dictionary<string, float[]>();
        string[] dataTypes = {"right_ts","left_idx","body_idx",
            "cr_paw_l_x", "cr_paw_l_y", "cr_paw_r_x", "cr_paw_r_y",
            "cl_paw_l_x", "cl_paw_l_y", "cl_paw_r_x", "cl_paw_r_y",
            "wheel"};
        foreach (string type in dataTypes)
        {
            Debug.Log("Loading: " + type);
            string path = string.Format("Assets/AddressableAssets/{0}/{0}.{1}.bytes", pid, type);
            AsyncOperationHandle<TextAsset> dataHandle = Addressables.LoadAssetAsync<TextAsset>(path);
            await dataHandle.Task;

            int nBytes = dataHandle.Result.bytes.Length;
            Debug.Log(string.Format("Loading {0} with {1} bytes", path, nBytes));
            float[] data = new float[nBytes / 4];

            Buffer.BlockCopy(dataHandle.Result.bytes, 0, data, 0, nBytes);
            Debug.LogFormat("Found {0} floats", data.Length);

            timestampData[type] = data;
        }


        // parse trial data
        trialData = CSVReader.ParseTrialData(trialHandle.Result.text);

        //Debug.Log(timestampData.Count);
        Debug.Log(trialData.Count);

        trial = 1;
        UpdateTrial();
    }

    #endregion 

    private void Update()
    {
        if (playing)
        {
            time += Time.deltaTime;

            if (leftVideoPlayer.isPrepared)
            {
                // find the next frame
                int frameIdx;
                for (frameIdx = currentTrialData.start; frameIdx < nextTrialData.start; frameIdx++)
                    if (timestampData["right_ts"][frameIdx] >= time)
                        break;

                if (frameIdx >= nextTrialData.start)
                    NextTrial();

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
                    infoImage.sprite = goSprite;
                    infoImage.color = Color.yellow;

                    if (infoCoroutine != null)
                        StopCoroutine(infoCoroutine);
                    infoCoroutine = StartCoroutine(ClearSprite(0.2f));
                }

                if (frameIdx >= currentTrialData.feedback && !playedFeedback)
                {
                    playedFeedback = true;
                    if (currentTrialData.correct)
                    {
                        infoImage.sprite = goSprite;
                        infoImage.color = Color.green;

                        if (infoCoroutine != null)
                            StopCoroutine(infoCoroutine);
                        infoCoroutine = StartCoroutine(ClearSprite(0.5f));
                    }
                    else
                    {
                        infoImage.sprite = wrongSprite;
                        infoImage.color = Color.red;

                        if (infoCoroutine != null)
                            StopCoroutine(infoCoroutine);
                        infoCoroutine = StartCoroutine(ClearSprite(0.5f));
                    }
                }

                // Set DLC points
                pawLcamR.SetPosition(timestampData["cr_paw_l_x"][frameIdx], timestampData["cr_paw_l_y"][frameIdx]);
                pawRcamR.SetPosition(timestampData["cr_paw_r_x"][frameIdx], timestampData["cr_paw_r_y"][frameIdx]);
                pawLcamL.SetPosition(timestampData["cl_paw_l_x"][frameIdx], timestampData["cl_paw_l_y"][frameIdx]);
                pawRcamL.SetPosition(timestampData["cl_paw_r_x"][frameIdx], timestampData["cl_paw_r_y"][frameIdx]);

                // Set videos
                rightVideoPlayer.frame = frameIdx;
                rightVideoPlayer.Play();
                bodyVideoPlayer.frame = (long)timestampData["body_idx"][frameIdx];
                bodyVideoPlayer.Play();
                leftVideoPlayer.frame = (long)timestampData["left_idx"][frameIdx];
                leftVideoPlayer.Play();
            }
        }
    }

    public IEnumerator ClearSprite(float delay)
    {
        yield return new WaitForSeconds(delay);

        infoImage.sprite = defaultSprite;
        infoImage.color = defaultColor;
    }


    public void UpdateTrial()
    {
        //reset trial variables
        playedGo = false;
        playedFeedback = false;

        currentTrialData = trialData[trial];
        nextTrialData = trialData[trial + 1];
        time = timestampData["right_ts"][currentTrialData.start]; // get the starting time for the trial
        Debug.Log(string.Format("Starting trial {0}: start {1} end {2} right {3} correct {4}", trial,
            currentTrialData.start, nextTrialData.start, currentTrialData.right, currentTrialData.correct));

        // set the stimulus properties
        stimulus.SetContrast(currentTrialData.contrast);
        sideFlip = currentTrialData.right ? 1 : -1;

        // set the wheel properties
        initDeg = wheel.CalculateDegrees(timestampData["wheel"][currentTrialData.stimOn]);
        endDeg = wheel.CalculateDegrees(timestampData["wheel"][currentTrialData.feedback]);
    }

    #region webpage callbacks
    public void SetSession(string pid)
    {
        Debug.LogWarning("Session cannot be changed until additional sessions are created");
    }

    public void SetTrial(int newTrial)
    {
        trial = newTrial;
        UpdateTrial();
        
        if (newTrial == 0)
            prevTrialButton.enabled = false;
        if (newTrial == trialData.Count)
            nextTrialButton.enabled = false;
    }
    #endregion

    #region button controls

    public void PrevTrial()
    {
        trial -= 1;
        UpdateTrial();

#if !UNITY_EDITOR && UNITY_WEBGL
        ChangeTrial(trial);
#endif
    }

    public void NextTrial()
    {
        trial += 1;
        UpdateTrial();

#if !UNITY_EDITOR && UNITY_WEBGL
        ChangeTrial(trial);
#endif
    }

    public void Play()
    {
        playing = true;
        pawLcamL.gameObject.SetActive(true);
        pawRcamL.gameObject.SetActive(true);
        pawLcamR.gameObject.SetActive(true);
        pawRcamR.gameObject.SetActive(true);
    }

    public void Stop()
    {
        playing = false;
        pawLcamL.gameObject.SetActive(false);
        pawRcamL.gameObject.SetActive(false);
        pawLcamR.gameObject.SetActive(false);
        pawRcamR.gameObject.SetActive(false);
    }
    #endregion
}
