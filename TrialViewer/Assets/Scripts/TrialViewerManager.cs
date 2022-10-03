using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.AddressableAssets;
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
    private List<(float time, int leftIdx, int bodyIdx,
                float cr_pawlx, float cr_pawly, float cr_pawrx, float cr_pawry,
                float cl_pawlx, float cl_pawly, float cl_pawrx, float cl_pawry,
                float wheel)> timestampData;
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

        LoadData("0802ced5-33a3-405e-8336-b65ebc5cb07c");

        Stop();
    }

    #region data loading
    public async void LoadData(string pid)
    {
        // for now we ignore the PID and just load the referenced assets
        AsyncOperationHandle<TextAsset> timestampHandle = timestampTextAsset.LoadAssetAsync();
        AsyncOperationHandle<TextAsset> trialHandle = trialTextAsset.LoadAssetAsync();
        AsyncOperationHandle<VideoClip> leftHandle = leftClip.LoadAssetAsync();
        AsyncOperationHandle<VideoClip> rightHandle = rightClip.LoadAssetAsync();
        AsyncOperationHandle<VideoClip> bodyHandle = bodyClip.LoadAssetAsync();
        await Task.WhenAll(new Task[] { timestampHandle.Task, trialHandle.Task , leftHandle.Task, rightHandle.Task, bodyHandle.Task});

        // videos
        leftVideoPlayer.clip = leftHandle.Result;
        rightVideoPlayer.clip = rightHandle.Result;
        bodyVideoPlayer.clip = bodyHandle.Result;

        leftVideoPlayer.Prepare();
        rightVideoPlayer.Prepare();
        bodyVideoPlayer.Prepare();

        // parse timestamp data
        timestampData = CSVReader.ParseTimestampData(timestampHandle.Result.text);

        // parse trial data
        trialData = CSVReader.ParseTrialData(trialHandle.Result.text);

        Debug.Log(timestampData.Count);
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
                    if (timestampData[frameIdx].time >= time)
                        break;

                if (frameIdx >= nextTrialData.start)
                    NextTrial();

                // wheel properties
                wheel.SetRotation(timestampData[frameIdx].wheel);


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
                    Debug.Log(frameIdx);
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
                pawLcamR.SetPosition(timestampData[frameIdx].cr_pawlx, timestampData[frameIdx].cr_pawly);
                pawRcamR.SetPosition(timestampData[frameIdx].cr_pawrx, timestampData[frameIdx].cr_pawry);
                pawLcamL.SetPosition(timestampData[frameIdx].cl_pawlx, timestampData[frameIdx].cl_pawly);
                pawRcamL.SetPosition(timestampData[frameIdx].cl_pawrx, timestampData[frameIdx].cl_pawry);

                // Set videos
                rightVideoPlayer.frame = frameIdx;
                rightVideoPlayer.Play();
                bodyVideoPlayer.frame = timestampData[frameIdx].bodyIdx;
                bodyVideoPlayer.Play();
                leftVideoPlayer.frame = timestampData[frameIdx].leftIdx;
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
        time = timestampData[currentTrialData.start].time; // get the starting time for the trial
        Debug.Log(string.Format("Starting trial {0}: start {1} end {2} right {3} correct {4}", trial,
            currentTrialData.start, nextTrialData.start, currentTrialData.right, currentTrialData.correct));

        // set the stimulus properties
        stimulus.SetContrast(currentTrialData.contrast);
        sideFlip = currentTrialData.right ? 1 : -1;

        // set the wheel properties
        initDeg = wheel.CalculateDegrees(timestampData[currentTrialData.stimOn].wheel);
        endDeg = wheel.CalculateDegrees(timestampData[currentTrialData.feedback].wheel);
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
        pawLcamL.enabled = true;
        pawRcamL.enabled = true;
        pawLcamR.enabled = true;
        pawRcamL.enabled = true;
    }

    public void Stop()
    {
        playing = false;
        pawLcamL.enabled = false;
        pawRcamL.enabled = false;
        pawLcamR.enabled = false;
        pawRcamL.enabled = false;
    }
    #endregion
}
