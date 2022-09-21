using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;

public class TrialViewerManager : MonoBehaviour
{
    [DllImport("__Internal")]
    private static extern void UpdateTrialTime(float time);
    [DllImport("__Internal")]
    private static extern void ChangeTrial(int trialInc);
    [DllImport("__Internal")]
    private static extern void TrialViewerLoaded();

    [SerializeField] private Button prevTrialButton;
    [SerializeField] private Button nextTrialButton;
    [SerializeField] private Button playButton;
    [SerializeField] private Button stopButton;

    private List<(float start, float stimOn, float feedback, bool right, float contrast, bool correct)> trialData;

    private void Awake()
    {
#if !UNITY_EDITOR && UNITY_WEBGL
        // disable WebGLInput.captureAllKeyboardInput so elements in web page can handle keyboard inputs
        WebGLInput.captureAllKeyboardInput = false;
#endif

        LoadData("0802ced5-33a3-405e-8336-b65ebc5cb07c");
    }

    public void LoadData(string pid)
    {

    }

    public void SetSession(string pid)
    {

    }

    public void SetTrial(int newTrial)
    {

        if (newTrial == 0)
            prevTrialButton.enabled = false;
        if (newTrial == trialData.Count)
            nextTrialButton.enabled = false;
    }

    public void PrevTrial()
    {

    }

    public void NextTrial()
    {

    }

    public void Play()
    {

    }

    public void Stop()
    {

    }
}
