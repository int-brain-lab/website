using System.Collections;
using TMPro;
using UnityEngine;

public class TimeBehavior : MonoBehaviour
{
    [SerializeField] private TrialViewerManager _tvManager;
    [SerializeField] private TMP_Text _text;
    // Start is called before the first frame update
    
    public void OnValueChanged(float value)
    {
        _tvManager.SetSpeed(value);

        _text.gameObject.SetActive(true);
        _text.text = string.Format("{0:F2}x", value);
        StartCoroutine(HideText());
    }

    private IEnumerator HideText()
    {
        yield return new WaitForSecondsRealtime(2f);
        _text.gameObject.SetActive(false);
    }
}
