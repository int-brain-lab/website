using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class InfoText : MonoBehaviour
{
    [SerializeField] private TMP_Text labText;
    [SerializeField] private TMP_Text subjText;
    [SerializeField] private TMP_Text dateText;

    private void Awake()
    {
        DisableText();
    }

    public void SetText(string lab, string subj, string date, Color color)
    {
        labText.enabled = true;
        subjText.enabled = true;
        dateText.enabled = true;
        labText.text = lab;
        labText.color = color;
        subjText.text = subj;
        subjText.color = color;
        dateText.text = date;
        dateText.color = color;
    }

    public void DisableText()
    {
        labText.enabled = false;
        subjText.enabled = false;
        dateText.enabled = false;
    }
}
