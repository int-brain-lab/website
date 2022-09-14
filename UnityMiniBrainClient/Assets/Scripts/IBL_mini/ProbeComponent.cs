using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ProbeComponent : MonoBehaviour
{
    [SerializeField] private GameObject probeTrackPrefab;

    private string pid;
    private string lab;
    private string mouse;
    private string date;

    private GameObject probeTrack;

    public void SetTrackHighlight(bool state)
    {
        if (state)
            probeTrack.GetComponent<Renderer>().material.color = Color.red;
        else
            probeTrack.GetComponent<Renderer>().material.color = Color.cyan;
    }

    public void SetTrackActive(bool state)
    {
        if (state)
        {
            if (probeTrack == null)
                probeTrack = Instantiate(probeTrackPrefab, transform);
            probeTrack.SetActive(true);
        }
        else
            probeTrack.SetActive(false);
    }

    public void SetInfo(string pid, string lab, string mouse, string date)
    {
        this.pid = pid;
        this.lab = lab;
        this.mouse = mouse;
        this.date = date;
    }

    public (string, string ,string, string) GetInfo()
    {
        return (pid, lab, mouse, date);
    }
}
