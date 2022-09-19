using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ProbeComponent : MonoBehaviour
{
    [SerializeField] private GameObject probeTrack;

    private string pid;
    private string lab;
    private string mouse;
    private string date;


    public void SetTrackHighlight(Color color)
    {
        probeTrack.GetComponent<Renderer>().material.color = color;
    }

    public void SetTrackActive(bool state)
    {
        probeTrack.SetActive(state);
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
