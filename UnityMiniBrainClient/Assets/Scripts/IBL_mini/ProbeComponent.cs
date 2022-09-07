using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ProbeComponent : MonoBehaviour
{
    [SerializeField] private GameObject probeTrackPrefab;

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
}
