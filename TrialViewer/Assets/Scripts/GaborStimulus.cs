using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GaborStimulus : MonoBehaviour
{
    [SerializeField] private float maxX;

    private Vector3 origPosition;

    private void Awake()
    {

        origPosition = transform.localPosition;
    }

    /// <summary>
    /// Expects an input from -1 to 1, will be re-scaled
    /// </summary>
    /// <param name="perc"></param>
    public void SetPosition(float perc)
    {
        Vector3 newPosition = origPosition;
        newPosition.x = perc * maxX;
        transform.localPosition = newPosition;
    }

    public void SetContrast(float contrast)
    {
        GetComponent<Renderer>().material.SetFloat("_Contrast", contrast);
    }
}
