using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class WheelComponent : MonoBehaviour
{
    /*
     * WHEEL NOTES
     * Wheel needs to be moved 35 deg to reach threshold
     * wheel gain is 4 visual degrees / mm moved
     * 
     * The wheel itself is 3.1 * 2
     * Encoder resolution is 1024 * 4
     * 
     */

    private const float WHEEL_DIAMETER = 3.1f * 2f;

    private float _degrees;
    public float Degrees() { return _degrees; }

    public void SetRotation(float mm)
    {
        _degrees = CalculateDegrees(mm);
        transform.localRotation = Quaternion.Euler(new Vector3(-_degrees, 0f, 0f));
    }

    public float CalculateDegrees(float mm)
    {
        return mm / (WHEEL_DIAMETER * Mathf.PI) * 360;
    }
}
