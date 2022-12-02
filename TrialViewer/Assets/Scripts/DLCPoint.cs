using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DLCPoint : MonoBehaviour
{
    private Vector2 _offset = new Vector2(80f, 64f);
    private float scale = 1f;

    private void Start()
    {
        gameObject.SetActive(false);
    }

    public void SetPosition(float originalX, float originalY)
    {
        if (originalX == -1 || originalY == -1)
        {
            gameObject.SetActive(false);
            return;
        }

        gameObject.SetActive(true);
        // Rescale the pixel position to match the screen position
        transform.localPosition = new Vector2(scale * (originalX - _offset.x), -scale * (originalY - _offset.y));
    }

    public void SetScale(float scale)
    {
        this.scale = scale;
    }
}
