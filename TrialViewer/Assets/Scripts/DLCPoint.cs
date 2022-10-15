using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DLCPoint : MonoBehaviour
{
    [SerializeField] private GameObject parentTexture;

    private Vector2 scale;

    private Vector2 originalScale = new Vector2(64f, 50f);

    private void Start()
    {
        StartCoroutine(GetScaleDelayed());
    }

    private IEnumerator GetScaleDelayed()
    {
        yield return new WaitForEndOfFrame();
        scale = Vector2.one;
        scale = new Vector2(parentTexture.GetComponent<RectTransform>().rect.width, parentTexture.GetComponent<RectTransform>().rect.height);

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
        //float x = (originalX / originalScale.x) * scale.x - scale.x/2;
        //float y = (originalY / originalScale.y) * scale.y - scale.y/2;
        transform.localPosition = new Vector2(originalX - 80, -(originalY - 64));
    }

}
