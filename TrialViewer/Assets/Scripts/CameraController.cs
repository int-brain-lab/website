using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UIElements;

public class CameraController : MonoBehaviour
{
    [SerializeField] Transform cameraTarget;
    [SerializeField] Transform cameraRotator;

    float fovDelta = 4f;
    float minFoV = 20f;
    float maxFoV = 100f;

    float pitch = 45f;
    float rotation = 180f;
    float rotDelta = 100f;

    private Vector2 _cameraOffset = new Vector2(0f, 0f);

    // Update is called once per frame
    void Update()
    {
        float scroll = -Input.GetAxis("Mouse ScrollWheel");

        if (Input.GetMouseButton(0) || scroll != 0)
        {
            UpdateCamera(scroll);
        }
    }
    
    private void UpdateCamera(float scroll = 0f)
    {
        Camera.main.transform.localPosition = new Vector3(0f, 0f, -22.4f);
        Camera.main.transform.LookAt(cameraTarget);
        Camera.main.transform.Translate(_cameraOffset);

        float fov = Camera.main.fieldOfView;

        fov += fovDelta * scroll;
        fov = Mathf.Clamp(fov, minFoV, maxFoV);

        Camera.main.fieldOfView = fov;

        if (Input.GetMouseButton(0))
        {
            // user is holding the left mouse button down
            float xRot = Input.GetAxis("Mouse X") * Time.deltaTime * rotDelta;
            float yRot = Input.GetAxis("Mouse Y") * Time.deltaTime * rotDelta;

            pitch -= yRot;
            rotation += xRot;

            cameraRotator.rotation = Quaternion.Euler(pitch, rotation, 0f);
        }
    }

    public void SetOffset(Vector2 cameraOffset)
    {
        _cameraOffset = cameraOffset;
        UpdateCamera();
    }

}
