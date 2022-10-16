using System.Collections;
using System.Collections.Generic;
using UnityEngine;

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

    // Update is called once per frame
    void Update()
    {
        Camera.main.transform.LookAt(cameraTarget);

        float fov = Camera.main.fieldOfView;

        float scroll = -Input.GetAxis("Mouse ScrollWheel");
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
}
