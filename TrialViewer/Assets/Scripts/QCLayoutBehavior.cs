using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class QCLayoutBehavior : MonoBehaviour
{
    [SerializeField] private GameObject _layoutA;
    [SerializeField] private GameObject _layoutB;

    [SerializeField] private GameObject LeftVideoGOA;
    [SerializeField] private GameObject RightVideoGOA;
    [SerializeField] private GameObject BodyVideoGOA;
    [SerializeField] private GameObject PupilVideoGOA;

    [SerializeField] private GameObject LeftVideoGOB;
    [SerializeField] private GameObject RightVideoGOB;
    [SerializeField] private GameObject BodyVideoGOB;
    [SerializeField] private GameObject PupilVideoGOB;

    [SerializeField] private GameObject CameraGO;

    [SerializeField] private List<Transform> _leftVideoDlCTs;
    [SerializeField] private List<Transform> _rightVideoDlCTs;
    [SerializeField] private List<Transform> _bodyVideoDlCTs;
    [SerializeField] private List<Transform> _pupilVideoDlCTs;

    [SerializeField] private CameraController _cameraController;
    [SerializeField] private GameObject _timeSlider;

    private bool layoutA = true;

    public void SwitchLayout()
    {
        layoutA = !layoutA;

        _layoutA.SetActive(layoutA);
        _layoutB.SetActive(!layoutA);
        _timeSlider.SetActive(!layoutA);

        float scale = layoutA ? 1f : 1.5f;

        if (layoutA)
        {
            foreach (Transform t in _leftVideoDlCTs)
            {
                t.SetParent(LeftVideoGOA.transform);
                t.gameObject.GetComponent<DLCPoint>().SetScale(scale);
            }
            foreach (Transform t in _rightVideoDlCTs)
            {
                t.SetParent(RightVideoGOA.transform);
                t.gameObject.GetComponent<DLCPoint>().SetScale(scale);
            }
            foreach (Transform t in _bodyVideoDlCTs)
            {
                t.SetParent(BodyVideoGOA.transform);
                t.gameObject.GetComponent<DLCPoint>().SetScale(scale);
            }
            foreach (Transform t in _pupilVideoDlCTs)
            {
                t.SetParent(PupilVideoGOA.transform);
                t.gameObject.GetComponent<DLCPoint>().SetScale(scale);
            }

            _cameraController.SetOffset(Vector2.zero);
        }
        else
        {
            foreach (Transform t in _leftVideoDlCTs)
            {
                t.SetParent(LeftVideoGOB.transform);
                t.gameObject.GetComponent<DLCPoint>().SetScale(scale);
            }
            foreach (Transform t in _rightVideoDlCTs)
            {
                t.SetParent(RightVideoGOB.transform);
                t.gameObject.GetComponent<DLCPoint>().SetScale(scale);
            }
            foreach (Transform t in _bodyVideoDlCTs)
            {
                t.SetParent(BodyVideoGOB.transform);
                t.gameObject.GetComponent<DLCPoint>().SetScale(scale);
            }
            foreach (Transform t in _pupilVideoDlCTs)
            {
                t.SetParent(PupilVideoGOB.transform);
                t.gameObject.GetComponent<DLCPoint>().SetScale(scale);
            }

            _cameraController.SetOffset(new Vector2(-135f, 0f));
        }
    }
}
