using UnityEngine;
using Color = UnityEngine.Color;

public class ProbeComponent : MonoBehaviour
{
    private LineRenderer lineRenderer;

    private string pid;
    private string lab;
    private string mouse;
    private string date;
    private bool _positionSet;

    public bool Highlighted { get; set; }

    private void Awake()
    {
        _positionSet = false;
        lineRenderer = GetComponent<LineRenderer>();
        lineRenderer.enabled = false;
    }

    public void SetTrackHighlight(Color color)
    {
        lineRenderer.startColor = color;
        lineRenderer.endColor = color;
    }

    public void SetTrackActive(bool state)
    {
        if (!_positionSet)
        {
            lineRenderer.SetPositions(new Vector3[] {
            transform.position + transform.up * -15,
            transform.position + transform.up * 15});
            _positionSet = true;
            lineRenderer.startColor = Color.red;
            lineRenderer.endColor = Color.red;
        }
        lineRenderer.enabled = state;
    }

    public void SetInfo(string pid)
    {
        this.pid = pid;
        //this.lab = lab;
        //this.mouse = mouse;
        //this.date = date;
    }

    public (string, string ,string, string) GetInfo()
    {
        return (pid, lab, mouse, date);
    }
}
