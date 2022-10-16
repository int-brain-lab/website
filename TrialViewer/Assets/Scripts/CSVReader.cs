using UnityEngine;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Globalization;
using static UnityEngine.Rendering.DebugUI;

public class CSVReader
{
	static string SPLIT_RE = @",(?=(?:[^""]*""[^""]*"")*(?![^""]*""))";
	static string LINE_SPLIT_RE = @"\r\n|\n\r|\n|\r";
	static char[] TRIM_CHARS = { '\"' };


    //private List<(float time, int leftIdx, int rightIdx, float pawlx, float pawly, float pawrx, float pawry, float wheel)> timestampData;
    //private List<(int start, int stimOn, int feedback, bool right, float contrast, bool correct)> trialData;

    public static List<(float time, int leftIdx, int bodyIdx,
                float cr_pawlx, float cr_pawly, float cr_pawrx, float cr_pawry,
                float cl_pawlx, float cl_pawly, float cl_pawrx, float cl_pawry,
                float wheel)> ParseTimestampData(string text)
    {
        float tic = Time.realtimeSinceStartup;
        var data = new List<(float time, int leftIdx, int bodyIdx,
                float cr_pawlx, float cr_pawly, float cr_pawrx, float cr_pawry,
                float cl_pawlx, float cl_pawly, float cl_pawrx, float cl_pawry,
                float wheel)>();

        var lines = Regex.Split(text, LINE_SPLIT_RE);

        for (var i = 1; i < lines.Length; i++)
        {
            var values = Regex.Split(lines[i], SPLIT_RE);
            if (values.Length == 0 || values[0] == "") continue;
            for (int j = 0; j < values.Length; j++)
                values[j] = values[j].TrimStart(TRIM_CHARS).TrimEnd(TRIM_CHARS).Replace("\\", "");

			(float time, int leftIdx, int bodyIdx, 
                float cr_pawlx, float cr_pawly, float cr_pawrx, float cr_pawry,
                float cl_pawlx, float cl_pawly, float cl_pawrx, float cl_pawry,
                float wheel) entry;

            entry.time = float.Parse(values[1], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.leftIdx = int.Parse(values[2], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.bodyIdx = int.Parse(values[3], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.cr_pawlx = float.Parse(values[4], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.cr_pawly = float.Parse(values[5], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.cr_pawrx = float.Parse(values[6], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.cr_pawry = float.Parse(values[7], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.cl_pawlx = float.Parse(values[8], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.cl_pawly = float.Parse(values[9], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.cl_pawrx = float.Parse(values[10], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.cl_pawry = float.Parse(values[11], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.wheel = float.Parse(values[12], NumberStyles.Any, CultureInfo.InvariantCulture);

            data.Add(entry);
        }
        Debug.Log(string.Format("Time spent loading timestamps: {0}", Time.realtimeSinceStartup - tic));
        return data;
    }

    public static List<(int start, int stimOn, int feedback, bool right, float contrast, bool correct)> ParseTrialData(string text)
    {
        var data = new List<(int start, int stimOn, int feedback, bool right, float contrast, bool correct)>();

        var lines = Regex.Split(text, LINE_SPLIT_RE);

        for (var i = 1; i < lines.Length; i++)
        {
            var values = Regex.Split(lines[i], SPLIT_RE);
            if (values.Length == 0 || values[0] == "") continue;
            for (int j = 0; j < values.Length; j++)
                values[j] = values[j].TrimStart(TRIM_CHARS).TrimEnd(TRIM_CHARS).Replace("\\", "");

            (int start, int stimOn, int feedback, bool right, float contrast, bool correct) entry;

            entry.start = int.Parse(values[0], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.stimOn = int.Parse(values[1], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.feedback = int.Parse(values[2], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.right = values[3].Equals("R");
            entry.contrast = float.Parse(values[4], NumberStyles.Any, CultureInfo.InvariantCulture);
            entry.correct = int.Parse(values[5], NumberStyles.Any, CultureInfo.InvariantCulture)==1;

            data.Add(entry);
        }
        return data;
    }
}