using System.Collections.Generic;
using System.Text.RegularExpressions;

public class CSVReader
{
	static string SPLIT_RE = @",(?=(?:[^""]*""[^""]*"")*(?![^""]*""))";
	static string LINE_SPLIT_RE = @"\r\n|\n\r|\n|\r";
	static char[] TRIM_CHARS = { '\"' };

	public static List<(string pid, string eid, string lab, float depth, float theta, float phi, float ml, float ap, float dv)> ParseText(string text)
	{
		var list = new List<(string pid, string eid, string lab, float depth, float theta, float phi, float ml, float ap, float dv)>();

		var lines = Regex.Split(text, LINE_SPLIT_RE);

		if (lines.Length <= 1) return list;

		var header = Regex.Split(lines[0], SPLIT_RE);
		for (var i = 1; i < lines.Length; i++)
		{

			var values = Regex.Split(lines[i], SPLIT_RE);
			if (values.Length == 0 || values[0] == "") continue;

			// pid, eid, depth, theta, phi, ml, ap, dv
			string pid = values[0].ToLower();
			string eid = values[1].ToLower();
			string lab = values[2].ToLower();
			float depth = float.Parse(values[3]);
			float theta = float.Parse(values[4]);
			float phi = float.Parse(values[5]);
			float ml = float.Parse(values[6]);
			float ap = float.Parse(values[7]);
			float dv = float.Parse(values[8]);

			list.Add((pid, eid, lab, depth, theta, phi, ml, ap, dv));
		}
		return list;
	}
}