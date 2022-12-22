using System.Collections.Generic;
using System.Globalization;
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
			string pid = values[0].ToLowerInvariant();
			string eid = values[1].ToLowerInvariant();
			string lab = values[2].ToLowerInvariant();
			float depth = float.Parse(values[3], NumberStyles.Any, CultureInfo.InvariantCulture);
			float theta = float.Parse(values[4], NumberStyles.Any, CultureInfo.InvariantCulture);
			float phi = float.Parse(values[5], NumberStyles.Any, CultureInfo.InvariantCulture);
			float ml = float.Parse(values[6], NumberStyles.Any, CultureInfo.InvariantCulture);
			float ap = float.Parse(values[7], NumberStyles.Any, CultureInfo.InvariantCulture);
			float dv = float.Parse(values[8], NumberStyles.Any, CultureInfo.InvariantCulture);

			list.Add((pid, eid, lab, depth, theta, phi, ml, ap, dv));
		}
		return list;
	}
}