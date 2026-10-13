using System.Globalization;
using System.Text;
using AegisOne.Core.AI.Guardian;

if (args.Length < 2)
{
    Console.Error.WriteLine("usage: FeatureDump <corpus-directory> <output.csv>");
    return 1;
}

string corpus = args[0];
string output = args[1];

if (!Directory.Exists(corpus))
{
    Console.Error.WriteLine($"not a directory: {corpus}");
    return 1;
}

int written = 0, skipped = 0;

using var writer = new StreamWriter(output, false, new UTF8Encoding(false));

foreach (var path in Directory.EnumerateFiles(corpus, "*", SearchOption.AllDirectories))
{
    float[]? vector;
    try { vector = PeFeatureVector.Compute(path); }
    catch { vector = null; }

    if (vector == null) { skipped++; continue; }

    var line = new StringBuilder();
    line.Append(path.Replace(",", "%2C"));

    foreach (var value in vector)
    {
        line.Append(',');
        line.Append(value.ToString("R", CultureInfo.InvariantCulture));
    }

    writer.WriteLine(line.ToString());
    written++;

    if (written % 2000 == 0) Console.Error.WriteLine($"  {written} files");
}

Console.Error.WriteLine($"wrote {written} rows to {output}, skipped {skipped} unreadable");
Console.Error.WriteLine($"schema v{PeFeatureVector.Version}, {PeFeatureVector.Length} features");
return 0;
