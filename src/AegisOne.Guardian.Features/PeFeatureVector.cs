using System;
using System.Buffers.Binary;
using System.IO;

namespace AegisOne.Core.AI.Guardian
{
    public static class PeFeatureVector
    {
        public const int Version = 1;
        public const int Length = 48;
        public const int WindowBytes = 256 * 1024;
        public const int HistogramBuckets = 32;

        public const int IdxLogSize = 0;
        public const int IdxHeaderEntropy = 1;
        public const int IdxMidEntropy = 2;
        public const int IdxTailEntropy = 3;
        public const int IdxMaxEntropy = 4;
        public const int IdxEntropyGap = 5;
        public const int IdxHistogram = 6;
        public const int IdxPrintableRatio = 38;
        public const int IdxNullRatio = 39;
        public const int IdxIsPe = 40;
        public const int IdxSectionCount = 41;
        public const int IdxIsDll = 42;
        public const int IdxTimestampBad = 43;
        public const int IdxIs64Bit = 44;
        public const int IdxIsGuiSubsystem = 45;
        public const int IdxRelocsStripped = 46;
        public const int IdxOverlayRatio = 47;

        private static readonly DateTime EarliestPlausibleLink = new DateTime(1995, 1, 1, 0, 0, 0, DateTimeKind.Utc);

        public static readonly string[] FeatureNames = BuildFeatureNames();

        private static string[] BuildFeatureNames()
        {
            var names = new string[Length];
            names[IdxLogSize] = "log_size";
            names[IdxHeaderEntropy] = "header_entropy";
            names[IdxMidEntropy] = "mid_entropy";
            names[IdxTailEntropy] = "tail_entropy";
            names[IdxMaxEntropy] = "max_entropy";
            names[IdxEntropyGap] = "entropy_gap";
            for (int i = 0; i < HistogramBuckets; i++)
                names[IdxHistogram + i] = "hist_" + i.ToString("00", System.Globalization.CultureInfo.InvariantCulture);
            names[IdxPrintableRatio] = "printable_ratio";
            names[IdxNullRatio] = "null_ratio";
            names[IdxIsPe] = "is_pe";
            names[IdxSectionCount] = "section_count";
            names[IdxIsDll] = "is_dll";
            names[IdxTimestampBad] = "timestamp_bad";
            names[IdxIs64Bit] = "is_64bit";
            names[IdxIsGuiSubsystem] = "is_gui_subsystem";
            names[IdxRelocsStripped] = "relocs_stripped";
            names[IdxOverlayRatio] = "overlay_ratio";
            return names;
        }

        public static float[]? Compute(string path)
        {
            long size;
            try
            {
                size = new FileInfo(path).Length;
            }
            catch (Exception ex) when (IsFileError(ex))
            {
                return null;
            }

            var vector = new float[Length];

            try
            {
                using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);

                Window head = ReadWindow(stream, 0, (int)Math.Min(WindowBytes, size));
                Window tail = size > WindowBytes ? ReadWindow(stream, size - WindowBytes, WindowBytes) : head;
                Window mid = size > (long)WindowBytes * 3
                    ? ReadWindow(stream, (size / 2) - (WindowBytes / 2), WindowBytes)
                    : head;

                Window combined = Merge(head, mid, tail);

                double headE = head.Entropy();
                double midE = mid.Entropy();
                double tailE = tail.Entropy();

                vector[IdxLogSize] = Clamp01(Math.Log10(size + 1.0) / 10.0);
                vector[IdxHeaderEntropy] = Clamp01(headE / 8.0);
                vector[IdxMidEntropy] = Clamp01(midE / 8.0);
                vector[IdxTailEntropy] = Clamp01(tailE / 8.0);
                vector[IdxMaxEntropy] = Clamp01(Math.Max(headE, Math.Max(midE, tailE)) / 8.0);
                vector[IdxEntropyGap] = Clamp01((tailE - headE + 8.0) / 16.0);

                for (int bucket = 0; bucket < HistogramBuckets; bucket++)
                    vector[IdxHistogram + bucket] = Clamp01(combined.Bucket(bucket));

                vector[IdxPrintableRatio] = Clamp01(combined.PrintableRatio());
                vector[IdxNullRatio] = Clamp01(combined.NullRatio());

                PeInfo pe = ReadPe(stream, size);

                vector[IdxIsPe] = pe.IsPe ? 1f : 0f;
                vector[IdxSectionCount] = Clamp01(pe.Sections / 16.0);
                vector[IdxIsDll] = pe.IsDll ? 1f : 0f;
                vector[IdxTimestampBad] = pe.TimestampBad ? 1f : 0f;
                vector[IdxIs64Bit] = pe.Is64Bit ? 1f : 0f;
                vector[IdxIsGuiSubsystem] = pe.IsGui ? 1f : 0f;
                vector[IdxRelocsStripped] = pe.RelocsStripped ? 1f : 0f;

                if (pe.IsPe && pe.EndOfLastSection > 0 && size > 0)
                {
                    double overlay = (double)(size - pe.EndOfLastSection) / size;
                    vector[IdxOverlayRatio] = Clamp01(overlay);
                }
            }
            catch (Exception ex) when (IsFileError(ex))
            {
                return null;
            }

            return vector;
        }

        private static bool IsFileError(Exception ex) =>
            ex is IOException || ex is UnauthorizedAccessException || ex is ArgumentException || ex is NotSupportedException || ex is System.Security.SecurityException;

        private static float Clamp01(double value)
        {
            if (double.IsNaN(value)) return 0f;
            if (value < 0.0) return 0f;
            if (value > 1.0) return 1f;
            return (float)value;
        }

        private sealed class Window
        {
            public readonly long[] Freq = new long[256];
            public long Total;

            public double Entropy()
            {
                if (Total == 0) return 0.0;
                double result = 0.0;
                for (int i = 0; i < 256; i++)
                {
                    long count = Freq[i];
                    if (count == 0) continue;
                    double p = (double)count / Total;
                    result -= p * Math.Log2(p);
                }
                return result;
            }

            public double Bucket(int index)
            {
                if (Total == 0) return 0.0;
                int start = index * 8;
                long sum = 0;
                for (int i = start; i < start + 8; i++) sum += Freq[i];
                return (double)sum / Total;
            }

            public double PrintableRatio()
            {
                if (Total == 0) return 0.0;
                long sum = 0;
                for (int i = 0x20; i < 0x7F; i++) sum += Freq[i];
                sum += Freq[0x09] + Freq[0x0A] + Freq[0x0D];
                return (double)sum / Total;
            }

            public double NullRatio() => Total == 0 ? 0.0 : (double)Freq[0] / Total;
        }

        private static Window ReadWindow(FileStream stream, long offset, int length)
        {
            var window = new Window();
            if (length <= 0) return window;

            offset = Math.Max(0, offset);
            try
            {
                stream.Seek(offset, SeekOrigin.Begin);
            }
            catch (IOException)
            {
                return window;
            }

            var buffer = new byte[65536];
            int remaining = length;
            while (remaining > 0)
            {
                int read = stream.Read(buffer, 0, Math.Min(buffer.Length, remaining));
                if (read <= 0) break;
                for (int i = 0; i < read; i++) window.Freq[buffer[i]]++;
                window.Total += read;
                remaining -= read;
            }

            return window;
        }

        private static Window Merge(params Window[] windows)
        {
            var merged = new Window();
            for (int w = 0; w < windows.Length; w++)
            {
                bool seen = false;
                for (int p = 0; p < w; p++)
                {
                    if (ReferenceEquals(windows[p], windows[w])) { seen = true; break; }
                }
                if (seen) continue;

                for (int i = 0; i < 256; i++) merged.Freq[i] += windows[w].Freq[i];
                merged.Total += windows[w].Total;
            }
            return merged;
        }

        private sealed class PeInfo
        {
            public bool IsPe;
            public int Sections;
            public bool IsDll;
            public bool TimestampBad;
            public bool Is64Bit;
            public bool IsGui;
            public bool RelocsStripped;
            public long EndOfLastSection;
        }

        private static byte[] ReadAt(FileStream stream, long offset, int count)
        {
            stream.Seek(offset, SeekOrigin.Begin);
            var buffer = new byte[count];
            int total = 0;
            while (total < count)
            {
                int read = stream.Read(buffer, total, count - total);
                if (read <= 0) break;
                total += read;
            }
            if (total == count) return buffer;
            Array.Resize(ref buffer, total);
            return buffer;
        }

        private static PeInfo ReadPe(FileStream stream, long size)
        {
            var pe = new PeInfo();

            try
            {
                if (size < 0x40) return pe;

                byte[] mz = ReadAt(stream, 0, 2);
                if (mz.Length != 2 || mz[0] != (byte)'M' || mz[1] != (byte)'Z') return pe;

                byte[] raw = ReadAt(stream, 0x3C, 4);
                if (raw.Length < 4) return pe;
                long peOffset = BinaryPrimitives.ReadInt32LittleEndian(raw);
                if (peOffset <= 0 || peOffset + 24 > size) return pe;

                byte[] coff = ReadAt(stream, peOffset, 24);
                if (coff.Length < 24 || coff[0] != (byte)'P' || coff[1] != (byte)'E' || coff[2] != 0 || coff[3] != 0) return pe;

                ushort machine = BinaryPrimitives.ReadUInt16LittleEndian(coff.AsSpan(4));
                ushort sections = BinaryPrimitives.ReadUInt16LittleEndian(coff.AsSpan(6));
                uint timestamp = BinaryPrimitives.ReadUInt32LittleEndian(coff.AsSpan(8));
                ushort optionalSize = BinaryPrimitives.ReadUInt16LittleEndian(coff.AsSpan(20));
                ushort characteristics = BinaryPrimitives.ReadUInt16LittleEndian(coff.AsSpan(22));

                pe.IsPe = true;
                pe.Sections = sections;
                pe.IsDll = (characteristics & 0x2000) != 0;
                pe.RelocsStripped = (characteristics & 0x0001) != 0;
                pe.Is64Bit = machine == 0x8664 || machine == 0xAA64;

                if (timestamp == 0)
                {
                    pe.TimestampBad = true;
                }
                else
                {
                    DateTime linked = DateTime.UnixEpoch.AddSeconds(timestamp);
                    DateTime now = DateTime.UtcNow;
                    pe.TimestampBad = linked < EarliestPlausibleLink || Math.Floor((linked - now).TotalDays) > 1;
                }

                long optionalOffset = peOffset + 24;
                if (optionalSize >= 70 && optionalOffset + 70 <= size)
                {
                    byte[] subsystem = ReadAt(stream, optionalOffset + 68, 2);
                    if (subsystem.Length == 2)
                        pe.IsGui = BinaryPrimitives.ReadUInt16LittleEndian(subsystem) == 2;
                }

                long sectionTable = optionalOffset + optionalSize;
                long end = 0;
                int limit = Math.Min((int)sections, 96);
                for (int i = 0; i < limit; i++)
                {
                    long entry = sectionTable + (long)i * 40;
                    if (entry + 40 > size) break;

                    byte[] sec = ReadAt(stream, entry + 16, 8);
                    if (sec.Length < 8) break;

                    long sizeOfRaw = BinaryPrimitives.ReadUInt32LittleEndian(sec.AsSpan(0));
                    long pointerToRaw = BinaryPrimitives.ReadUInt32LittleEndian(sec.AsSpan(4));
                    long candidate = pointerToRaw + sizeOfRaw;
                    if (end < candidate && candidate <= size) end = candidate;
                }

                pe.EndOfLastSection = end;
                return pe;
            }
            catch (IOException)
            {
                return pe;
            }
        }
    }
}
