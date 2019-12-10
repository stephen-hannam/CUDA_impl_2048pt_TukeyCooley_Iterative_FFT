using System.IO;

namespace DigitalMusicAnalysis
{
    public class wavefile
    {
        public float[] wave;
        public byte[] data;
        public char[] ChunkID = new char[4];
	    public int ChunkSize;
	    public char[] Format = new char[4];
	    public char[] Subchunk1ID = new char[4];
	    public int Subchunk1Size;
	    public short AudioFormat;
	    public short NumChannels;
	    public int SampleRate;
	    public int ByteRate;
	    public short BlockAlign;
	    public short BitsPerSample;
	    public char[] Subchunk2ID = new char[4];
	    public int Subchunk2Size;

        public wavefile(FileStream file)
        {
            BinaryReader binRead = new BinaryReader(file);
            //string filename = @"C:\CAB401data\wave_out";
            //FileStream outFile = File.Create(filename);
            //BinaryWriter binWrite = new BinaryWriter(outFile);

            ChunkID =  binRead.ReadChars(4);
            ChunkSize = binRead.ReadInt32();
            Format = binRead.ReadChars(4);
            Subchunk1ID = binRead.ReadChars(4);
            Subchunk1Size = binRead.ReadInt32();
            AudioFormat = binRead.ReadInt16();
            NumChannels = binRead.ReadInt16();
            SampleRate = binRead.ReadInt32();
            ByteRate = binRead.ReadInt32();
            BlockAlign = binRead.ReadInt16();
            BitsPerSample = binRead.ReadInt16();
            Subchunk2ID = binRead.ReadChars(4);
            Subchunk2Size = binRead.ReadInt32();

            int numSamples = Subchunk2Size / (BitsPerSample / 8);
            data = new byte[numSamples];
            wave = new float[numSamples];

            data = binRead.ReadBytes(numSamples);

            //for (int i = 0; i < numSamples; i++)
            //{
            //    wave[i] = ((float)data[i] - 128) / 128;
            //    binWrite.Write(wave[i]);
            //}

            //binWrite.Flush();
            //binWrite.Close();
            //binWrite.Dispose();
        }
    }
}
