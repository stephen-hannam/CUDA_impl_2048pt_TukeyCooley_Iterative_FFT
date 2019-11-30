using System;


namespace DigitalMusicAnalysis
{
    public class musicNote
    {
        public enum notePitch { Ab, A_, Bb, B_, C_, Db, D_, Eb, E_, F_, Gb, G_ }
        public int pitch;
        public double duration;
        public bool flat;
        public double error;
        public int staffPos;
        public int mult;
        public double frequency;

        public musicNote(double freq, double dur)
        {
            frequency = freq;
            duration = dur;
            double freqPitch = (Math.Log((freq / 110), 2) * 12 + 1);

            if ((Math.Ceiling(freqPitch) - freqPitch) >= (freqPitch - Math.Floor(freqPitch)))
            {
                pitch = (int)Math.Floor(freqPitch);
                error = (freqPitch - Math.Floor(freqPitch));
            }
            else
            {
                pitch = (int)Math.Ceiling(freqPitch);
                error = (freqPitch - Math.Ceiling(freqPitch));
            }

            if (pitch%12 == 0 || pitch%12 == 2 || pitch%12 == 5 || pitch%12 == 7 || pitch%12 == 10)
            {
                flat = true;
            }

            mult = (pitch - pitch % 12) / 12;

            switch (pitch%12)
            {
                case 0:
                    staffPos = 7*mult;
                    break;

                case 1:
                    staffPos = 7 * mult;
                    break;

                case 2:
                    staffPos = 1 + 7 * mult;
                    break;

                case 3:
                    staffPos = 1 + 7 * mult;
                    break;

                case 4:
                    staffPos = 2 + 7 * mult;
                    break;

                case 5:
                    staffPos = 3 + 7 * mult;
                    break;

                case 6:
                    staffPos = 3 + 7 * mult;
                    break;

                case 7:
                    staffPos = 4 + 7 * mult;
                    break;

                case 8:
                    staffPos = 4 + 7 * mult;
                    break;

                case 9:
                    staffPos = 5 + 7 * mult;
                    break;

                case 10:
                    staffPos = 6 + 7 * mult;
                    break;

                case 11:
                    staffPos = 6 + 7 * mult;
                    break;
            }
        }
    }

    

}
