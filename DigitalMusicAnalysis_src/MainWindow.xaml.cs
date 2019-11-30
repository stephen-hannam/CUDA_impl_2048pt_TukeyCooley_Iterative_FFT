using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.IO;
using System.Threading;
using System.Numerics;
using NAudio.Wave;
using System.Xml;

namespace DigitalMusicAnalysis
{
    public partial class MainWindow : Window
    {
        private WaveFileReader waveReader;
        private wavefile waveIn;
        private timefreq stftRep;
        private float[] pixelArray;
        private musicNote[] sheetmusic;
        private WaveOut playback; // = new WaveOut();
        private Complex[] twiddles;
        private string filename;
        private enum pitchConv { C, Db, D, Eb, E, F, Gb, G, Ab, A, Bb, B };
        private double bpm = 70;

        public MainWindow()
        {
            InitializeComponent();
            filename = openFile("Select Audio (wav) file");
            string xmlfile = openFile("Select Score (xml) file");
            Thread check = new Thread(new ThreadStart(updateSlider));
            loadWave(filename);
            freqDomain();
            sheetmusic = readXML(xmlfile);
            onsetDetection();
            loadImage();
            loadHistogram();
            playBack();
            check.Start();

            button1.Click += zoomIN;
            button2.Click += zoomOUT;

            slider1.ValueChanged += updateHistogram;
            playback.PlaybackStopped += closeMusic;
        }

        // Loads time-freq image for tab 1

        private void loadImage()
        {

            float fs = waveIn.SampleRate;
            float divisor = fs / stftRep.wSamp;

            int rows = stftRep.wSamp / 2;
            int cols = stftRep.timeFreqData[0].Length;

            slider1.Maximum = cols - 1;

            int croppedHeight = 3520 * stftRep.wSamp / (int)fs;

            showImage.Height = croppedHeight;
            showImage.Width = cols;

            BitmapSource bmpSrc = BitmapSource.Create(cols, croppedHeight, 96, 96, PixelFormats.Gray32Float, null, pixelArray, 4 * cols);
            showImage.Source = bmpSrc;

            var scaler = (ScaleTransform)showImage.LayoutTransform;
            scaler.ScaleY = 600 / (double)croppedHeight;

        }

        // Loads Histogram for tab 2

        private void loadHistogram()
        {
            // HISTOGRAM

            float fs = waveIn.SampleRate;
            float divisor = fs / stftRep.wSamp;

            noteGraph LOWEST = new noteGraph(110, divisor);
            noteGraph LOW = new noteGraph(220, divisor);
            noteGraph MIDDLE = new noteGraph(440, divisor);
            noteGraph HIGH = new noteGraph(880, divisor);
            noteGraph HIGHEST = new noteGraph(1760, divisor);

            int rows = stftRep.wSamp / 2;

            float[] column = new float[rows];

            for (int i = 0; i < rows; i++)
            {
                column[i] = stftRep.timeFreqData[i][(int)Math.Floor(slider1.Value)];
            }

            LOWEST.setRectHeights(column);
            LOW.setRectHeights(column);
            MIDDLE.setRectHeights(column);
            HIGH.setRectHeights(column);
            HIGHEST.setRectHeights(column);

            double[] maxi = new double[5];

            maxi[0] = LOWEST.heights.Max() / 60;
            maxi[1] = LOW.heights.Max() / 60;
            maxi[2] = MIDDLE.heights.Max() / 60;
            maxi[3] = HIGH.heights.Max() / 60;
            maxi[4] = HIGHEST.heights.Max() / 60;

            double absMax = maxi.Max();

            // DYNAMIC RECTANGLES

            SolidColorBrush myBrush = new SolidColorBrush(Colors.Black);

            //Lowest Octif

            Rectangle[] lowestRects = null;
            lowestRects = new Rectangle[(int)Math.Floor(110 / divisor)];
            for (int ii = 0; ii < (int)Math.Floor(110 / divisor); ii++)
            {
                double lowestMarg = Math.Log(((110 + ii * divisor) / 110), 2) * 240;

                lowestRects[ii] = new Rectangle();
                lowestRects[ii].Fill = myBrush;
                lowestRects[ii].Width = 1;
                lowestRects[ii].Height = LOWEST.heights[ii] / absMax; //maxi[0];
                lowestRects[ii].Margin = new Thickness(10 + lowestMarg, 0, 0, 0);
                LowestOctif.Children.Insert(ii, lowestRects[ii]);
            }

            //Low Octif

            Rectangle[] lowRects = null;
            lowRects = new Rectangle[(int)Math.Floor(220 / divisor)];
            for (int ii = 0; ii < (int)Math.Floor(220 / divisor); ii++)
            {
                double lowMarg = Math.Log(((220 + ii * divisor) / 220), 2) * 240;

                lowRects[ii] = new Rectangle();
                lowRects[ii].Fill = myBrush;
                lowRects[ii].Width = 1;
                lowRects[ii].Height = LOW.heights[ii] / absMax; //maxi[1];
                lowRects[ii].Margin = new Thickness(10 + lowMarg, 0, 0, 0);
                LowOctif.Children.Insert(ii, lowRects[ii]);
            }

            //Middle Octif

            Rectangle[] midRects = null;
            midRects = new Rectangle[(int)Math.Floor(440 / divisor)];

            for (int ii = 0; ii < (int)Math.Floor(440 / divisor); ii++)
            {
                double midMarg = Math.Log(((440 + ii * divisor) / 440), 2) * 240;

                midRects[ii] = new Rectangle();
                midRects[ii].Fill = myBrush;
                midRects[ii].Width = 1;
                midRects[ii].Height = MIDDLE.heights[ii] / absMax; //maxi[2];
                midRects[ii].Margin = new Thickness(10 + midMarg, 0, 0, 0);
                MiddleOctif.Children.Insert(ii, midRects[ii]);
            }

            //High Octif

            Rectangle[] highRects = null;
            highRects = new Rectangle[(int)Math.Floor(880 / divisor)];

            for (int ii = 0; ii < (int)Math.Floor(880 / divisor); ii++)
            {
                double highMarg = Math.Log(((880 + ii * divisor) / 880), 2) * 240;

                highRects[ii] = new Rectangle();
                highRects[ii].Fill = myBrush;
                highRects[ii].Width = 1;
                highRects[ii].Height = HIGH.heights[ii] / absMax; //maxi[3];
                highRects[ii].Margin = new Thickness(10 + highMarg, 0, 0, 0);
                HighOctif.Children.Insert(ii, highRects[ii]);
            }

            //Highest Octif

            Rectangle[] highestRects = null;
            highestRects = new Rectangle[(int)Math.Floor(1760 / divisor)];

            for (int ii = 0; ii < (int)Math.Floor(1760 / divisor); ii++)
            {
                double highestMarg = Math.Log(((1760 + ii * divisor) / 1760), 2) * 240;

                highestRects[ii] = new Rectangle();
                highestRects[ii].Fill = myBrush;
                highestRects[ii].Width = 1;
                highestRects[ii].Height = HIGHEST.heights[ii] / absMax; //maxi[4];
                highestRects[ii].Margin = new Thickness(10 + highestMarg, 0, 0, 0);
                HighestOctif.Children.Insert(ii, highestRects[ii]);
            }

        }

        // Zoom In Button control

        private void zoomIN(object sender, RoutedEventArgs e)
        {

            var z = (ScaleTransform)showImage.LayoutTransform;
            double scale = 0.1;
            z.ScaleX += scale;
            //z.ScaleY += scale;           

        }

        // Zoom Out Button control

        private void zoomOUT(object sender, RoutedEventArgs e)
        {

            var z = (ScaleTransform)showImage.LayoutTransform;
            double scale = 0.1;
            z.ScaleX -= scale;
            //z.ScaleY -= scale;

        }

        // Open File Dialog Box

        private string openFile(string title)
        {
            Microsoft.Win32.OpenFileDialog fileOpen = new Microsoft.Win32.OpenFileDialog();
            fileOpen.Title = title;
            fileOpen.ShowDialog();
            return fileOpen.FileName;
        }

        // Reads in a .wav file

        private void loadWave(string filename)
        {
            // Sound File
            FileStream file = new FileStream(filename, FileMode.Open, FileAccess.Read);
            if (file == null)
            {
                System.Console.Write("Failed to Open File!");
            }
            else
            {
                waveIn = new wavefile(file);
                waveReader = new WaveFileReader(filename);
            }

        }

        // Transforms data into Time-Frequency representation

        private void freqDomain()
        {
            stftRep = new timefreq(waveIn.wave, 2048);
            pixelArray = new float[stftRep.timeFreqData[0].Length * stftRep.wSamp / 2];
            for (int jj = 0; jj < stftRep.wSamp / 2; jj++)
            {
                for (int ii = 0; ii < stftRep.timeFreqData[0].Length; ii++)
                {
                    pixelArray[jj * stftRep.timeFreqData[0].Length + ii] = stftRep.timeFreqData[jj][ii];
                }
            }

        }

        // Onset Detection function - Determines Start and Finish times of a note and the frequency of the note over each duration.

        private void onsetDetection()
        {
            float[] HFC;
            int starts = 0;
            int stops = 0;
            Complex[] Y;
            Complex[] compX;
            double[] absY;
            List<int> lengths;
            List<int> noteStarts;
            List<int> noteStops;
            List<double> pitches;

            int ll;
            double pi = 3.14159265;
            Complex i = Complex.ImaginaryOne;

            noteStarts = new List<int>(100);
            noteStops = new List<int>(100);
            lengths = new List<int>(100);
            pitches = new List<double>(100);

            SolidColorBrush sheetBrush = new SolidColorBrush(Colors.Black);
            SolidColorBrush ErrorBrush = new SolidColorBrush(Colors.Red);
            SolidColorBrush whiteBrush = new SolidColorBrush(Colors.White);

            HFC = new float[stftRep.timeFreqData[0].Length];

            for (int jj = 0; jj < stftRep.timeFreqData[0].Length; jj++)
            {
                for (int ii = 0; ii < stftRep.wSamp / 2; ii++)
                {
                    HFC[jj] = HFC[jj] + (float)Math.Pow((double)stftRep.timeFreqData[ii][jj] * ii, 2);
                }

            }

            float maxi = HFC.Max();

            for (int jj = 0; jj < stftRep.timeFreqData[0].Length; jj++)
            {
                HFC[jj] = (float)Math.Pow((HFC[jj] / maxi), 2);
            }

            for (int jj = 0; jj < stftRep.timeFreqData[0].Length; jj++)
            {
                if (starts > stops)
                {
                    if (HFC[jj] < 0.001)
                    {
                        noteStops.Add(jj * ((stftRep.wSamp - 1) / 2));
                        stops = stops + 1;
                    }
                }
                else if (starts - stops == 0)
                {
                    if (HFC[jj] > 0.001)
                    {
                        noteStarts.Add(jj * ((stftRep.wSamp - 1) / 2));
                        starts = starts + 1;
                    }

                }
            }

            if (starts > stops)
            {
                noteStops.Add(waveIn.data.Length);
            }


            // DETERMINES START AND FINISH TIME OF NOTES BASED ON ONSET DETECTION       

            ///*

         

            for (int ii = 0; ii < noteStops.Count; ii++)
            {
                lengths.Add(noteStops[ii] - noteStarts[ii]);
            }

            for (int mm = 0; mm < lengths.Count; mm++)
            {
                int nearest = (int)Math.Pow(2, Math.Ceiling(Math.Log(lengths[mm], 2)));
                twiddles = new Complex[nearest];
                for (ll = 0; ll < nearest; ll++)
                {
                    double a = 2 * pi * ll / (double)nearest;
                    twiddles[ll] = Complex.Pow(Complex.Exp(-i), (float)a);
                }

                compX = new Complex[nearest];
                for (int kk = 0; kk < nearest; kk++)
                {
                    if (kk < lengths[mm] && (noteStarts[mm] + kk) < waveIn.wave.Length)
                    {
                        compX[kk] = waveIn.wave[noteStarts[mm] + kk];
                    }
                    else
                    {
                        compX[kk] = Complex.Zero;
                    }
                }

                Y = new Complex[nearest];
                
                Y = fft(compX, nearest);
                
                absY = new double[nearest];

                double maximum = 0;
                int maxInd = 0;

                for (int jj = 0; jj < Y.Length; jj++)
                {
                    absY[jj] = Y[jj].Magnitude;
                    if (absY[jj] > maximum)
                    {
                        maximum = absY[jj];
                        maxInd = jj;
                    }
                }

                for (int div = 6; div > 1; div--)
                {

                    if (maxInd > nearest / 2)
                    {
                        if (absY[(int)Math.Floor((double)(nearest - maxInd) / div)] / absY[(maxInd)] > 0.10)
                        {
                            maxInd = (nearest - maxInd) / div;
                        }
                    }
                    else
                    {
                        if (absY[(int)Math.Floor((double)maxInd / div)] / absY[(maxInd)] > 0.10)
                        {
                            maxInd = maxInd / div;
                        }
                    }
                }

                if (maxInd > nearest / 2)
                {
                    pitches.Add((nearest - maxInd) * waveIn.SampleRate / nearest);
                }
                else
                {
                    pitches.Add(maxInd * waveIn.SampleRate / nearest);
                }


            }
            
            
            musicNote[] noteArray;
            noteArray = new musicNote[noteStarts.Count()];

            for (int ii = 0; ii < noteStarts.Count(); ii++)
            {
                noteArray[ii] = new musicNote(pitches[ii], lengths[ii]);
            }

            int[] sheetPitchArray = new int[sheetmusic.Length];
            int[] notePitchArray = new int[noteArray.Length];

            for (int ii = 0; ii < sheetmusic.Length; ii++)
            {
                sheetPitchArray[ii] = sheetmusic[ii].pitch % 12;
            }

            for (int jj = 0; jj < noteArray.Length; jj++)
            {
                notePitchArray[jj] = noteArray[jj].pitch % 12;
            }

            string[] alignedStrings = new string[2];

            alignedStrings = stringMatch(sheetPitchArray, notePitchArray);

            musicNote[] alignedStaffArray = new musicNote[alignedStrings[0].Length / 2];
            musicNote[] alignedNoteArray = new musicNote[alignedStrings[1].Length / 2];
            int staffCount = 0;
            int noteCount = 0;

            for (int ii = 0; ii < alignedStrings[0].Length / 2; ii++)
            {

                if (alignedStrings[0][2 * ii] == ' ')
                {
                    alignedStaffArray[ii] = new musicNote(0, 0);
                }
                else
                {
                    alignedStaffArray[ii] = sheetmusic[staffCount];
                    staffCount++;
                }

                if (alignedStrings[1][2 * ii] == ' ')
                {
                    alignedNoteArray[ii] = new musicNote(0, 0);
                }
                else
                {
                    alignedNoteArray[ii] = noteArray[noteCount];
                    noteCount++;
                }
            }

            // STAFF TAB DISPLAY

            Ellipse[] notes;
            Line[] stems;
            notes = new Ellipse[alignedNoteArray.Length];
            stems = new Line[alignedNoteArray.Length];
            SolidColorBrush myBrush = new SolidColorBrush(Colors.Green);

            RotateTransform rotate = new RotateTransform(45);

            for (int ii = 0; ii < alignedNoteArray.Length; ii++)
            {
                //noteArray[ii] = new musicNote(pitches[ii], lengths[ii]);
                //System.Console.Out.Write("Note " + (ii + 1) + ": \nDuration: " + noteArray[ii].duration / waveIn.SampleRate + " seconds \nPitch: " + Enum.GetName(typeof(musicNote.notePitch), (noteArray[ii].pitch) % 12) + " / " + pitches[ii] + "\nError: " + noteArray[ii].error * 100 + "%\n");
                notes[ii] = new Ellipse();
                notes[ii].Tag = alignedNoteArray[ii];
                notes[ii].Height = 20;
                notes[ii].Width = 15;
                notes[ii].Margin = new Thickness(ii * 30, 0, 0, 0);
                notes[ii].LayoutTransform = rotate;
                notes[ii].MouseEnter += DisplayStats;
                notes[ii].MouseLeave += ClearStats;
                stems[ii] = new Line();
                stems[ii].StrokeThickness = 1;
                stems[ii].X1 = ii * 30 + 20;
                stems[ii].X2 = ii * 30 + 20;
                stems[ii].Y1 = 250 - 10 * alignedNoteArray[ii].staffPos;
                stems[ii].Y2 = 250 - 10 * alignedNoteArray[ii].staffPos - 40;
                notes[ii].Fill = ErrorBrush;
                notes[ii].StrokeThickness = 1;
                stems[ii].Stroke = ErrorBrush;


                Canvas.SetTop(notes[ii], (240 - 10 * alignedNoteArray[ii].staffPos));
                if (alignedNoteArray[ii].flat)
                {
                    System.Windows.Controls.Label flat = new System.Windows.Controls.Label();
                    flat.Content = "b";
                    flat.FontFamily = new FontFamily("Mistral");
                    flat.Margin = new Thickness(ii * 30 + 15, 0, 0, 0);
                    Canvas.SetTop(flat, (240 - 10 * alignedNoteArray[ii].staffPos));
                    noteStaff.Children.Insert(ii, flat);
                }

                noteStaff.Children.Insert(ii, notes[ii]);
                noteStaff.Children.Insert(ii, stems[ii]);

            }

            Ellipse[] sheetNotes;
            Rectangle[] timeRect;
            Line[] sheetStems;
            sheetNotes = new Ellipse[alignedStaffArray.Length];
            sheetStems = new Line[alignedStaffArray.Length];
            timeRect = new Rectangle[2 * alignedStaffArray.Length];

            Fline.Width = alignedStaffArray.Length * 30;
            Dline.Width = alignedStaffArray.Length * 30;
            Bline.Width = alignedStaffArray.Length * 30;
            Gline.Width = alignedStaffArray.Length * 30;
            Eline.Width = alignedStaffArray.Length * 30;
            noteStaff.Width = alignedStaffArray.Length * 30;


            for (int ii = 0; ii < alignedStaffArray.Length; ii++)
            {

                sheetNotes[ii] = new Ellipse();
                sheetNotes[ii].Tag = alignedStaffArray[ii];
                sheetNotes[ii].Height = 20;
                sheetNotes[ii].Width = 15;
                sheetNotes[ii].Margin = new Thickness(ii * 30, 0, 0, 0);
                sheetNotes[ii].LayoutTransform = rotate;
                sheetNotes[ii].MouseEnter += DisplayStats;
                sheetNotes[ii].MouseLeave += ClearStats;
                sheetStems[ii] = new Line();
                sheetStems[ii].StrokeThickness = 1;
                sheetStems[ii].X1 = ii * 30 + 20;
                sheetStems[ii].X2 = ii * 30 + 20;
                sheetStems[ii].Y1 = 250 - 10 * alignedStaffArray[ii].staffPos;
                sheetStems[ii].Y2 = 250 - 10 * alignedStaffArray[ii].staffPos - 40;

                sheetNotes[ii].Fill = sheetBrush;
                sheetNotes[ii].StrokeThickness = 1;
                sheetStems[ii].Stroke = sheetBrush;


                Canvas.SetTop(sheetNotes[ii], (240 - 10 * alignedStaffArray[ii].staffPos));
                if (alignedStaffArray[ii].flat)
                {
                    System.Windows.Controls.Label flat = new System.Windows.Controls.Label();
                    flat.Content = "b";
                    flat.FontFamily = new FontFamily("Mistral");
                    flat.Margin = new Thickness(ii * 30 + 15, 0, 0, 0);
                    Canvas.SetTop(flat, (240 - 10 * alignedStaffArray[ii].staffPos));
                    noteStaff.Children.Insert(ii, flat);
                }
                noteStaff.Children.Insert(ii, sheetNotes[ii]);
                noteStaff.Children.Insert(ii, sheetStems[ii]);
            }

            // FOR TIMING ERROR RECTANGLES

            for (int ii = 0; ii < alignedStaffArray.Length; ii++)
            {

                timeRect[ii] = new Rectangle();
                timeRect[ii].Fill = sheetBrush;
                timeRect[ii].Height = 10 * alignedStaffArray[ii].duration * 4 * bpm / (60 * waveIn.SampleRate);
                timeRect[ii].Width = 15;
                timeRect[ii].Margin = new Thickness(ii * 30 + 5, 0, 0, 0);

                Canvas.SetTop(timeRect[ii], 200);

                noteStaff.Children.Insert(ii, timeRect[ii]);

            }

            for (int ii = alignedStaffArray.Length; ii < alignedStaffArray.Length + alignedNoteArray.Length; ii++)
            {

                timeRect[ii] = new Rectangle();
                timeRect[ii].Fill = ErrorBrush;
                timeRect[ii].Height = 10 * alignedNoteArray[ii - alignedStaffArray.Length].duration * 4 * bpm / (60 * waveIn.SampleRate);
                timeRect[ii].Width = 10;
                timeRect[ii].Margin = new Thickness((ii - alignedStaffArray.Length) * 30 + 5, 0, 0, 0);

                Canvas.SetTop(timeRect[ii], 200);
                noteStaff.Children.Insert(ii, timeRect[ii]);
            }


        }

        private void DisplayStats(object sender, System.Windows.Input.MouseEventArgs e)
        {
            Ellipse note = (Ellipse)sender;
            musicNote details = (musicNote)note.Tag;

            NoteStatsP.Text = Enum.GetName(typeof(musicNote.notePitch), (details.pitch) % 12);
            NoteStatsF.Text = details.frequency.ToString();
            NoteStatsE.Text = (details.error * 100).ToString() + "%";
            if (details.error > 0.2)
                Comments.Text = "Too sharp";
            else if (details.error < -0.2)
                Comments.Text = "Too flat";
            else
                Comments.Text = "";
        }

        private void ClearStats(object sender, System.Windows.Input.MouseEventArgs e)
        {
            NoteStatsP.Text = "";
            NoteStatsF.Text = "";
            NoteStatsE.Text = "";
            Comments.Text = "";
        }


        // Updates Histogram values in tab 2 - Octifs

        private void updateHistogram(object sender, RoutedEventArgs e)
        {
            LowestOctif.Children.Clear();
            LowOctif.Children.Clear();
            MiddleOctif.Children.Clear();
            HighOctif.Children.Clear();
            HighestOctif.Children.Clear();

            loadHistogram();

        }

        // Sets up and plays music file that was read in

        private void playBack()
        {
            playback = new WaveOut();
            playback.Init(waveReader);
            playback.Play();
        }

        // Updating thread - Gets position in music file, uses it as slider value.

        private void updateSlider()
        {

            while (playback.GetPosition() < waveIn.data.Length)
            {
                slider1.Dispatcher.BeginInvoke(new Action(delegate ()
                {
                    slider1.Value = Math.Floor((double)playback.GetPosition() / 1024);
                }));

                System.Threading.Thread.Sleep(100);

            }
            playback.Stop();
        }

        private void Window_Loaded(object sender, RoutedEventArgs e)
        {

        }

        // Slider Update Button 1 - -1

        private void button3_Click(object sender, RoutedEventArgs e)
        {
            slider1.Value--;
        }

        // Slider Update Button 2 - +1

        private void button4_Click(object sender, RoutedEventArgs e)
        {
            slider1.Value++;
        }

        // Replay sound file - File -> Replay

        private void replay(object sender, RoutedEventArgs e)
        {
            slider1.Value = 0;
            loadWave(filename);
            playBack();
        }

        private void closeMusic(object sender, EventArgs e)
        {
            playback.Dispose();
        }

        // FFT function for Pitch Detection

        private Complex[] fft(Complex[] x, int L)
        {
            int ii = 0;
            int kk = 0;
            int N = x.Length;

            Complex[] Y = new Complex[N];

            if (N == 1)
            {
                Y[0] = x[0];
            }
            else
            {

                Complex[] E = new Complex[N / 2];
                Complex[] O = new Complex[N / 2];
                Complex[] even = new Complex[N / 2];
                Complex[] odd = new Complex[N / 2];

                for (ii = 0; ii < N; ii++)
                {

                    if (ii % 2 == 0)
                    {
                        even[ii / 2] = x[ii];
                    }
                    if (ii % 2 == 1)
                    {
                        odd[(ii - 1) / 2] = x[ii];
                    }
                }

                E = fft(even, L);
                O = fft(odd, L);

                for (kk = 0; kk < N; kk++)
                {
                    Y[kk] = E[(kk % (N / 2))] + O[(kk % (N / 2))] * twiddles[kk * (L / N)];
                }
            }

            return Y;
        }

        private musicNote[] readXML(string filename)
        {

            List<string> stepList = new List<string>(100);
            List<int> octaveList = new List<int>(100);
            List<int> durationList = new List<int>(100);
            List<int> alterList = new List<int>(100);
            int noteCount = 0;
            bool sharp;
            musicNote[] scoreArray;

            FileStream file = new FileStream(filename, FileMode.Open, FileAccess.Read);
            if (file == null)
            {
                System.Console.Write("Failed to Open File!");
            }

            XmlTextReader reader = new XmlTextReader(filename);

            bool finished = false;

            while (finished == false)
            {
                sharp = false;
                while ((!reader.Name.Equals("note") || reader.NodeType == XmlNodeType.EndElement) && !finished)
                {
                    reader.Read();
                    if (reader.ReadState == ReadState.EndOfFile)
                    {
                        finished = true;
                    }
                }

                reader.Read();
                reader.Read();
                if (reader.Name.Equals("rest"))
                {
                }
                else if (reader.Name.Equals("pitch"))
                {

                    while (!reader.Name.Equals("step"))
                    {
                        reader.Read();
                    }
                    reader.Read();
                    stepList.Add(reader.Value);
                    while (!reader.Name.Equals("octave"))
                    {
                        if (reader.Name.Equals("alter") && reader.NodeType == XmlNodeType.Element)
                        {
                            reader.Read();
                            alterList.Add(int.Parse(reader.Value));
                            sharp = true;
                        }
                        reader.Read();
                    }
                    reader.Read();
                    if (!sharp)
                    {
                        alterList.Add(0);
                    }
                    sharp = false;
                    octaveList.Add(int.Parse(reader.Value));
                    while (!reader.Name.Equals("duration"))
                    {
                        reader.Read();
                    }
                    reader.Read();
                    durationList.Add(int.Parse(reader.Value));
                    //System.Console.Out.Write("Note ~ Pitch: " + stepList[noteCount] + alterList[noteCount] + " Octave: " + octaveList[noteCount] + " Duration: " + durationList[noteCount] + "\n");
                    noteCount++;

                }

            }

            scoreArray = new musicNote[noteCount];

            double c0 = 16.351625;

            for (int nn = 0; nn < noteCount; nn++)
            {
                int step = (int)Enum.Parse(typeof(pitchConv), stepList[nn]);

                double freq = c0 * Math.Pow(2, octaveList[nn]) * (Math.Pow(2, ((double)step + (double)alterList[nn]) / 12));
                scoreArray[nn] = new musicNote(freq, (double)durationList[nn] * 60 * waveIn.SampleRate / (4 * bpm));

            }

            return scoreArray;
        }

        private string[] stringMatch(string A, string B)
        {
            // SETUP SIMILARITY MATRIX
            int[][] S = new int[12][];

            for (int i = 0; i < 12; i++)
            {
                S[i] = new int[12];
            }

            for (int i = 0; i < 12; i++)
            {
                for (int j = i; j < 12; j++)
                {
                    if (i == j)
                        S[i][j] = 10;
                    else
                        S[i][j] = -Math.Abs(i - j);
                }
            }

            //GAP PENALTY

            int d = -10;

            int[][] F = new int[A.Length + 1][];

            for (int i = 0; i < A.Length + 1; i++)
            {
                F[i] = new int[B.Length + 1];
            }

            for (int j = 0; j < B.Length + 1; j++)
            {
                F[0][j] = d * j;
            }

            for (int i = 0; i < A.Length + 1; i++)
            {
                F[i][0] = d * i;
            }

            for (int i = 1; i < A.Length + 1; i++)
            {
                for (int j = 1; j < B.Length + 1; j++)
                {
                    int Ai = (int)A[i - 1] - 65;//parseChar(A[i - 1]);
                    int Bj = (int)B[j - 1] - 65;// parseChar(B[j - 1]);

                    F[i][j] = Math.Max(Math.Max((F[i - 1][j - 1] + S[Ai][Bj]), (F[i][j - 1] + d)), (F[i - 1][j] + d));
                }
            }

            string AlignA = "";
            string AlignB = "";

            int ii = (A.Length);
            int jj = (B.Length);

            while (ii > 0 && jj > 0)
            {

                int Score = F[ii][jj];
                int ScoreDiag = F[ii - 1][jj - 1];
                int ScoreUp = F[ii][jj - 1];
                int ScoreLeft = F[ii - 1][jj];

                int Ai = (int)(A[ii - 1]) - 65;
                int Bj = (int)(B[jj - 1]) - 65;

                if (Score == ScoreDiag + S[Ai][Bj])
                {
                    AlignA = A[ii - 1] + AlignA;
                    AlignB = B[jj - 1] + AlignB;

                    ii = ii - 1;
                    jj = jj - 1;

                }

                else if (Score == ScoreUp + d)
                {
                    AlignA = "-" + AlignA;
                    AlignB = B[jj - 1] + AlignB;

                    jj = jj - 1;
                }

                else if (Score == ScoreLeft + d)
                {
                    AlignA = A[ii - 1] + AlignA;
                    AlignB = "-" + AlignB;

                    ii = ii - 1;

                }
            }

            while (ii > 0)
            {
                AlignA = A[ii - 1] + AlignA;
                AlignB = "-" + AlignB;

                ii = ii - 1;
            }

            while (jj > 0)
            {
                AlignA = "-" + AlignA;
                AlignB = B[jj - 1] + AlignB;

                jj = jj - 1;
            }

            System.Console.Out.Write("Original:   " + A + "\n");
            System.Console.Out.Write("New String: " + B + "\n\n");
            System.Console.Out.Write("Optimal Alignment: \n\n");
            System.Console.Out.Write(AlignA + "\n");
            System.Console.Out.Write(AlignB + "\n");

            string[] returnArray = new string[2];

            returnArray[0] = AlignA;
            returnArray[1] = AlignB;

            return returnArray;


        }

        private string[] stringMatch(int[] A, int[] B)
        {
            // SETUP SIMILARITY MATRIX
            int[][] S = new int[12][];

            for (int i = 0; i < 12; i++)
            {
                S[i] = new int[12];
            }

            for (int i = 0; i < 12; i++)
            {
                for (int j = i; j < 12; j++)
                {
                    if (i == j)
                        S[i][j] = 10;
                    else if (Math.Abs(i - j) <= 6)
                        S[i][j] = -Math.Abs(i - j);
                    else
                        S[i][j] = Math.Abs(i - j) - 12;

                    S[j][i] = S[i][j];
                }
            }

            //GAP PENALTY

            int d = -20;

            int[][] F = new int[A.Length + 1][];

            for (int i = 0; i < A.Length + 1; i++)
            {
                F[i] = new int[B.Length + 1];
            }

            for (int j = 0; j < B.Length + 1; j++)
            {
                F[0][j] = d * j;
            }

            for (int i = 0; i < A.Length + 1; i++)
            {
                F[i][0] = d * i;
            }

            for (int i = 1; i < A.Length + 1; i++)
            {
                for (int j = 1; j < B.Length + 1; j++)
                {
                    int Ai = A[i - 1];
                    int Bj = B[j - 1];

                    F[i][j] = Math.Max(Math.Max((F[i - 1][j - 1] + S[Ai][Bj]), (F[i][j - 1] + d)), (F[i - 1][j] + d));
                }
            }

            string AlignA = "";
            string AlignB = "";

            int ii = (A.Length);
            int jj = (B.Length);

            while (ii > 0 && jj > 0)
            {

                int Score = F[ii][jj];
                int ScoreDiag = F[ii - 1][jj - 1];
                int ScoreUp = F[ii][jj - 1];
                int ScoreLeft = F[ii - 1][jj];

                int Ai = (A[ii - 1]);
                int Bj = (B[jj - 1]);

                if (Score == ScoreDiag + S[Ai][Bj])
                {
                    AlignA = Enum.GetName(typeof(musicNote.notePitch), (A[ii - 1])) + AlignA;
                    AlignB = Enum.GetName(typeof(musicNote.notePitch), (B[jj - 1])) + AlignB;

                    ii = ii - 1;
                    jj = jj - 1;

                }

                else if (Score == ScoreUp + d)
                {
                    AlignA = "  " + AlignA;
                    AlignB = Enum.GetName(typeof(musicNote.notePitch), (B[jj - 1])) + AlignB;

                    jj = jj - 1;
                }

                else if (Score == ScoreLeft + d)
                {
                    AlignA = Enum.GetName(typeof(musicNote.notePitch), (A[ii - 1])) + AlignA;
                    AlignB = "  " + AlignB;

                    ii = ii - 1;

                }
            }

            while (ii > 0)
            {
                AlignA = Enum.GetName(typeof(musicNote.notePitch), (A[ii - 1])) + AlignA;
                AlignB = "  " + AlignB;

                ii = ii - 1;
            }

            while (jj > 0)
            {
                AlignA = "  " + AlignA;
                AlignB = Enum.GetName(typeof(musicNote.notePitch), (B[jj - 1])) + AlignB;

                jj = jj - 1;
            }

            System.Console.Out.Write("\n\n----------------  String Matching ------------------\n\n");

            System.Console.Out.Write(AlignA + "\n");
            System.Console.Out.Write(AlignB + "\n");

            string[] returnArray = new string[2];

            returnArray[0] = AlignA;
            returnArray[1] = AlignB;

            return returnArray;
        }

    }

}
