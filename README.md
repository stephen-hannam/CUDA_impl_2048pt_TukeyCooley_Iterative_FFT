# CUDA Implementation of Tukey-Cooley Iterative FFT

Taking a Digital Music Analysis single-threaded application (produces a spectrogram from a wave file) written in C#, and applying methods of parallelization to speed it up.

Undertaken during CAB401: High Performance Computing while completing a Bachelor of Electrical Engineering (Honours) with a Second Major in Computer/Software at the Queensland University of Technology - a four-year degree.

### Includes

1. Original source for the single-threaded application
2. Modified source code of said original application used to verify spectrogram returned from the GPU

	-- modified code is in the form of commented out lines of code

3. Folder with Visual Studio project files and directories or above said application
4. Complete CUDA source code for the single-precision kernel
5. Incomplete CUDA source code for 11th hour attempt at a double-precision kernel
6. The .wav and related .xml file of the audio I used to verify the kernel
7. The full .pdf report submitted at the end of the assignment

### Requirements/Dependencies

1. An NVidia GPU of compute capability 5.0 or higher
2. CUDA Toolkit and NVCC compiler v9.x
3. Visual Studio 2015

NB 1: This was an exercise in parallelizing an algorithm using CUDA. Making the Digital Music Analysis program run faster is the premise but it is an educational contrivance. In an industry setting, the optimal solution would be to use FFTW or cuFFT and focus on application and presentation layers at a higher level; this was done in a facile way with FFTW in order to compare a best-possible CPU only performance to the GPU performance.

NB 2: **It is for 2048pt FFT to single floating point precision only**. This would not be useful in a production solution. As stated, this parallelization is an educational exercise, and first and foremost concerned with transforming a sequential algorithm into a parallel one.

![](https://i.imgur.com/o0mCPzX.png)

![](https://i.imgur.com/vNb6rQp.png)

```
BIT-REVERSE-COPY(a, A)
n = length [a]
for k = 0 to n-1
  A[k] = a[reverseBits(k)]

ITERATIVE-FFT
BIT-REVERSE-COPY(a, A)
n = length(a)
for s = 1 to log2(n) do
  m = 2**s
  w_m = e**(-2*PI*i/m)
  for k = 0 to n - 1 step m do
    w = 1
    for j = 0 to m/2 - 1 do
      t = w*A[k + j + m/2]
      w = w*w_m;
      u = A[k + j]
      A[k + j] = u + t
      A[k + j + m/2] = u - t
return A
```

![](https://i.imgur.com/rnh7Pbh.png)

![](https://i.imgur.com/shFrCYd.png)

![](https://i.imgur.com/LvEvK2l.png)

![](https://i.imgur.com/FzVa57u.png)
