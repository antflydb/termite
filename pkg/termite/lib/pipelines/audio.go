// Copyright 2025 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package pipelines

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/cmplx"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
)

// AudioProcessor handles audio loading and preprocessing for speech models.
// It converts raw audio bytes to mel spectrogram features suitable for
// encoder-decoder speech models like Whisper.
type AudioProcessor struct {
	Config *backends.AudioConfig

	// Precomputed mel filter bank
	melFilters [][]float32

	// Precomputed Hann window for FFT
	hannWindow []float32
}

// NewAudioProcessor creates a new AudioProcessor with the given configuration.
func NewAudioProcessor(config *backends.AudioConfig) *AudioProcessor {
	if config == nil {
		config = backends.DefaultAudioConfig()
	}

	ap := &AudioProcessor{
		Config: config,
	}

	// Precompute mel filter bank and Hann window
	ap.melFilters = ap.computeMelFilterBank()
	ap.hannWindow = ap.computeHannWindow()

	return ap
}

// Process converts raw audio bytes (WAV format) to mel spectrogram features.
// Returns features in shape [1, time, n_mels] suitable for batch processing.
func (ap *AudioProcessor) Process(audioData []byte) ([]float32, int, error) {
	// Load WAV file
	samples, err := ap.loadWAV(audioData)
	if err != nil {
		return nil, 0, fmt.Errorf("loading audio: %w", err)
	}

	// Compute mel spectrogram
	melSpec, numFrames := ap.computeMelSpectrogram(samples)

	return melSpec, numFrames, nil
}

// ProcessSamples converts raw audio samples (float32, mono, at target sample rate)
// to mel spectrogram features.
func (ap *AudioProcessor) ProcessSamples(samples []float32) ([]float32, int) {
	return ap.computeMelSpectrogram(samples)
}

// loadWAV parses a WAV file and returns mono float32 samples at the target sample rate.
func (ap *AudioProcessor) loadWAV(data []byte) ([]float32, error) {
	reader := bytes.NewReader(data)

	// Read RIFF header
	var riffHeader [4]byte
	if _, err := io.ReadFull(reader, riffHeader[:]); err != nil {
		return nil, fmt.Errorf("reading RIFF header: %w", err)
	}
	if string(riffHeader[:]) != "RIFF" {
		return nil, fmt.Errorf("not a RIFF file")
	}

	// Skip file size
	var fileSize uint32
	if err := binary.Read(reader, binary.LittleEndian, &fileSize); err != nil {
		return nil, fmt.Errorf("reading file size: %w", err)
	}

	// Read WAVE format
	var waveHeader [4]byte
	if _, err := io.ReadFull(reader, waveHeader[:]); err != nil {
		return nil, fmt.Errorf("reading WAVE header: %w", err)
	}
	if string(waveHeader[:]) != "WAVE" {
		return nil, fmt.Errorf("not a WAVE file")
	}

	// Parse chunks
	var audioFormat, numChannels uint16
	var sampleRate, byteRate uint32
	var blockAlign, bitsPerSample uint16
	var audioData []byte

	for {
		var chunkID [4]byte
		if _, err := io.ReadFull(reader, chunkID[:]); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("reading chunk ID: %w", err)
		}

		var chunkSize uint32
		if err := binary.Read(reader, binary.LittleEndian, &chunkSize); err != nil {
			return nil, fmt.Errorf("reading chunk size: %w", err)
		}

		switch string(chunkID[:]) {
		case "fmt ":
			if err := binary.Read(reader, binary.LittleEndian, &audioFormat); err != nil {
				return nil, fmt.Errorf("reading audio format: %w", err)
			}
			if err := binary.Read(reader, binary.LittleEndian, &numChannels); err != nil {
				return nil, fmt.Errorf("reading num channels: %w", err)
			}
			if err := binary.Read(reader, binary.LittleEndian, &sampleRate); err != nil {
				return nil, fmt.Errorf("reading sample rate: %w", err)
			}
			if err := binary.Read(reader, binary.LittleEndian, &byteRate); err != nil {
				return nil, fmt.Errorf("reading byte rate: %w", err)
			}
			if err := binary.Read(reader, binary.LittleEndian, &blockAlign); err != nil {
				return nil, fmt.Errorf("reading block align: %w", err)
			}
			if err := binary.Read(reader, binary.LittleEndian, &bitsPerSample); err != nil {
				return nil, fmt.Errorf("reading bits per sample: %w", err)
			}
			// Skip any extra format bytes
			remaining := int(chunkSize) - 16
			if remaining > 0 {
				reader.Seek(int64(remaining), io.SeekCurrent)
			}

		case "data":
			audioData = make([]byte, chunkSize)
			if _, err := io.ReadFull(reader, audioData); err != nil {
				return nil, fmt.Errorf("reading audio data: %w", err)
			}

		default:
			// Skip unknown chunks
			reader.Seek(int64(chunkSize), io.SeekCurrent)
		}
	}

	if audioData == nil {
		return nil, fmt.Errorf("no audio data found")
	}

	// Only support PCM format
	if audioFormat != 1 {
		return nil, fmt.Errorf("unsupported audio format %d (only PCM supported)", audioFormat)
	}

	// Convert to float32 samples
	samples, err := ap.bytesToSamples(audioData, int(bitsPerSample), int(numChannels))
	if err != nil {
		return nil, fmt.Errorf("converting to samples: %w", err)
	}

	// Resample if needed
	if int(sampleRate) != ap.Config.SampleRate {
		samples = ap.resample(samples, int(sampleRate), ap.Config.SampleRate)
	}

	return samples, nil
}

// bytesToSamples converts raw PCM bytes to float32 samples in range [-1, 1].
// Handles mono conversion if needed.
func (ap *AudioProcessor) bytesToSamples(data []byte, bitsPerSample, numChannels int) ([]float32, error) {
	bytesPerSample := bitsPerSample / 8
	numSamples := len(data) / (bytesPerSample * numChannels)
	samples := make([]float32, numSamples)

	reader := bytes.NewReader(data)

	for i := 0; i < numSamples; i++ {
		var sampleSum float64
		for ch := 0; ch < numChannels; ch++ {
			var sample float64
			switch bitsPerSample {
			case 8:
				var s uint8
				binary.Read(reader, binary.LittleEndian, &s)
				// 8-bit WAV is unsigned, center at 128
				sample = (float64(s) - 128) / 128.0
			case 16:
				var s int16
				binary.Read(reader, binary.LittleEndian, &s)
				sample = float64(s) / 32768.0
			case 24:
				var buf [3]byte
				reader.Read(buf[:])
				// Convert 24-bit to 32-bit signed
				s := int32(buf[0]) | int32(buf[1])<<8 | int32(buf[2])<<16
				if s&0x800000 != 0 {
					s |= -0x1000000 // Sign extend (equivalent to 0xFF000000 but fits int32)
				}
				sample = float64(s) / 8388608.0
			case 32:
				var s int32
				binary.Read(reader, binary.LittleEndian, &s)
				sample = float64(s) / 2147483648.0
			default:
				return nil, fmt.Errorf("unsupported bits per sample: %d", bitsPerSample)
			}
			sampleSum += sample
		}
		// Average channels for mono output
		samples[i] = float32(sampleSum / float64(numChannels))
	}

	return samples, nil
}

// resample performs simple linear interpolation resampling.
// For production use, consider using a proper resampling library.
func (ap *AudioProcessor) resample(samples []float32, fromRate, toRate int) []float32 {
	if fromRate == toRate {
		return samples
	}

	ratio := float64(fromRate) / float64(toRate)
	newLen := int(float64(len(samples)) / ratio)
	resampled := make([]float32, newLen)

	for i := 0; i < newLen; i++ {
		srcIdx := float64(i) * ratio
		srcIdxInt := int(srcIdx)
		frac := float32(srcIdx - float64(srcIdxInt))

		if srcIdxInt+1 < len(samples) {
			resampled[i] = samples[srcIdxInt]*(1-frac) + samples[srcIdxInt+1]*frac
		} else if srcIdxInt < len(samples) {
			resampled[i] = samples[srcIdxInt]
		}
	}

	return resampled
}

// computeMelSpectrogram converts audio samples to a mel spectrogram.
// Returns the flattened spectrogram and number of time frames.
func (ap *AudioProcessor) computeMelSpectrogram(samples []float32) ([]float32, int) {
	nFft := ap.Config.NFft
	hopLength := ap.Config.HopLength
	nMels := ap.Config.NMels

	// Pad samples if needed for chunk length
	targetLen := ap.Config.ChunkLength * ap.Config.SampleRate
	if len(samples) < targetLen {
		padded := make([]float32, targetLen)
		copy(padded, samples)
		for i := len(samples); i < targetLen; i++ {
			padded[i] = ap.Config.PaddingValue
		}
		samples = padded
	} else if len(samples) > targetLen {
		samples = samples[:targetLen]
	}

	// Whisper uses center=True STFT, which pads by nFft//2 on each side
	// This gives exactly n_samples / hop_length frames
	padAmount := nFft / 2
	paddedSamples := make([]float32, len(samples)+2*padAmount)
	// Fill with padding value (zeros)
	for i := 0; i < padAmount; i++ {
		paddedSamples[i] = ap.Config.PaddingValue
	}
	copy(paddedSamples[padAmount:], samples)
	for i := padAmount + len(samples); i < len(paddedSamples); i++ {
		paddedSamples[i] = ap.Config.PaddingValue
	}
	samples = paddedSamples

	// Calculate number of frames: exactly n_samples / hop_length for Whisper
	// (original samples length before center padding)
	numFrames := targetLen / hopLength
	if numFrames < 1 {
		numFrames = 1
	}

	// Compute STFT magnitude
	stftMag := make([][]float32, numFrames)
	nBins := nFft/2 + 1

	for frame := 0; frame < numFrames; frame++ {
		start := frame * hopLength
		end := start + nFft
		if end > len(samples) {
			end = len(samples)
		}

		// Extract frame and apply window
		frameData := make([]float32, nFft)
		for i := 0; i < nFft && start+i < len(samples); i++ {
			frameData[i] = samples[start+i] * ap.hannWindow[i]
		}

		// Compute FFT
		fftResult := ap.fft(frameData)

		// Compute magnitude spectrum
		stftMag[frame] = make([]float32, nBins)
		for i := 0; i < nBins; i++ {
			stftMag[frame][i] = float32(cmplx.Abs(fftResult[i]))
		}
	}

	// Apply mel filter bank
	melSpec := make([][]float32, numFrames)
	for frame := 0; frame < numFrames; frame++ {
		melSpec[frame] = make([]float32, nMels)
		for mel := 0; mel < nMels; mel++ {
			var sum float32
			for bin := 0; bin < nBins && bin < len(ap.melFilters[mel]); bin++ {
				sum += stftMag[frame][bin] * ap.melFilters[mel][bin]
			}
			melSpec[frame][mel] = sum
		}
	}

	// Convert to log scale (log mel spectrogram)
	for frame := 0; frame < numFrames; frame++ {
		for mel := 0; mel < nMels; mel++ {
			// Add small epsilon to avoid log(0)
			val := melSpec[frame][mel]
			if val < 1e-10 {
				val = 1e-10
			}
			melSpec[frame][mel] = float32(math.Log(float64(val)))
		}
	}

	// Flatten to [frames * mels] for model input
	// Note: Whisper expects [batch, n_mels, time] but we'll transpose in the model
	// For now, return [time, n_mels] flattened
	result := make([]float32, numFrames*nMels)
	for frame := 0; frame < numFrames; frame++ {
		for mel := 0; mel < nMels; mel++ {
			result[frame*nMels+mel] = melSpec[frame][mel]
		}
	}

	return result, numFrames
}

// computeMelFilterBank creates triangular mel filter banks.
func (ap *AudioProcessor) computeMelFilterBank() [][]float32 {
	nMels := ap.Config.NMels
	nFft := ap.Config.NFft
	sampleRate := ap.Config.SampleRate
	nBins := nFft/2 + 1

	// Frequency to mel conversion
	freqToMel := func(f float64) float64 {
		return 2595.0 * math.Log10(1.0+f/700.0)
	}
	melToFreq := func(m float64) float64 {
		return 700.0 * (math.Pow(10.0, m/2595.0) - 1.0)
	}

	// Create mel points
	lowFreq := 0.0
	highFreq := float64(sampleRate) / 2.0
	lowMel := freqToMel(lowFreq)
	highMel := freqToMel(highFreq)

	melPoints := make([]float64, nMels+2)
	for i := 0; i < nMels+2; i++ {
		melPoints[i] = lowMel + float64(i)*(highMel-lowMel)/float64(nMels+1)
	}

	// Convert to frequency and then to FFT bin indices
	binIndices := make([]int, nMels+2)
	for i := 0; i < nMels+2; i++ {
		freq := melToFreq(melPoints[i])
		binIndices[i] = int(math.Floor((float64(nFft)+1)*freq/float64(sampleRate) + 0.5))
		if binIndices[i] >= nBins {
			binIndices[i] = nBins - 1
		}
	}

	// Create filter bank
	filters := make([][]float32, nMels)
	for mel := 0; mel < nMels; mel++ {
		filters[mel] = make([]float32, nBins)
		startBin := binIndices[mel]
		centerBin := binIndices[mel+1]
		endBin := binIndices[mel+2]

		// Rising slope
		for bin := startBin; bin < centerBin; bin++ {
			if centerBin != startBin {
				filters[mel][bin] = float32(bin-startBin) / float32(centerBin-startBin)
			}
		}

		// Falling slope
		for bin := centerBin; bin <= endBin; bin++ {
			if endBin != centerBin {
				filters[mel][bin] = float32(endBin-bin) / float32(endBin-centerBin)
			}
		}
	}

	return filters
}

// computeHannWindow creates a Hann window of the given size.
func (ap *AudioProcessor) computeHannWindow() []float32 {
	n := ap.Config.NFft
	window := make([]float32, n)
	for i := 0; i < n; i++ {
		window[i] = float32(0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(n-1))))
	}
	return window
}

// fft computes the Fast Fourier Transform of the input.
// Uses the Cooley-Tukey algorithm for power-of-2 sizes.
func (ap *AudioProcessor) fft(input []float32) []complex128 {
	n := len(input)

	// Pad to next power of 2 if needed
	nextPow2 := 1
	for nextPow2 < n {
		nextPow2 *= 2
	}

	// Convert to complex
	data := make([]complex128, nextPow2)
	for i := 0; i < n; i++ {
		data[i] = complex(float64(input[i]), 0)
	}

	// Bit-reversal permutation
	j := 0
	for i := 0; i < nextPow2-1; i++ {
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
		k := nextPow2 / 2
		for k <= j {
			j -= k
			k /= 2
		}
		j += k
	}

	// Cooley-Tukey FFT
	for size := 2; size <= nextPow2; size *= 2 {
		halfSize := size / 2
		tableStep := nextPow2 / size
		for i := 0; i < nextPow2; i += size {
			for j := 0; j < halfSize; j++ {
				angle := -2 * math.Pi * float64(j*tableStep) / float64(nextPow2)
				w := complex(math.Cos(angle), math.Sin(angle))
				t := w * data[i+j+halfSize]
				data[i+j+halfSize] = data[i+j] - t
				data[i+j] = data[i+j] + t
			}
		}
	}

	return data
}

// ResolveAudioConfig returns the audio config from the model if available,
// otherwise returns the override config or defaults.
func ResolveAudioConfig(model backends.Model, override *backends.AudioConfig) *backends.AudioConfig {
	if override != nil {
		return override
	}

	// Check if model provides audio config
	type audioConfigProvider interface {
		AudioConfig() *backends.AudioConfig
	}
	if provider, ok := model.(audioConfigProvider); ok {
		if cfg := provider.AudioConfig(); cfg != nil {
			return cfg
		}
	}

	return backends.DefaultAudioConfig()
}
