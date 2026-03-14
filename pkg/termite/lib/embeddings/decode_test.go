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

package embeddings

import (
	"bytes"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"testing"

	"github.com/antflydb/antfly/pkg/libaf/ai"
)

func TestDecodeImage_JPEG(t *testing.T) {
	data := encodeTestJPEG(t, 8, 8)

	img, err := decodeImage(data)
	if err != nil {
		t.Fatalf("decodeImage(jpeg) error: %v", err)
	}
	bounds := img.Bounds()
	if bounds.Dx() != 8 || bounds.Dy() != 8 {
		t.Errorf("got %dx%d, want 8x8", bounds.Dx(), bounds.Dy())
	}
}

func TestDecodeImage_PNG(t *testing.T) {
	data := encodeTestPNG(t, 4, 4)

	img, err := decodeImage(data)
	if err != nil {
		t.Fatalf("decodeImage(png) error: %v", err)
	}
	bounds := img.Bounds()
	if bounds.Dx() != 4 || bounds.Dy() != 4 {
		t.Errorf("got %dx%d, want 4x4", bounds.Dx(), bounds.Dy())
	}
}

func TestDecodeImage_GIF(t *testing.T) {
	data := encodeTestGIF(t, 4, 4)

	img, err := decodeImage(data)
	if err != nil {
		t.Fatalf("decodeImage(gif) error: %v", err)
	}
	bounds := img.Bounds()
	if bounds.Dx() != 4 || bounds.Dy() != 4 {
		t.Errorf("got %dx%d, want 4x4", bounds.Dx(), bounds.Dy())
	}
}

func TestDecodeImage_InvalidData(t *testing.T) {
	_, err := decodeImage([]byte("not an image"))
	if err == nil {
		t.Fatal("expected error for invalid data, got nil")
	}
}

func TestExtractContent_GIF(t *testing.T) {
	data := encodeTestGIF(t, 4, 4)

	parts := []ai.ContentPart{
		ai.BinaryContent{MIMEType: "image/gif", Data: data},
	}

	text, img, audio, err := extractContent(parts)
	if err != nil {
		t.Fatalf("extractContent(gif) error: %v", err)
	}
	if text != "" {
		t.Error("expected empty text")
	}
	if img == nil {
		t.Fatal("expected non-nil image")
	}
	if audio != nil {
		t.Error("expected nil audio")
	}
	bounds := img.Bounds()
	if bounds.Dx() != 4 || bounds.Dy() != 4 {
		t.Errorf("got %dx%d, want 4x4", bounds.Dx(), bounds.Dy())
	}
}

func TestExtractContent_TextPreferred(t *testing.T) {
	parts := []ai.ContentPart{
		ai.TextContent{Text: "hello world"},
	}

	text, img, audio, err := extractContent(parts)
	if err != nil {
		t.Fatalf("extractContent error: %v", err)
	}
	if text != "hello world" {
		t.Errorf("got text %q, want %q", text, "hello world")
	}
	if img != nil {
		t.Error("expected nil image")
	}
	if audio != nil {
		t.Error("expected nil audio")
	}
}

func testImage(w, h int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			img.Set(x, y, color.RGBA{R: uint8(x * 32), G: uint8(y * 32), B: 128, A: 255})
		}
	}
	return img
}

func encodeTestJPEG(t *testing.T, w, h int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, testImage(w, h), nil); err != nil {
		t.Fatalf("encoding test JPEG: %v", err)
	}
	return buf.Bytes()
}

func encodeTestPNG(t *testing.T, w, h int) []byte {
	t.Helper()
	var buf bytes.Buffer
	if err := png.Encode(&buf, testImage(w, h)); err != nil {
		t.Fatalf("encoding test PNG: %v", err)
	}
	return buf.Bytes()
}

func encodeTestGIF(t *testing.T, w, h int) []byte {
	t.Helper()
	img := testImage(w, h)
	palettedImg := image.NewPaletted(img.Bounds(), color.Palette{
		color.Black, color.White,
		color.RGBA{R: 255, A: 255}, color.RGBA{G: 255, A: 255},
		color.RGBA{B: 255, A: 255}, color.RGBA{R: 128, G: 128, B: 128, A: 255},
	})
	for y := range h {
		for x := range w {
			palettedImg.Set(x, y, img.At(x, y))
		}
	}
	var buf bytes.Buffer
	if err := gif.Encode(&buf, palettedImg, nil); err != nil {
		t.Fatalf("encoding test GIF: %v", err)
	}
	return buf.Bytes()
}
