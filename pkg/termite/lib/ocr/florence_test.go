package ocr

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFlorencePrompt(t *testing.T) {
	tests := []struct {
		task     FlorenceTask
		expected string
	}{
		{FlorenceCaption, "<CAPTION>"},
		{FlorenceDetailedCaption, "<DETAILED_CAPTION>"},
		{FlorenceMoreDetailedCaption, "<MORE_DETAILED_CAPTION>"},
		{FlorenceOCR, "<OCR>"},
		{FlorenceOCRWithRegion, "<OCR_WITH_REGION>"},
		{FlorenceObjectDetection, "<OD>"},
		{FlorenceDenseRegionCaption, "<DENSE_REGION_CAPTION>"},
		{FlorenceRegionProposal, "<REGION_PROPOSAL>"},
		{FlorenceCaptionToGrounding, "<CAPTION_TO_PHRASE_GROUNDING>"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			assert.Equal(t, tt.expected, FlorencePrompt(tt.task))
		})
	}
}

func TestFlorenceGroundingPrompt(t *testing.T) {
	caption := "A green car parked in front of a yellow building."
	expected := "<CAPTION_TO_PHRASE_GROUNDING>A green car parked in front of a yellow building."
	assert.Equal(t, expected, FlorenceGroundingPrompt(caption))
}

func TestFlorenceDocVQAPrompt(t *testing.T) {
	question := "What is the total amount?"
	expected := "<DocVQA>What is the total amount?"
	assert.Equal(t, expected, FlorenceDocVQAPrompt(question))
}

func TestFlorenceParseOCR(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"Hello World", "Hello World"},
		{"  trimmed  ", "trimmed"},
		{"Line1\nLine2", "Line1\nLine2"},
	}

	for _, tt := range tests {
		result := FlorenceParseOCR(tt.input)
		assert.Equal(t, tt.expected, result)
	}
}

func TestFlorenceTaskFromString(t *testing.T) {
	tests := []struct {
		input    string
		expected FlorenceTask
		hasError bool
	}{
		{"<CAPTION>", FlorenceCaption, false},
		{"<OCR>", FlorenceOCR, false},
		{"<INVALID>", "", true},
	}

	for _, tt := range tests {
		result, err := FlorenceTaskFromString(tt.input)
		if tt.hasError {
			assert.Error(t, err)
		} else {
			assert.NoError(t, err)
			assert.Equal(t, tt.expected, result)
		}
	}
}

func TestFlorenceTaskString(t *testing.T) {
	task := FlorenceOCR
	assert.Equal(t, "<OCR>", task.String())
}
