package ocr

import (
	"fmt"
	"strings"
)

// Florence-2 Task Types
// Florence-2 uses natural language prompts for each task type.
// The model's processor converts task tokens to these prompts internally.

// FlorenceTask represents a Florence-2 task type
type FlorenceTask string

const (
	// FlorenceCaption generates a brief image caption
	FlorenceCaption FlorenceTask = "What does the image describe?"
	// FlorenceDetailedCaption generates a detailed image caption
	FlorenceDetailedCaption FlorenceTask = "Describe in detail what is shown in the image."
	// FlorenceMoreDetailedCaption generates a very detailed image caption
	FlorenceMoreDetailedCaption FlorenceTask = "Describe with a paragraph what is shown in the image."
	// FlorenceOCR extracts text from the image
	FlorenceOCR FlorenceTask = "What is the text in the image?"
	// FlorenceOCRWithRegion extracts text with bounding box regions
	FlorenceOCRWithRegion FlorenceTask = "What is the text in the image, with regions?"
	// FlorenceObjectDetection detects objects in the image
	FlorenceObjectDetection FlorenceTask = "Locate the objects with category name in the image."
	// FlorenceDenseRegionCaption generates captions for dense regions
	FlorenceDenseRegionCaption FlorenceTask = "Locate the objects in the image, with their descriptions."
	// FlorenceRegionProposal generates region proposals
	FlorenceRegionProposal FlorenceTask = "Locate the region proposals in the image."
	// FlorenceCaptionToGrounding grounds a caption's phrases to image regions
	// Note: This task requires input, use FlorenceGroundingPrompt() instead
	FlorenceCaptionToGrounding FlorenceTask = "Locate the phrases in the caption:"
)

// FlorencePrompt returns the task prompt string for a Florence-2 task.
func FlorencePrompt(task FlorenceTask) string {
	return string(task)
}

// FlorenceGroundingPrompt creates a grounding prompt with a caption.
// Example: FlorenceGroundingPrompt("A green car parked in front of a yellow building.")
// Returns: "<CAPTION_TO_PHRASE_GROUNDING>A green car parked in front of a yellow building."
func FlorenceGroundingPrompt(caption string) string {
	return string(FlorenceCaptionToGrounding) + caption
}

// FlorenceDocVQAPrompt creates a document VQA prompt for Florence-2-DocVQA models.
// Note: This requires a model fine-tuned for DocVQA (e.g., HuggingFaceM4/Florence-2-DocVQA).
// Example: FlorenceDocVQAPrompt("What is the total amount?")
// Returns: "<DocVQA>What is the total amount?"
func FlorenceDocVQAPrompt(question string) string {
	return "<DocVQA>" + question
}

// FlorenceParseOCR extracts text from Florence-2 OCR output.
// Florence OCR output is typically just the extracted text.
func FlorenceParseOCR(text string) string {
	return strings.TrimSpace(text)
}

// FlorenceOCRResult represents OCR output with optional bounding boxes
type FlorenceOCRResult struct {
	Text   string
	Boxes  []FlorenceBoundingBox
	Labels []string
}

// FlorenceBoundingBox represents a bounding box in normalized coordinates
type FlorenceBoundingBox struct {
	X1, Y1, X2, Y2 float64
}

// FlorenceParseOCRWithRegion parses OCR_WITH_REGION output.
// Florence returns text with location tokens like <loc_123><loc_456>...
// This is a simplified parser - full parsing requires the model's tokenizer.
func FlorenceParseOCRWithRegion(text string) FlorenceOCRResult {
	result := FlorenceOCRResult{
		Text: strings.TrimSpace(text),
	}
	// TODO: Parse location tokens to extract bounding boxes
	// Location tokens are model-specific and require tokenizer access
	return result
}

// FlorenceDetectionResult represents object detection output
type FlorenceDetectionResult struct {
	Objects []FlorenceDetectedObject
}

// FlorenceDetectedObject represents a detected object with bounding box
type FlorenceDetectedObject struct {
	Label string
	Box   FlorenceBoundingBox
	Score float64
}

// String returns the task prompt as a string
func (t FlorenceTask) String() string {
	return string(t)
}

// FlorenceTaskFromString converts a string to a FlorenceTask
// Accepts both uppercase (correct) and lowercase (legacy) formats.
func FlorenceTaskFromString(s string) (FlorenceTask, error) {
	switch s {
	case "<CAPTION>", "<cap>":
		return FlorenceCaption, nil
	case "<DETAILED_CAPTION>", "<dcap>":
		return FlorenceDetailedCaption, nil
	case "<MORE_DETAILED_CAPTION>", "<ncap>":
		return FlorenceMoreDetailedCaption, nil
	case "<OCR>", "<ocr>":
		return FlorenceOCR, nil
	case "<OCR_WITH_REGION>":
		return FlorenceOCRWithRegion, nil
	case "<OD>", "<od>":
		return FlorenceObjectDetection, nil
	case "<DENSE_REGION_CAPTION>", "<region_cap>":
		return FlorenceDenseRegionCaption, nil
	case "<REGION_PROPOSAL>", "<proposal>":
		return FlorenceRegionProposal, nil
	case "<CAPTION_TO_PHRASE_GROUNDING>", "<grounding>":
		return FlorenceCaptionToGrounding, nil
	default:
		return "", fmt.Errorf("unknown Florence-2 task: %s", s)
	}
}
