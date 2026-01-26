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
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/ajroetker/go-highway/hwy/contrib/nn"
	"github.com/ajroetker/go-highway/hwy/contrib/vec"

	"github.com/antflydb/termite/pkg/termite/lib/backends"
)

// DecoderState holds the state for autoregressive decoding.
type DecoderState struct {
	// InputIDs are the current token IDs being decoded.
	InputIDs []int32
	// KVCache holds the key-value cache from previous steps.
	KVCache *backends.KVCache
	// GeneratedTokens are the tokens generated so far (excluding prompt).
	GeneratedTokens []int32
}

// DecoderStepFunc is called for each decoding step.
// It receives the current state and returns logits for the next token and updated KV-cache.
// The inputIDs in state may be the full sequence or just the last token (for KV-cached models).
type DecoderStepFunc func(ctx context.Context, state *DecoderState) (logits []float32, newKVCache *backends.KVCache, err error)

// Generator handles autoregressive text generation.
// It implements the generation loop with support for greedy decoding, sampling,
// top-k, top-p (nucleus), temperature, and repetition penalty.
type Generator struct {
	Config *backends.GenerationConfig

	// EOSTokenID is the end-of-sequence token ID.
	EOSTokenID int32
	// BOSTokenID is the beginning-of-sequence token ID.
	BOSTokenID int32
	// PadTokenID is the padding token ID.
	PadTokenID int32

	// Debug enables debug logging during generation.
	Debug bool
}

// NewGenerator creates a new Generator with the given configuration.
func NewGenerator(config *backends.GenerationConfig, eosID, bosID, padID int32) *Generator {
	if config == nil {
		config = backends.DefaultGenerationConfig()
	}
	return &Generator{
		Config:     config,
		EOSTokenID: eosID,
		BOSTokenID: bosID,
		PadTokenID: padID,
	}
}

// NewGeneratorFromDecoderConfig creates a Generator from a DecoderConfig.
func NewGeneratorFromDecoderConfig(genConfig *backends.GenerationConfig, decConfig *backends.DecoderConfig) *Generator {
	if genConfig == nil {
		genConfig = backends.DefaultGenerationConfig()
	}
	return &Generator{
		Config:     genConfig,
		EOSTokenID: decConfig.EOSTokenID,
		BOSTokenID: decConfig.BOSTokenID,
		PadTokenID: decConfig.PadTokenID,
	}
}

// GenerateResult holds the result of generation.
type GenerateResult struct {
	// TokenIDs are all generated token IDs.
	TokenIDs []int32
	// FinalKVCache is the KV-cache after generation (for continued generation).
	FinalKVCache *backends.KVCache
	// StoppedAtEOS indicates whether generation stopped due to EOS token.
	StoppedAtEOS bool
	// LogProb is the cumulative log probability of the generated sequence.
	// This is the sum of log probabilities for each token selected.
	LogProb float64
}

// Generate performs autoregressive generation.
// startTokens are the initial tokens (e.g., BOS token or prompt tokens).
// stepFn is called for each decoding step to get logits.
// useKVCache indicates whether the model supports KV-caching (only pass last token after first step).
func (g *Generator) Generate(
	ctx context.Context,
	startTokens []int32,
	stepFn DecoderStepFunc,
	useKVCache bool,
) (*GenerateResult, error) {
	state := &DecoderState{
		InputIDs:        make([]int32, len(startTokens)),
		GeneratedTokens: make([]int32, 0, g.Config.MaxNewTokens),
	}
	copy(state.InputIDs, startTokens)

	// Don't include BOS/start tokens in generated tokens
	// but do include any prompt tokens after the start token
	if len(startTokens) > 1 {
		state.GeneratedTokens = append(state.GeneratedTokens, startTokens[1:]...)
	}

	result := &GenerateResult{}
	var cumulativeLogProb float64

	for i := 0; i < g.Config.MaxNewTokens; i++ {
		select {
		case <-ctx.Done():
			result.TokenIDs = state.GeneratedTokens
			result.FinalKVCache = state.KVCache
			result.LogProb = cumulativeLogProb
			return result, ctx.Err()
		default:
		}

		// For KV-cached models, only pass the last token after the first step
		stepState := state
		if useKVCache && state.KVCache != nil && state.KVCache.SeqLen > 0 {
			stepState = &DecoderState{
				InputIDs:        state.InputIDs[len(state.InputIDs)-1:],
				KVCache:         state.KVCache,
				GeneratedTokens: state.GeneratedTokens,
			}
		}

		// Get logits from decoder
		logits, newKVCache, err := stepFn(ctx, stepState)
		if err != nil {
			result.TokenIDs = state.GeneratedTokens
			result.FinalKVCache = state.KVCache
			result.LogProb = cumulativeLogProb
			return result, err
		}

		// Debug: print top 5 tokens and their logits
		if g.Debug {
			type tokenScore struct {
				id    int
				score float32
			}
			topN := make([]tokenScore, 0, len(logits))
			for idx, score := range logits {
				topN = append(topN, tokenScore{idx, score})
			}
			sort.Slice(topN, func(i, j int) bool { return topN[i].score > topN[j].score })
			if len(topN) > 5 {
				topN = topN[:5]
			}
			fmt.Printf("[DEBUG Generator] Step %d, inputIDs=%v, top tokens: ", i, stepState.InputIDs)
			for _, ts := range topN {
				fmt.Printf("{%d: %.4f} ", ts.id, ts.score)
			}
			fmt.Printf("(EOSTokenID=%d)\n", g.EOSTokenID)
		}

		// Select next token and get its log probability
		nextToken, logProb := g.selectNextTokenWithProb(logits, state.GeneratedTokens)
		cumulativeLogProb += logProb

		if g.Debug {
			fmt.Printf("[DEBUG Generator] Step %d: selected token %d, logProb=%.4f, cumLogProb=%.4f, generatedTokens=%d, minLength=%d\n",
				i, nextToken, logProb, cumulativeLogProb, len(state.GeneratedTokens), g.Config.MinLength)
		}

		// Check for EOS
		if nextToken == g.EOSTokenID {
			// Check minimum length
			if len(state.GeneratedTokens) >= g.Config.MinLength {
				result.TokenIDs = state.GeneratedTokens
				result.FinalKVCache = newKVCache
				result.StoppedAtEOS = true
				result.LogProb = cumulativeLogProb
				return result, nil
			}
			// Force continue - set EOS logits to -inf and resample
			logits[g.EOSTokenID] = float32(math.Inf(-1))
			oldLogProb := logProb
			nextToken, logProb = g.selectNextTokenWithProb(logits, state.GeneratedTokens)
			// Replace the EOS log prob with the new token's log prob
			cumulativeLogProb = cumulativeLogProb - oldLogProb + logProb
		}

		// Append token
		state.GeneratedTokens = append(state.GeneratedTokens, nextToken)
		state.InputIDs = append(state.InputIDs, nextToken)
		state.KVCache = newKVCache
	}

	result.TokenIDs = state.GeneratedTokens
	result.FinalKVCache = state.KVCache
	result.LogProb = cumulativeLogProb
	return result, nil
}

// GenerateStreaming performs autoregressive generation with streaming output.
// The callback is called for each generated token. Return false to stop generation.
func (g *Generator) GenerateStreaming(
	ctx context.Context,
	startTokens []int32,
	stepFn DecoderStepFunc,
	useKVCache bool,
	callback func(token int32) bool,
) (*GenerateResult, error) {
	state := &DecoderState{
		InputIDs:        make([]int32, len(startTokens)),
		GeneratedTokens: make([]int32, 0, g.Config.MaxNewTokens),
	}
	copy(state.InputIDs, startTokens)

	result := &GenerateResult{}

	for i := 0; i < g.Config.MaxNewTokens; i++ {
		select {
		case <-ctx.Done():
			result.TokenIDs = state.GeneratedTokens
			result.FinalKVCache = state.KVCache
			return result, ctx.Err()
		default:
		}

		// For KV-cached models, only pass the last token after the first step
		stepState := state
		if useKVCache && state.KVCache != nil && state.KVCache.SeqLen > 0 {
			stepState = &DecoderState{
				InputIDs:        state.InputIDs[len(state.InputIDs)-1:],
				KVCache:         state.KVCache,
				GeneratedTokens: state.GeneratedTokens,
			}
		}

		// Get logits from decoder
		logits, newKVCache, err := stepFn(ctx, stepState)
		if err != nil {
			result.TokenIDs = state.GeneratedTokens
			result.FinalKVCache = state.KVCache
			return result, err
		}

		// Select next token
		nextToken := g.selectNextToken(logits, state.GeneratedTokens)

		// Check for EOS
		if nextToken == g.EOSTokenID {
			if len(state.GeneratedTokens) >= g.Config.MinLength {
				result.TokenIDs = state.GeneratedTokens
				result.FinalKVCache = newKVCache
				result.StoppedAtEOS = true
				return result, nil
			}
			logits[g.EOSTokenID] = float32(math.Inf(-1))
			nextToken = g.selectNextToken(logits, state.GeneratedTokens)
		}

		// Callback for streaming
		if !callback(nextToken) {
			result.TokenIDs = state.GeneratedTokens
			result.FinalKVCache = newKVCache
			return result, nil
		}

		// Append token
		state.GeneratedTokens = append(state.GeneratedTokens, nextToken)
		state.InputIDs = append(state.InputIDs, nextToken)
		state.KVCache = newKVCache
	}

	result.TokenIDs = state.GeneratedTokens
	result.FinalKVCache = state.KVCache
	return result, nil
}

// selectNextToken selects the next token based on generation config.
func (g *Generator) selectNextToken(logits []float32, generatedTokens []int32) int32 {
	token, _ := g.selectNextTokenWithProb(logits, generatedTokens)
	return token
}

// selectNextTokenWithProb selects the next token and returns its log probability.
// The log probability is computed from the softmax of the (possibly modified) logits.
func (g *Generator) selectNextTokenWithProb(logits []float32, generatedTokens []int32) (int32, float64) {
	// Make a copy to avoid modifying the original
	logitsCopy := make([]float32, len(logits))
	copy(logitsCopy, logits)

	// Apply repetition penalty
	if g.Config.RepetitionPenalty != 1.0 {
		applyRepetitionPenalty(logitsCopy, generatedTokens, g.Config.RepetitionPenalty)
	}

	// Convert to probabilities (before top-k/top-p filtering for accurate probability)
	probs := Softmax(logitsCopy)

	var token int32
	if !g.Config.DoSample {
		// Greedy decoding
		token = Argmax(logitsCopy)
	} else {
		// Sampling with temperature
		if g.Config.Temperature != 1.0 && g.Config.Temperature > 0 {
			for i := range logitsCopy {
				logitsCopy[i] /= g.Config.Temperature
			}
			// Recompute probs with temperature
			probs = Softmax(logitsCopy)
		}

		// Apply top-k
		filteredProbs := probs
		if g.Config.TopK > 0 && g.Config.TopK < len(filteredProbs) {
			filteredProbs = TopK(filteredProbs, g.Config.TopK)
		}

		// Apply top-p (nucleus sampling)
		if g.Config.TopP < 1.0 && g.Config.TopP > 0 {
			filteredProbs = TopP(filteredProbs, g.Config.TopP)
		}

		// Sample from distribution
		token = Sample(filteredProbs)
	}

	// Get the probability of the selected token (from original probs, not filtered)
	prob := float64(probs[token])
	logProb := math.Log(prob + 1e-10) // Add small epsilon to avoid log(0)

	return token, logProb
}

// applyRepetitionPenalty applies repetition penalty to logits.
func applyRepetitionPenalty(logits []float32, generatedTokens []int32, penalty float32) {
	for _, tok := range generatedTokens {
		if int(tok) < len(logits) {
			if logits[tok] > 0 {
				logits[tok] /= penalty
			} else {
				logits[tok] *= penalty
			}
		}
	}
}

// Argmax returns the index of the maximum value using SIMD acceleration.
// This is particularly beneficial for decoder vocab sizes (30k-100k elements).
func Argmax(values []float32) int32 {
	if len(values) == 0 {
		return 0
	}
	return int32(vec.Argmax(values))
}

// Softmax applies softmax normalization using SIMD acceleration.
func Softmax(logits []float32) []float32 {
	if len(logits) == 0 {
		return nil
	}
	probs := make([]float32, len(logits))
	nn.Softmax(logits, probs)
	return probs
}

// TopK zeros out all but the top k probabilities and renormalizes.
func TopK(probs []float32, k int) []float32 {
	if k >= len(probs) {
		return probs
	}

	// Find the k-th largest value using partial sort
	sorted := make([]float32, len(probs))
	copy(sorted, probs)

	for i := 0; i < k && i < len(sorted); i++ {
		maxIdx := i
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j] > sorted[maxIdx] {
				maxIdx = j
			}
		}
		sorted[i], sorted[maxIdx] = sorted[maxIdx], sorted[i]
	}

	threshold := sorted[k-1]

	// Zero out values below threshold and renormalize
	result := make([]float32, len(probs))
	var sum float32
	for i, p := range probs {
		if p >= threshold {
			result[i] = p
			sum += p
		}
	}

	if sum > 0 {
		for i := range result {
			result[i] /= sum
		}
	}

	return result
}

// TopP applies nucleus sampling (top-p) and renormalizes.
func TopP(probs []float32, p float32) []float32 {
	// Create index-probability pairs and sort by probability descending
	type indexProb struct {
		idx  int
		prob float32
	}
	pairs := make([]indexProb, len(probs))
	for i, prob := range probs {
		pairs[i] = indexProb{i, prob}
	}

	// Sort descending by probability
	for i := 0; i < len(pairs); i++ {
		maxIdx := i
		for j := i + 1; j < len(pairs); j++ {
			if pairs[j].prob > pairs[maxIdx].prob {
				maxIdx = j
			}
		}
		pairs[i], pairs[maxIdx] = pairs[maxIdx], pairs[i]
	}

	// Find cutoff
	var cumSum float32
	cutoff := len(pairs)
	for i, pair := range pairs {
		cumSum += pair.prob
		if cumSum >= p {
			cutoff = i + 1
			break
		}
	}

	// Create result with only top-p tokens
	result := make([]float32, len(probs))
	var sum float32
	for i := 0; i < cutoff; i++ {
		result[pairs[i].idx] = pairs[i].prob
		sum += pairs[i].prob
	}

	// Renormalize
	if sum > 0 {
		for i := range result {
			result[i] /= sum
		}
	}

	return result
}

// Sample samples from a probability distribution.
func Sample(probs []float32) int32 {
	r := rand.Float32()
	var cumSum float32
	for i, p := range probs {
		cumSum += p
		if r < cumSum {
			return int32(i)
		}
	}
	return int32(len(probs) - 1)
}
