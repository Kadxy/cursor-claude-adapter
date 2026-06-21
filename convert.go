package main

// convert.go — OpenAI → Anthropic request conversion.
// Translates an OpenAI Chat Completions body into an Anthropic Messages body:
// model/effort split, message normalization (system + merged sequence), tools,
// tool_choice, and per-block content conversion. Also holds tool-name recovery.

import (
	"encoding/json"
	"strings"
)

const defMaxTokens = 8192

var (
	// baseModels: the real models exposed to Cursor (names after the prefix is stripped).
	// Add a line here when a new model ships.
	baseModels = []string{"claude-opus-4-8", "claude-opus-4-7"}
	// effortOrder: thinking levels. Defines both the valid values and the display order
	// of the variants in /v1/models.
	effortOrder = []string{"low", "medium", "high", "xhigh", "max"}
)

// effortLevels is derived from effortOrder; used by splitEffort for validation.
var effortLevels = func() map[string]bool {
	m := map[string]bool{}
	for _, e := range effortOrder {
		m[e] = true
	}
	return m
}()

func toAnthropic(body map[string]any) (map[string]any, []string) {
	out := map[string]any{}
	model, _ := body["model"].(string)
	realModel, effort := splitEffort(strings.TrimPrefix(model, modelPrefix))
	out["model"] = realModel
	// A -<effort> suffix on the model name (low/medium/high/xhigh/max) turns on adaptive
	// thinking at that level.
	// e.g. cursor-claude-opus-4-8-xhigh -> claude-opus-4-8 + thinking:adaptive + effort:xhigh
	if effort != "" {
		out["thinking"] = map[string]any{"type": "adaptive"}
		out["output_config"] = map[string]any{"effort": effort}
	}

	mt := defMaxTokens
	if v, ok := numField(body, "max_tokens"); ok {
		mt = v
	} else if v, ok := numField(body, "max_output_tokens"); ok {
		mt = v
	}
	out["max_tokens"] = mt
	if s, ok := body["stream"].(bool); ok {
		out["stream"] = s
	}
	if v, ok := body["temperature"]; ok {
		out["temperature"] = v
	}
	if v, ok := body["top_p"]; ok {
		out["top_p"] = v
	}
	if v, ok := body["stop"]; ok {
		switch s := v.(type) {
		case string:
			out["stop_sequences"] = []any{s}
		case []any:
			out["stop_sequences"] = s
		}
	}

	// messages -> system + normalized message sequence (then merge adjacent same-role)
	var systemParts []string
	type rmsg struct {
		role    string
		content []any
	}
	var seq []rmsg
	msgs, _ := body["messages"].([]any)
	for _, mm := range msgs {
		m, ok := mm.(map[string]any)
		if !ok {
			continue
		}
		role, _ := m["role"].(string)
		switch role {
		case "system", "developer":
			systemParts = append(systemParts, contentToText(m["content"]))
		case "tool":
			seq = append(seq, rmsg{"user", []any{map[string]any{
				"type": "tool_result", "tool_use_id": strField(m, "tool_call_id"), "content": contentToText(m["content"]),
			}}})
		case "assistant":
			c := convertParts(m["content"])
			if tcs, ok := m["tool_calls"].([]any); ok {
				for _, tcv := range tcs {
					tc, ok := tcv.(map[string]any)
					if !ok {
						continue
					}
					fn := mapField(tc, "function")
					if fn == nil {
						fn = tc
					}
					name, _ := fn["name"].(string)
					if name == "" {
						continue
					}
					var input any = map[string]any{}
					if a, ok := fn["arguments"].(string); ok && a != "" {
						var parsed any
						if json.Unmarshal([]byte(a), &parsed) == nil {
							input = parsed
						}
					} else if a, ok := fn["arguments"]; ok && a != nil {
						input = a
					}
					c = append(c, map[string]any{"type": "tool_use", "id": tc["id"], "name": name, "input": input})
				}
			}
			if len(c) == 0 {
				continue // skip empty assistant message (an empty text block triggers 400)
			}
			seq = append(seq, rmsg{"assistant", c})
		default: // user
			c := convertParts(m["content"])
			if len(c) == 0 {
				continue // skip empty user message
			}
			seq = append(seq, rmsg{"user", c})
		}
	}
	// merge adjacent same-role messages
	var merged []map[string]any
	for _, m := range seq {
		if n := len(merged); n > 0 && merged[n-1]["role"] == m.role {
			merged[n-1]["content"] = append(merged[n-1]["content"].([]any), m.content...)
		} else {
			merged = append(merged, map[string]any{"role": m.role, "content": append([]any{}, m.content...)})
		}
	}
	// Anthropic requires at least one message, and the first must be a user message.
	if len(merged) == 0 {
		merged = []map[string]any{{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Continue."}}}}
	} else if merged[0]["role"] != "user" {
		merged = append([]map[string]any{{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Continue."}}}}, merged...)
	}
	out["messages"] = merged
	if len(systemParts) > 0 {
		out["system"] = strings.Join(systemParts, "\n\n")
	}

	// tools (accept both flat and nested shapes)
	var toolNames []string
	if tools, ok := body["tools"].([]any); ok && len(tools) > 0 {
		var at []any
		for _, tv := range tools {
			t, ok := tv.(map[string]any)
			if !ok {
				continue
			}
			fn := mapField(t, "function")
			if fn == nil {
				fn = t
			}
			name, _ := fn["name"].(string)
			if name == "" {
				continue
			}
			schema := fn["parameters"]
			if schema == nil {
				schema = fn["input_schema"]
			}
			if schema == nil {
				schema = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			desc, _ := fn["description"].(string)
			at = append(at, map[string]any{"name": name, "description": desc, "input_schema": schema})
			toolNames = append(toolNames, name)
		}
		if len(at) > 0 {
			out["tools"] = at
		}
	}
	// tool_choice
	switch tc := body["tool_choice"].(type) {
	case string:
		if tc == "required" {
			out["tool_choice"] = map[string]any{"type": "any"}
		} else if tc == "auto" {
			out["tool_choice"] = map[string]any{"type": "auto"}
		}
	case map[string]any:
		if fn := mapField(tc, "function"); fn != nil {
			if n, ok := fn["name"].(string); ok {
				out["tool_choice"] = map[string]any{"type": "tool", "name": n}
			}
		}
	}
	return out, toolNames
}

// splitEffort splits a trailing thinking-level suffix off the prefix-stripped model name.
// "claude-opus-4-8-xhigh" -> ("claude-opus-4-8", "xhigh"); "claude-opus-4-8" -> (unchanged, "")
func splitEffort(model string) (string, string) {
	if i := strings.LastIndex(model, "-"); i >= 0 && effortLevels[model[i+1:]] {
		return model[:i], model[i+1:]
	}
	return model, ""
}

// convertParts normalizes a message's content into Anthropic content blocks.
// Key point: in multi-turn tool use, Cursor puts tool_use / tool_result (and images)
// directly into the content array as native Anthropic blocks (the ungate source comment
// reads "Cursor sends these"). They must be passed through as-is, otherwise tool results
// are dropped and multi-turn round-trips break.
func convertParts(content any) []any {
	if s, ok := content.(string); ok {
		if strings.TrimSpace(s) == "" {
			return nil
		}
		return []any{map[string]any{"type": "text", "text": s}}
	}
	arr, ok := content.([]any)
	if !ok {
		return nil
	}
	var blocks []any
	for _, pv := range arr {
		p, ok := pv.(map[string]any)
		if !ok {
			if s, ok := pv.(string); ok && strings.TrimSpace(s) != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": s})
			}
			continue
		}
		switch p["type"] {
		case "tool_use", "tool_result", "image", "thinking", "redacted_thinking":
			stripCacheTTL(p)           // keep cache_control (cache breakpoint), drop the ttl that needs a beta header
			blocks = append(blocks, p) // native Anthropic block, passed through
		case "text":
			if t, ok := p["text"].(string); ok && strings.TrimSpace(t) != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": t})
			}
		case "image_url":
			if iu, ok := p["image_url"].(map[string]any); ok {
				if url, ok := iu["url"].(string); ok {
					if blk := dataURIToImage(url); blk != nil {
						blocks = append(blocks, blk)
					}
				}
			}
		}
	}
	return blocks
}

// ─────────────────── Tool-name recovery ───────────────────
// Non-streaming backends rewrite tool names to Compat<PascalName><hash>. Recover the
// original by matching against the real tool names from the request.
func makeRecover(names []string) func(string) string {
	exact := map[string]bool{}
	type c struct{ norm, orig string }
	var cands []c
	for _, n := range names {
		exact[n] = true
		cands = append(cands, c{normName(n), n})
	}
	// longest prefix first
	for i := 0; i < len(cands); i++ {
		for j := i + 1; j < len(cands); j++ {
			if len(cands[j].norm) > len(cands[i].norm) {
				cands[i], cands[j] = cands[j], cands[i]
			}
		}
	}
	return func(returned string) string {
		if returned == "" || exact[returned] {
			return returned
		}
		ns := normName(strings.TrimPrefix(returned, "Compat"))
		for _, cd := range cands {
			if cd.norm != "" && strings.HasPrefix(ns, cd.norm) {
				return cd.orig
			}
		}
		return returned
	}
}
