package main

import (
	"encoding/json"
	"testing"
)

// blockTypes collects the type of every content block in toAnthropic's output, for assertions.
func blockTypes(msgs []map[string]any) []string {
	var types []string
	for _, m := range msgs {
		for _, bv := range m["content"].([]any) {
			if b, ok := bv.(map[string]any); ok {
				types = append(types, m["role"].(string)+":"+toStr(b["type"]))
			}
		}
	}
	return types
}
func toStr(v any) string { s, _ := v.(string); return s }

func msgsOf(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	out, _ := toAnthropic(body)
	raw, _ := json.Marshal(out["messages"])
	var msgs []map[string]any
	json.Unmarshal(raw, &msgs)
	return msgs
}

func has(types []string, want string) bool {
	for _, t := range types {
		if t == want {
			return true
		}
	}
	return false
}

// Core regression: Cursor puts tool_result into the user content array as a native Anthropic
// block. Before the fix it was dropped; after the fix it must pass through to upstream.
func TestEmbeddedToolResultSurvives(t *testing.T) {
	body := map[string]any{
		"model": "cursor-claude-opus-4-8",
		"messages": []any{
			map[string]any{"role": "user", "content": "what's my ip"},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "tool_use", "id": "toolu_1", "name": "run_terminal_cmd", "input": map[string]any{"command": "curl ifconfig.me"}},
			}},
			// Cursor multi-turn: tool_result goes straight into the user content array (native Anthropic format)
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "toolu_1", "content": "1.2.3.4"},
			}},
		},
	}
	msgs := msgsOf(t, body)
	types := blockTypes(msgs)
	if !has(types, "assistant:tool_use") {
		t.Fatalf("assistant tool_use lost: %v", types)
	}
	if !has(types, "user:tool_result") {
		t.Fatalf("user tool_result dropped (this is the root cause of broken round-trips): %v", types)
	}
}

// The role:"tool" path (standard OpenAI format) must also convert correctly into tool_result.
func TestRoleToolPath(t *testing.T) {
	body := map[string]any{
		"model": "cursor-claude-opus-4-8",
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "tool_calls": []any{
				map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "get_ip", "arguments": "{}"}},
			}},
			map[string]any{"role": "tool", "tool_call_id": "call_1", "content": "5.6.7.8"},
		},
	}
	msgs := msgsOf(t, body)
	types := blockTypes(msgs)
	if !has(types, "assistant:tool_use") || !has(types, "user:tool_result") {
		t.Fatalf("role:tool conversion failed: %v", types)
	}
}

// Multiple tool_results must be merged into a single user message (Anthropic requires it).
func TestMultipleToolResultsMerge(t *testing.T) {
	body := map[string]any{
		"model": "cursor-claude-opus-4-8",
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "tool_calls": []any{
				map[string]any{"id": "c1", "type": "function", "function": map[string]any{"name": "a", "arguments": "{}"}},
				map[string]any{"id": "c2", "type": "function", "function": map[string]any{"name": "b", "arguments": "{}"}},
			}},
			map[string]any{"role": "tool", "tool_call_id": "c1", "content": "r1"},
			map[string]any{"role": "tool", "tool_call_id": "c2", "content": "r2"},
		},
	}
	msgs := msgsOf(t, body)
	// the last user message must contain 2 tool_results
	last := msgs[len(msgs)-1]
	if last["role"] != "user" {
		t.Fatalf("last message should be user, got %v", last["role"])
	}
	n := 0
	for _, bv := range last["content"].([]any) {
		if b := bv.(map[string]any); b["type"] == "tool_result" {
			n++
		}
	}
	if n != 2 {
		t.Fatalf("the two tool_results should merge into one user message, got %d", n)
	}
}

// Model name -<effort> suffix -> strip suffix + inject thinking/effort; no suffix = no injection.
func TestEffortSuffix(t *testing.T) {
	cases := []struct {
		in        string
		wantModel string
		wantEff   string // "" = should have no thinking/output_config
	}{
		{"cursor-claude-opus-4-8", "claude-opus-4-8", ""},
		{"cursor-claude-opus-4-8-xhigh", "claude-opus-4-8", "xhigh"},
		{"cursor-claude-opus-4-8-high", "claude-opus-4-8", "high"},
		{"cursor-claude-opus-4-8-medium", "claude-opus-4-8", "medium"},
		{"cursor-claude-opus-4-8-low", "claude-opus-4-8", "low"},
		{"cursor-claude-opus-4-8-max", "claude-opus-4-8", "max"},
	}
	for _, c := range cases {
		out, _ := toAnthropic(map[string]any{
			"model":    c.in,
			"messages": []any{map[string]any{"role": "user", "content": "hi"}},
		})
		if out["model"] != c.wantModel {
			t.Errorf("%s: model=%v want %s", c.in, out["model"], c.wantModel)
		}
		if c.wantEff == "" {
			if _, ok := out["thinking"]; ok {
				t.Errorf("%s: should not inject thinking", c.in)
			}
			if _, ok := out["output_config"]; ok {
				t.Errorf("%s: should not inject output_config", c.in)
			}
			continue
		}
		th, _ := out["thinking"].(map[string]any)
		if th == nil || th["type"] != "adaptive" {
			t.Errorf("%s: thinking should be {type:adaptive}, got %v", c.in, out["thinking"])
		}
		oc, _ := out["output_config"].(map[string]any)
		if oc == nil || oc["effort"] != c.wantEff {
			t.Errorf("%s: effort should be %s, got %v", c.in, c.wantEff, out["output_config"])
		}
	}
}

func TestFirstMustBeUserAndSkipEmpty(t *testing.T) {
	body := map[string]any{
		"model": "cursor-claude-opus-4-8",
		"messages": []any{
			map[string]any{"role": "assistant", "content": "leading assistant"},
			map[string]any{"role": "user", "content": "   "}, // pure whitespace, should be skipped
			map[string]any{"role": "user", "content": "real question"},
		},
	}
	msgs := msgsOf(t, body)
	if msgs[0]["role"] != "user" {
		t.Fatalf("first message must be user, got %v", msgs[0]["role"])
	}
}
