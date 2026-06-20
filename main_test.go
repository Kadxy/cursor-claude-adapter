package main

import (
	"encoding/json"
	"testing"
)

// 把 toAnthropic 输出里所有 content block 的 type 收集出来,方便断言。
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

// 核心回归:Cursor 把 tool_result 当原生 Anthropic block 塞进 user content 数组。
// 修复前会被丢弃,修复后必须透传到上游。
func TestEmbeddedToolResultSurvives(t *testing.T) {
	body := map[string]any{
		"model": "cursor-claude-opus-4-8",
		"messages": []any{
			map[string]any{"role": "user", "content": "what's my ip"},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "tool_use", "id": "toolu_1", "name": "run_terminal_cmd", "input": map[string]any{"command": "curl ifconfig.me"}},
			}},
			// Cursor 多轮:tool_result 直接塞进 user content 数组(原生 Anthropic 格式)
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "toolu_1", "content": "1.2.3.4"},
			}},
		},
	}
	msgs := msgsOf(t, body)
	types := blockTypes(msgs)
	if !has(types, "assistant:tool_use") {
		t.Fatalf("assistant tool_use 丢失: %v", types)
	}
	if !has(types, "user:tool_result") {
		t.Fatalf("user tool_result 被丢弃(这就是 round-trip 崩的根因): %v", types)
	}
}

// role:"tool" 路径(OpenAI 标准格式)也要正确转成 tool_result。
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
		t.Fatalf("role:tool 转换失败: %v", types)
	}
}

// 多个 tool_result 必须合并进同一个 user 消息(Anthropic 要求)。
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
	// 最后一条 user 必须含 2 个 tool_result
	last := msgs[len(msgs)-1]
	if last["role"] != "user" {
		t.Fatalf("末条应为 user, got %v", last["role"])
	}
	n := 0
	for _, bv := range last["content"].([]any) {
		if b := bv.(map[string]any); b["type"] == "tool_result" {
			n++
		}
	}
	if n != 2 {
		t.Fatalf("两个 tool_result 应合并进一个 user 消息, got %d", n)
	}
}

// 模型名 -<effort> 后缀 → 去后缀 + 注入 thinking/effort;无后缀 = 不注入。
func TestEffortSuffix(t *testing.T) {
	cases := []struct {
		in        string
		wantModel string
		wantEff   string // "" = 不应有 thinking/output_config
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
				t.Errorf("%s: 不该注入 thinking", c.in)
			}
			if _, ok := out["output_config"]; ok {
				t.Errorf("%s: 不该注入 output_config", c.in)
			}
			continue
		}
		th, _ := out["thinking"].(map[string]any)
		if th == nil || th["type"] != "adaptive" {
			t.Errorf("%s: thinking 应为 {type:adaptive}, got %v", c.in, out["thinking"])
		}
		oc, _ := out["output_config"].(map[string]any)
		if oc == nil || oc["effort"] != c.wantEff {
			t.Errorf("%s: effort 应为 %s, got %v", c.in, c.wantEff, out["output_config"])
		}
	}
}

func TestFirstMustBeUserAndSkipEmpty(t *testing.T) {
	body := map[string]any{
		"model": "cursor-claude-opus-4-8",
		"messages": []any{
			map[string]any{"role": "assistant", "content": "leading assistant"},
			map[string]any{"role": "user", "content": "   "}, // 纯空白,应跳过
			map[string]any{"role": "user", "content": "real question"},
		},
	}
	msgs := msgsOf(t, body)
	if msgs[0]["role"] != "user" {
		t.Fatalf("首条必须是 user, got %v", msgs[0]["role"])
	}
}
