package main

// util.go — pure helpers (stateless, no business logic).
// Pulled out of main.go to reduce noise there; this holds only generic utilities
// unrelated to the specific forward/convert flow.

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"log"
	"os"
	"strconv"
	"strings"
)

// env reads an environment variable, falling back to a default when empty.
func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

// mapStop maps an Anthropic stop_reason to an OpenAI finish_reason; unknown returns "".
func mapStop(r string) string {
	switch r {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	}
	return ""
}

// contentToText flattens any content (string / block array / nil / other) to plain text.
func contentToText(c any) string {
	switch v := c.(type) {
	case string:
		return v
	case []any:
		var parts []string
		for _, pv := range v {
			if p, ok := pv.(map[string]any); ok {
				if t, ok := p["text"].(string); ok {
					parts = append(parts, t)
				}
			} else if s, ok := pv.(string); ok {
				parts = append(parts, s)
			}
		}
		return strings.Join(parts, "\n")
	case nil:
		return ""
	}
	b, _ := json.Marshal(c)
	return string(b)
}

// stripCacheTTL removes the ttl inside a block's cache_control (extended cache needs a beta
// header, otherwise 400), but keeps cache_control itself (the standard ephemeral cache
// breakpoint, which is GA and needs no beta header).
func stripCacheTTL(block map[string]any) {
	if cc, ok := block["cache_control"].(map[string]any); ok {
		delete(cc, "ttl")
	}
}

// dataURIToImage converts a data:<mime>;base64,<data> URL into an Anthropic image block;
// returns nil if malformed.
func dataURIToImage(url string) map[string]any {
	if !strings.HasPrefix(url, "data:") {
		return nil
	}
	i := strings.Index(url, ";base64,")
	if i < 0 {
		return nil
	}
	mt := url[5:i]
	data := url[i+8:]
	return map[string]any{"type": "image", "source": map[string]any{"type": "base64", "media_type": mt, "data": data}}
}

// normName normalizes a tool name: lowercase and keep only alphanumerics (used for prefix
// matching during tool-name recovery).
func normName(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func numField(m map[string]any, k string) (int, bool) {
	if v, ok := m[k]; ok {
		return numFromAny(v)
	}
	return 0, false
}
func numFromAny(v any) (int, bool) {
	if f, ok := v.(float64); ok {
		return int(f), true
	}
	return 0, false
}
func usageInt(m map[string]any, k string) int {
	if m == nil {
		return 0
	}
	n, _ := numFromAny(m[k])
	return n
}
func logUsage(tag string, u any) {
	if m, ok := u.(map[string]any); ok {
		log.Printf("%s usage in=%d out=%d cacheRead=%d cacheCreate=%d", tag,
			usageInt(m, "input_tokens"), usageInt(m, "output_tokens"),
			usageInt(m, "cache_read_input_tokens"), usageInt(m, "cache_creation_input_tokens"))
	}
}
func truncate(b []byte, n int) string {
	if len(b) <= n {
		return string(b)
	}
	return string(b[:n]) + "...(" + strconv.Itoa(len(b)) + "B)"
}
func mapField(m map[string]any, k string) map[string]any {
	if v, ok := m[k].(map[string]any); ok {
		return v
	}
	return nil
}
func strField(m map[string]any, k string) string { s, _ := m[k].(string); return s }
func strOr(v any, d string) string {
	if s, ok := v.(string); ok && s != "" {
		return s
	}
	return d
}
func randID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return hex.EncodeToString(b)
}
