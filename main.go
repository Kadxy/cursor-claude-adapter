// cursor-claude-adapter lets Cursor BYOK use your own Anthropic-format relay.
//
// Flow: Cursor (OpenAI Chat Completions) -> this adapter converts to Anthropic
// Messages -> your upstream relay (/v1/messages) -> the Anthropic response
// (stream or not) is converted back to OpenAI -> Cursor.
//
// Cursor only lets you override the OpenAI Base URL, so the adapter takes OpenAI
// in and produces Anthropic out. It targets /v1/messages (not chat/completions)
// to keep Anthropic prompt caching and to build clean tool-call chunks itself,
// so multi-turn tool_result round-trips don't break.
//
// Config is via env vars (all have defaults); see README.md for the full setup,
// model list, thinking levels, and deployment notes.
//
// No third-party dependencies: main.go holds entry/convert/forward, util.go the
// pure helpers. go.mod exists only for the Go module declaration.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

var (
	upstreamURL  = strings.TrimRight(env("UPSTREAM_URL", "https://api.anthropic.com"), "/") + "/v1/messages"
	modelPrefix  = env("MODEL_PREFIX", "cursor-")
	anthVersion  = env("ANTHROPIC_VERSION", "2023-06-01")
	port         = env("PORT", "3000")
	defMaxTokens = 8192
	debug        = os.Getenv("DEBUG") == "1"

	// baseModels: the real models exposed to Cursor (names after the prefix is stripped).
	// Add a line here when a new model ships.
	baseModels = []string{"claude-opus-4-8", "claude-opus-4-7"}
	// effortOrder: thinking levels. Defines both the valid values and the display order
	// of the variants in /v1/models.
	effortOrder = []string{"low", "medium", "high", "xhigh", "max"}

	httpClient  = &http.Client{Timeout: 10 * time.Minute}
	retryStatus = map[int]bool{429: true, 500: true, 502: true, 503: true, 504: true, 524: true}
	retryDelays = []time.Duration{800 * time.Millisecond, 2 * time.Second}
)

// effortLevels is derived from effortOrder; used by splitEffort for validation.
var effortLevels = func() map[string]bool {
	m := map[string]bool{}
	for _, e := range effortOrder {
		m[e] = true
	}
	return m
}()

func main() {
	http.HandleFunc("/v1/chat/completions", handleChat)
	http.HandleFunc("/v1/responses", handleChat)
	http.HandleFunc("/v1/models", handleModels)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, map[string]any{"status": "ok"}) })
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("cursor-claude-adapter (go, messages) ok"))
	})
	log.Printf("cursor-claude-adapter (go, /v1/messages) on :%s -> %s", port, upstreamURL)
	log.Fatal(http.ListenAndServe(":"+port, nil))
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, 405, "method not allowed")
		return
	}
	key := bearer(r)
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		writeErr(w, 400, "read body failed")
		return
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		writeErr(w, 400, "invalid json")
		return
	}

	anth, toolNames := toAnthropic(body)
	stream, _ := anth["stream"].(bool)
	model, _ := body["model"].(string)
	recover := makeRecover(toolNames)

	// Request summary: toolResults>0 means Cursor sent tool results back (multi-turn).
	nMsgs, nToolRes := 0, 0
	if msgs, ok := body["messages"].([]any); ok {
		nMsgs = len(msgs)
		for _, mm := range msgs {
			if m, ok := mm.(map[string]any); ok && m["role"] == "tool" {
				nToolRes++
			}
		}
	}
	log.Printf("REQ  model=%v->%v stream=%v msgs=%d toolResults=%d tools=%d toolChoice=%v",
		model, anth["model"], stream, nMsgs, nToolRes, len(toolNames), body["tool_choice"])
	out, _ := json.Marshal(anth)
	if debug {
		log.Printf("DBG  anthReq=%s", truncate(out, 2000))
	}

	resp, status, errBody := forward(out, key, stream)
	if resp == nil {
		log.Printf("ERR  upstream status=%d body=%s", status, truncate(errBody, 400))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write(errBody)
		return
	}
	defer resp.Body.Close()

	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(200)
		streamToOpenAI(w, resp.Body, model, recover)
		return
	}
	var data map[string]any
	if b, _ := io.ReadAll(resp.Body); json.Unmarshal(b, &data) != nil {
		writeErr(w, 502, "bad upstream response")
		return
	}
	logUsage("RESP json", data["usage"])
	writeJSON(w, 200, toOpenAIResponse(data, model, recover))
}

// ─────────────────── OpenAI → Anthropic request conversion ───────────────────

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

// ─────────────────── Anthropic → OpenAI response conversion ───────────────────

func toOpenAIResponse(data map[string]any, model string, recover func(string) string) map[string]any {
	var text strings.Builder
	var toolCalls []any
	if content, ok := data["content"].([]any); ok {
		for _, bv := range content {
			b, ok := bv.(map[string]any)
			if !ok {
				continue
			}
			switch b["type"] {
			case "text":
				if t, ok := b["text"].(string); ok {
					text.WriteString(t)
				}
			case "tool_use":
				name, _ := b["name"].(string)
				args, _ := json.Marshal(b["input"])
				toolCalls = append(toolCalls, map[string]any{
					"id": b["id"], "type": "function",
					"function": map[string]any{"name": recover(name), "arguments": string(args)},
				})
			}
		}
	}
	msg := map[string]any{"role": "assistant"}
	if text.Len() > 0 {
		msg["content"] = text.String()
	} else {
		msg["content"] = nil
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
	}
	finish := "stop"
	if sr, ok := data["stop_reason"].(string); ok {
		if m := mapStop(sr); m != "" {
			finish = m
		}
	}
	resp := map[string]any{
		"id": strOr(data["id"], "chatcmpl-"+randID()), "object": "chat.completion",
		"created": time.Now().Unix(), "model": model,
		"choices": []any{map[string]any{"index": 0, "message": msg, "finish_reason": finish}},
	}
	if u, ok := data["usage"].(map[string]any); ok {
		in, _ := numFromAny(u["input_tokens"])
		o, _ := numFromAny(u["output_tokens"])
		resp["usage"] = map[string]any{"prompt_tokens": in, "completion_tokens": o, "total_tokens": in + o}
	}
	return resp
}

func streamToOpenAI(w http.ResponseWriter, body io.Reader, model string, recover func(string) string) {
	flusher, _ := w.(http.Flusher)
	id := "chatcmpl-" + randID()
	created := time.Now().Unix()
	emit := func(delta map[string]any, finish any) {
		chunk := map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
			"choices": []any{map[string]any{"index": 0, "delta": delta, "finish_reason": finish}}}
		b, _ := json.Marshal(chunk)
		w.Write([]byte("data: " + string(b) + "\n\n"))
		if flusher != nil {
			flusher.Flush()
		}
	}
	emit(map[string]any{"role": "assistant", "content": ""}, nil)

	toolIndex := -1
	blockToTool := map[int]int{}
	var finish any = nil
	var usageStart, usageDelta map[string]any
	textLen := 0

	sc := bufio.NewScanner(body)
	sc.Buffer(make([]byte, 1024*1024), 16*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		ds := strings.TrimSpace(line[5:])
		if ds == "" {
			continue
		}
		var d map[string]any
		if json.Unmarshal([]byte(ds), &d) != nil {
			continue
		}
		switch d["type"] {
		case "content_block_start":
			cb, _ := d["content_block"].(map[string]any)
			if cb != nil && cb["type"] == "tool_use" {
				toolIndex++
				if idx, ok := numFromAny(d["index"]); ok {
					blockToTool[idx] = toolIndex
				}
				name, _ := cb["name"].(string)
				emit(map[string]any{"tool_calls": []any{map[string]any{
					"index": toolIndex, "id": cb["id"], "type": "function",
					"function": map[string]any{"name": recover(name), "arguments": ""},
				}}}, nil)
			}
		case "content_block_delta":
			delta, _ := d["delta"].(map[string]any)
			if delta == nil {
				continue
			}
			switch delta["type"] {
			case "text_delta":
				if t, ok := delta["text"].(string); ok {
					textLen += len(t)
					emit(map[string]any{"content": t}, nil)
				}
			case "input_json_delta":
				idx, _ := numFromAny(d["index"])
				pj, _ := delta["partial_json"].(string)
				emit(map[string]any{"tool_calls": []any{map[string]any{
					"index": blockToTool[idx], "function": map[string]any{"arguments": pj},
				}}}, nil)
			}
		case "message_start":
			if msg, ok := d["message"].(map[string]any); ok {
				usageStart, _ = msg["usage"].(map[string]any)
			}
		case "message_delta":
			if u, ok := d["usage"].(map[string]any); ok {
				usageDelta = u
			}
			if delta, ok := d["delta"].(map[string]any); ok {
				if sr, ok := delta["stop_reason"].(string); ok {
					if m := mapStop(sr); m != "" {
						finish = m
					}
				}
			}
		case "message_stop":
			if finish == nil {
				finish = "stop"
			}
			emit(map[string]any{}, finish)
		}
	}
	w.Write([]byte("data: [DONE]\n\n"))
	if flusher != nil {
		flusher.Flush()
	}
	log.Printf("RESP stream text=%dchars toolCalls=%d finish=%v | usage in=%d out=%d cacheRead=%d cacheCreate=%d",
		textLen, toolIndex+1, finish,
		usageInt(usageStart, "input_tokens"), usageInt(usageDelta, "output_tokens"),
		usageInt(usageStart, "cache_read_input_tokens"), usageInt(usageStart, "cache_creation_input_tokens"))
}

// ─────────────────── Forwarding (with retry) ───────────────────

func forward(body []byte, key string, stream bool) (*http.Response, int, []byte) {
	lastStatus, lastBody := 502, []byte(`{"error":{"message":"upstream unreachable"}}`)
	attempts := 1 + len(retryDelays)
	for attempt := 0; attempt < attempts; attempt++ {
		req, _ := http.NewRequest(http.MethodPost, upstreamURL, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", key)
		req.Header.Set("anthropic-version", anthVersion)
		if stream {
			req.Header.Set("Accept", "text/event-stream")
		}
		resp, err := httpClient.Do(req)
		if err != nil {
			lastStatus, lastBody = 502, []byte(err.Error())
			sleep(attempt)
			continue
		}
		if resp.StatusCode < 400 {
			return resp, 200, nil
		}
		eb, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		lastStatus, lastBody = resp.StatusCode, eb
		if !shouldRetry(resp.StatusCode, eb) {
			break
		}
		log.Printf("upstream %d, retry %d/%d", resp.StatusCode, attempt+1, attempts)
		sleep(attempt)
	}
	return nil, lastStatus, lastBody
}

func shouldRetry(status int, body []byte) bool {
	if retryStatus[status] {
		return true
	}
	s := strings.ToLower(string(body))
	// Retry on upstream "no channel available" / quota errors. The first literal is the
	// Chinese phrase some relays return in the body — it is matched text, not a translatable comment.
	for _, kw := range []string{"无可用渠道", "no available channel", "quota", "insufficient"} {
		if strings.Contains(s, kw) {
			return true
		}
	}
	return false
}

func sleep(attempt int) {
	if attempt < len(retryDelays) {
		time.Sleep(retryDelays[attempt])
	}
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

// ─────────────────── Misc ───────────────────

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

// handleModels expands the model list dynamically: each baseModel yields a "no thinking"
// variant plus one variant per effort level, all with modelPrefix prepended. No config
// needed; to add a model, just edit baseModels.
// e.g. claude-opus-4-8 -> cursor-claude-opus-4-8, cursor-claude-opus-4-8-low ... -max
func handleModels(w http.ResponseWriter, r *http.Request) {
	var data []any
	add := func(id string) {
		data = append(data, map[string]any{"id": id, "object": "model", "created": 0, "owned_by": "adapter"})
	}
	for _, base := range baseModels {
		add(modelPrefix + base)
		for _, eff := range effortOrder {
			add(modelPrefix + base + "-" + eff)
		}
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data})
}
func bearer(r *http.Request) string {
	a := r.Header.Get("Authorization")
	if len(a) > 7 && strings.EqualFold(a[:7], "Bearer ") {
		return strings.TrimSpace(a[7:])
	}
	if k := r.Header.Get("x-api-key"); k != "" {
		return k
	}
	return strings.TrimSpace(a)
}
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}
func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": msg}})
}
