package main

// stream.go — Anthropic → OpenAI response conversion (streaming + non-streaming).

import (
	"bufio"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

func toOpenAIResponse(data map[string]any, model string, recoverName func(string) string) map[string]any {
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
					"function": map[string]any{"name": recoverName(name), "arguments": string(args)},
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

func streamToOpenAI(w http.ResponseWriter, body io.Reader, model string, recoverName func(string) string) {
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
					"function": map[string]any{"name": recoverName(name), "arguments": ""},
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
