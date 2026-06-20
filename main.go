// cursor-claude-adapter — 让 Cursor BYOK 用上你自己的 Anthropic 格式中转。
//
// 数据流:
//
//	Cursor(OpenAI Chat Completions 格式) --[override OpenAI Base URL]--> 本程序
//	  --> 转成 Anthropic Messages --> 你的上游中转(/v1/messages)
//	  --> Anthropic 响应(流式/非流式)转回 OpenAI 格式 --> Cursor
//
// 为什么是这个形态:
//   - Cursor BYOK 没有 Anthropic Base URL 覆写,唯一能改 URL 的是 Override OpenAI
//     Base URL,它发的是标准 Chat Completions。所以只能 OpenAI 进、Anthropic 出。
//   - Cursor 由服务端发请求,本程序必须公网可达 + HTTPS(cloudflared 打洞即可)。
//   - 走 /v1/messages 而非 /v1/chat/completions:只有 messages 路径保留 Anthropic
//     prompt cache(省钱),且 tool_call 分片由本程序生成、干净,多轮 tool_result 才不崩。
//
// 核心坑(convertParts):Cursor 多轮工具时,会把 tool_use / tool_result 当原生
// Anthropic block 直接塞进 message 的 content 数组,必须原样透传,否则工具结果被丢弃。
//
// 环境变量(均有默认值,生产配在 .env / docker-compose):
//
//	UPSTREAM_URL       上游 Anthropic /v1/messages 全路径(默认 http://152.53.52.170:3003/v1/messages)
//	MODEL_PREFIX       Cursor 模型名前缀,转发前去掉(默认 cursor-;cursor-claude-opus-4-8 → claude-opus-4-8)
//	MODELS             /v1/models 暴露给 Cursor 的模型名,逗号分隔(默认 cursor-claude-opus-4-8)
//	ANTHROPIC_VERSION  anthropic-version 头(默认 2023-06-01)
//	PORT               监听端口(默认 3000;docker 里容器固定 3000,对外端口看 compose)
//	DEBUG              =1 打印进出报文摘要
//
// 思考等级:模型名加后缀 -low/-medium/-high/-xhigh/-max,即注入 thinking:adaptive +
// output_config.effort 开启对应等级的思考(xhigh 是 Claude Code 默认、最适合编码)。
// 例:cursor-claude-opus-4-8-xhigh → claude-opus-4-8 + 思考;无后缀 = 不思考。
// 把这些变体名都填进 MODELS,Cursor 模型列表里就能选。
//
// 鉴权:key 透传 —— Cursor 填的 OpenAI API Key 原样转成上游 x-api-key,本程序不存 key。
//
// Cursor 配置:Settings → Models → Override OpenAI Base URL = https://你的域名/v1;
// OpenAI API Key = 你上游中转的 key;自定义模型名用 MODELS 里的;关掉 Anthropic API Key 栏。
//
// 部署:docker compose up -d --build  (HTTPS:cloudflared tunnel --url http://localhost:<PORT>)
//
// 单文件、零第三方依赖。go.mod 仅为 Go 构建必需(module 声明)。
package main

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

func env(k, d string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return d
}

var (
	upstreamURL  = env("UPSTREAM_URL", "http://152.53.52.170:3003/v1/messages")
	modelPrefix  = env("MODEL_PREFIX", "cursor-")
	anthVersion  = env("ANTHROPIC_VERSION", "2023-06-01")
	port         = env("PORT", "3000")
	models       = strings.Split(env("MODELS", "cursor-claude-opus-4-8"), ",")
	defMaxTokens = 8192
	debug        = os.Getenv("DEBUG") == "1"

	httpClient   = &http.Client{Timeout: 10 * time.Minute}
	retryStatus  = map[int]bool{429: true, 500: true, 502: true, 503: true, 504: true, 524: true}
	retryDelays  = []time.Duration{800 * time.Millisecond, 2 * time.Second}
	effortLevels = map[string]bool{"low": true, "medium": true, "high": true, "xhigh": true, "max": true}
)

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

	// 请求摘要:toolResults>0 = Cursor 把工具结果发回来了(多轮)
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

// ─────────────────── OpenAI → Anthropic 请求转换 ───────────────────

func toAnthropic(body map[string]any) (map[string]any, []string) {
	out := map[string]any{}
	model, _ := body["model"].(string)
	realModel, effort := splitEffort(strings.TrimPrefix(model, modelPrefix))
	out["model"] = realModel
	// 模型名带 -<effort> 后缀(low/medium/high/xhigh/max)= 开启自适应思考并设思考等级。
	// 例:cursor-claude-opus-4-8-xhigh → claude-opus-4-8 + thinking:adaptive + effort:xhigh
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

	// messages → system + 归一化消息序列(再合并相邻同 role)
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
				continue // 空 assistant 消息跳过(避免空 text block 触发 400)
			}
			seq = append(seq, rmsg{"assistant", c})
		default: // user
			c := convertParts(m["content"])
			if len(c) == 0 {
				continue // 跳过空 user 消息
			}
			seq = append(seq, rmsg{"user", c})
		}
	}
	// 合并相邻同 role
	var merged []map[string]any
	for _, m := range seq {
		if n := len(merged); n > 0 && merged[n-1]["role"] == m.role {
			merged[n-1]["content"] = append(merged[n-1]["content"].([]any), m.content...)
		} else {
			merged = append(merged, map[string]any{"role": m.role, "content": append([]any{}, m.content...)})
		}
	}
	// Anthropic 要求:至少一条消息,且首条必须是 user
	if len(merged) == 0 {
		merged = []map[string]any{{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Continue."}}}}
	} else if merged[0]["role"] != "user" {
		merged = append([]map[string]any{{"role": "user", "content": []any{map[string]any{"type": "text", "text": "Continue."}}}}, merged...)
	}
	out["messages"] = merged
	if len(systemParts) > 0 {
		out["system"] = strings.Join(systemParts, "\n\n")
	}

	// tools(兼容 flat / nested)
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

// ─────────────────── Anthropic → OpenAI 响应转换 ───────────────────

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

// ─────────────────── 转发(带重试) ───────────────────

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

// ─────────────────── 工具名还原 ───────────────────
// 后端非流式会把工具名改成 Compat<PascalName><hash>。用请求里的真实工具名反查还原。
func makeRecover(names []string) func(string) string {
	exact := map[string]bool{}
	type c struct{ norm, orig string }
	var cands []c
	for _, n := range names {
		exact[n] = true
		cands = append(cands, c{normName(n), n})
	}
	// 长前缀优先
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

func normName(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// ─────────────────── 杂项 ───────────────────

// splitEffort 从去前缀后的模型名末尾切出思考等级后缀。
// "claude-opus-4-8-xhigh" → ("claude-opus-4-8", "xhigh");"claude-opus-4-8" → (原样, "")
func splitEffort(model string) (string, string) {
	if i := strings.LastIndex(model, "-"); i >= 0 && effortLevels[model[i+1:]] {
		return model[:i], model[i+1:]
	}
	return model, ""
}

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

// convertParts 把一条消息的 content 归一化为 Anthropic content blocks。
// 关键:Cursor 在多轮工具里,会把 tool_use / tool_result(以及 image)直接以
// 原生 Anthropic block 形式塞进 content 数组(ungate 源码注释:"Cursor sends these"),
// 必须原样透传,否则工具结果会被丢弃 → 多轮 round-trip 崩。
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
			stripCacheTTL(p)           // 保留 cache_control(缓存断点),删掉需要 beta 头的 ttl
			blocks = append(blocks, p) // 原生 Anthropic block,透传
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

// stripCacheTTL 删除 block 上 cache_control 里的 ttl(扩展缓存需 beta 头,否则 400),
// 但保留 cache_control 本身(标准 ephemeral 缓存断点,GA,无需 beta)。
func stripCacheTTL(block map[string]any) {
	if cc, ok := block["cache_control"].(map[string]any); ok {
		delete(cc, "ttl")
	}
}

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

func handleModels(w http.ResponseWriter, r *http.Request) {
	var data []any
	for _, m := range models {
		data = append(data, map[string]any{"id": strings.TrimSpace(m), "object": "model", "created": 0, "owned_by": "adapter"})
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
